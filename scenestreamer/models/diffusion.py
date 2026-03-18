from typing import Any, Mapping, Tuple, Union
from typing import Callable
from typing import Dict, Optional, Sequence

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from einops import rearrange
from jax import Array
from jax.typing import ArrayLike

# adapted from https://github.com/google-research/vision_transformer/blob/main/vit_jax/models_vit.py
# from octo.model.components.base import TokenGroup
# from octo.utils.typing import Dtype, PRNGKey, Shape, Union

# from octo.model.components.base import TokenGroup
# from octo.model.components.diffusion import cosine_beta_schedule, create_diffusion_model
# from octo.model.components.tokenizers import BinTokenizer
# from octo.model.components.transformer import MAPHead
# from octo.utils.typing import PRNGKey

PRNGKey = jax.random.KeyArray
PyTree = Union[jax.typing.ArrayLike, Mapping[str, "PyTree"]]
Config = Union[Any, Mapping[str, "Config"]]
Params = Mapping[str, PyTree]
Data = Mapping[str, PyTree]
Shape = Sequence[int]
Dtype = jax.typing.DTypeLike

default_init = nn.initializers.xavier_uniform


@flax.struct.dataclass
class TokenGroup:
    """A group of tokens that have semantic meaning together (e.g. the tokens for a single observation)

    Attributes:
        tokens: jax.Array of shape (..., n_tokens, token_dim)
        mask: jax.Array of shape (..., n_tokens) indicating which tokens are valid (1) vs padding (0)
    """

    tokens: jax.typing.ArrayLike
    mask: jax.typing.ArrayLike

    @classmethod
    def create(cls, tokens: jax.typing.ArrayLike, mask: jax.typing.ArrayLike = None, **kwargs):
        if mask is None:
            mask = jnp.ones(tokens.shape[:-1])
        assert mask.ndim == tokens.ndim - 1
        return cls(tokens, mask, **kwargs)

    @classmethod
    def concatenate(cls, group_list: Sequence["TokenGroup"], axis=-2):
        data = jnp.concatenate([t.tokens for t in group_list], axis=axis)
        mask = jnp.concatenate([t.mask for t in group_list], axis=axis + 1)
        return cls(data, mask)


def masked_mean(x, mask):
    mask = jnp.broadcast_to(mask, x.shape)
    return jnp.mean(x * mask) / jnp.clip(jnp.mean(mask), a_min=1e-5, a_max=None)


def continuous_loss(
    pred_value: ArrayLike,
    ground_truth_value: ArrayLike,
    mask: ArrayLike,
    loss_type: str = "mse",
) -> Array:
    """
    Args:
        pred_value: shape (batch_dims...)
        ground_truth_value: continuous values w/ shape (batch_dims...)
        mask: broadcastable to ground_truth
    """
    if loss_type == "mse":
        loss = jnp.square(pred_value - ground_truth_value)
    elif loss_type == "l1":
        loss = jnp.abs(pred_value - ground_truth_value)
    else:
        raise ValueError(f"Invalid loss type: {loss_type}")

    loss = masked_mean(loss, mask)

    mse = jnp.square(pred_value - ground_truth_value)
    mse = masked_mean(mse, mask)
    return loss, {
        "loss": loss,
        "mse": mse,
    }


def chunk_actions(actions: ArrayLike, pred_horizon: int) -> Array:
    """Chunk actions for predicting actions `pred_horizon` steps into the future.

    The resulting actions have shape (batch, actions.shape[-2] - (pred_horizon - 1), pred_horizon, action_dim)

    For example: chunk_actions([a_1, a_2, a_3, a_4, a_5], 3) ->
        [
            [a_1, a_2, a_3],
            [a_2, a_3, a_4],
            [a_3, a_4, a_5],
        ]

    """
    assert (
        actions.ndim == 3
    ), f"Expected actions to have shape (batch, window_size, action_dim), but got shape {actions.shape}"
    window_size = actions.shape[1]
    assert (window_size >= pred_horizon), f"pred_horizon {pred_horizon} too large for window size {window_size}"
    chunk_window_size = window_size - (pred_horizon - 1)

    curr_step = jnp.arange(chunk_window_size)
    action_offset = jnp.arange(pred_horizon)
    chunk_indices = curr_step[:, None] + action_offset[None, :]
    return actions[:, chunk_indices]


def _check_action_window_size(actions, window_size, pred_horizon):
    assert (
        actions.shape[1] >= window_size + pred_horizon - 1
    ), f"""
        To predict actions for window_size {window_size} and future prediction horizon {pred_horizon},
        the ground-truth actions must have at least {window_size + pred_horizon - 1} timesteps, but got shape {actions.shape}.

        Did you make sure to set "future_action_window_size" correctly in the data config?
    """


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = jnp.linspace(0, timesteps, steps) / timesteps
    alphas_cumprod = jnp.cos((t + s) / (1 + s) * jnp.pi * 0.5)**2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return jnp.clip(betas, 0, 0.999)


class ScoreActor(nn.Module):
    time_preprocess: nn.Module
    cond_encoder: nn.Module
    reverse_network: nn.Module

    def __call__(self, obs_enc, actions, time, train=False):
        t_ff = self.time_preprocess(time)
        cond_enc = self.cond_encoder(t_ff, train=train)
        reverse_input = jnp.concatenate([cond_enc, obs_enc, actions], axis=-1)
        eps_pred = self.reverse_network(reverse_input, train=train)
        return eps_pred


class FourierFeatures(nn.Module):
    output_size: int
    learnable: bool = True

    @nn.compact
    def __call__(self, x: jax.Array):
        if self.learnable:
            w = self.param(
                "kernel",
                nn.initializers.normal(0.2),
                (self.output_size // 2, x.shape[-1]),
                jnp.float32,
            )
            f = 2 * jnp.pi * x @ w.T
        else:
            half_dim = self.output_size // 2
            f = jnp.log(10000) / (half_dim - 1)
            f = jnp.exp(jnp.arange(half_dim) * -f)
            f = x * f
        return jnp.concatenate([jnp.cos(f), jnp.sin(f)], axis=-1)


class MLP(nn.Module):
    hidden_dims: Sequence[int]
    activation: Callable = nn.swish
    activate_final: bool = False
    use_layer_norm: bool = False
    dropout_rate: Optional[float] = None

    @nn.compact
    def __call__(self, x: jax.Array, train: bool = False) -> jax.Array:
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=default_init())(x)

            if i + 1 < len(self.hidden_dims) or self.activate_final:
                if self.dropout_rate is not None and self.dropout_rate > 0:
                    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)
                if self.use_layer_norm:
                    x = nn.LayerNorm()(x)
                x = self.activation(x)
        return x


class MLPResNetBlock(nn.Module):
    features: int
    act: Callable
    dropout_rate: float = None
    use_layer_norm: bool = False

    @nn.compact
    def __call__(self, x, train: bool = False):
        residual = x
        if self.dropout_rate is not None and self.dropout_rate > 0:
            x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)
        if self.use_layer_norm:
            x = nn.LayerNorm()(x)
        x = nn.Dense(self.features * 4)(x)
        x = self.act(x)
        x = nn.Dense(self.features)(x)

        if residual.shape != x.shape:
            residual = nn.Dense(self.features)(residual)

        return residual + x


class MLPResNet(nn.Module):
    num_blocks: int
    out_dim: int
    dropout_rate: float = None
    use_layer_norm: bool = False
    hidden_dim: int = 256
    activation: Callable = nn.swish

    @nn.compact
    def __call__(self, x: jax.typing.ArrayLike, train: bool = False) -> jax.Array:
        x = nn.Dense(self.hidden_dim, kernel_init=default_init())(x)
        for _ in range(self.num_blocks):
            x = MLPResNetBlock(
                self.hidden_dim,
                act=self.activation,
                use_layer_norm=self.use_layer_norm,
                dropout_rate=self.dropout_rate,
            )(x, train=train)

        x = self.activation(x)
        x = nn.Dense(self.out_dim, kernel_init=default_init())(x)
        return x


def create_diffusion_model(
    out_dim: int,
    time_dim: int,
    num_blocks: int,
    dropout_rate: float,
    hidden_dim: int,
    use_layer_norm: bool,
):
    return ScoreActor(
        FourierFeatures(time_dim, learnable=True),
        MLP((2 * time_dim, time_dim)),
        MLPResNet(
            num_blocks,
            out_dim,
            dropout_rate=dropout_rate,
            hidden_dim=hidden_dim,
            use_layer_norm=use_layer_norm,
        ),
    )


class MlpBlock(nn.Module):
    """Transformer MLP / feed-forward block."""

    mlp_dim: int
    dtype: Dtype = jnp.float32
    out_dim: Optional[int] = None
    dropout_rate: float = 0.1
    kernel_init: Callable[[PRNGKey, Shape, Dtype], jax.Array] = nn.initializers.xavier_uniform()
    bias_init: Callable[[PRNGKey, Shape, Dtype], jax.Array] = nn.initializers.normal(stddev=1e-6)

    @nn.compact
    def __call__(self, inputs, *, deterministic):
        """Applies Transformer MlpBlock module."""
        actual_out_dim = inputs.shape[-1] if self.out_dim is None else self.out_dim
        x = nn.Dense(
            features=self.mlp_dim,
            dtype=self.dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )(inputs)
        x = nn.gelu(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
        output = nn.Dense(
            features=actual_out_dim,
            dtype=self.dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )(x)
        output = nn.Dropout(rate=self.dropout_rate)(output, deterministic=deterministic)
        return output


class MAPHead(nn.Module):
    """Multihead Attention Pooling.

    From https://github.com/google-research/big_vision/blob/main/big_vision/models/vit.py
    """

    mlp_dim: Optional[int] = None  # Defaults to 4x input dim
    num_heads: int = 8
    num_readouts: int = 1

    @nn.compact
    def __call__(self, x: Union[jax.Array, TokenGroup], train=True):
        if isinstance(x, TokenGroup):
            x, mask = x.tokens, x.mask
        else:
            mask = None

        *batch_dims, l, d = x.shape
        x = x.reshape(-1, l, d)
        batch_size = x.shape[0]

        probe = self.param(
            "probe",
            nn.initializers.xavier_uniform(),
            (1, self.num_readouts, d),
            x.dtype,
        )
        probe = jnp.tile(probe, [batch_size, 1, 1])

        if mask is not None:
            mask = mask.reshape(-1, l)
            mask = jnp.broadcast_to(mask[:, None, None, :], (batch_size, 1, self.num_readouts, l))

        out = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads, kernel_init=nn.initializers.xavier_uniform()
        )(probe, x, mask=mask)

        # TODO: dropout on head?
        y = nn.LayerNorm()(out)

        out = out + MlpBlock(mlp_dim=nn.merge_param("mlp_dim", self.mlp_dim, 4 * d))(y, deterministic=not train)
        out = out.reshape(*batch_dims, self.num_readouts, d)
        return out


class DiffusionActionHead(nn.Module):
    """Predicts actions uses a diffusion process.

    Only a single pass through the transformer is done to obtain an action embedding at each timestep. The
    action is then predicted using a diffusion process conditioned on this embedding. The diffusion model
    architecture is an MLP with residual connections (see `octo.model.components.diffusion`).

    You may create an embedding by either mean-pooling across tokens (use_map=False) or using multi-head
    attention pooling (use_map=True). It is recommended to use MAP when decoding from the observation token
    stream.
    """

    readout_key: str
    use_map: bool = False
    pred_horizon: int = 1
    action_dim: int = 7
    max_action: float = 5.0
    loss_type: str = "mse"

    # diffusion-specific config with sane defaults
    time_dim: int = 32
    num_blocks: int = 3
    dropout_rate: float = 0.1
    hidden_dim: int = 256
    use_layer_norm: bool = True
    diffusion_steps: int = 20

    def setup(self):
        if self.use_map:
            self.map_head = MAPHead()

        # create the diffusion model (score network)
        self.diffusion_model = create_diffusion_model(
            self.action_dim * self.pred_horizon,
            time_dim=self.time_dim,
            num_blocks=self.num_blocks,
            dropout_rate=self.dropout_rate,
            hidden_dim=self.hidden_dim,
            use_layer_norm=self.use_layer_norm,
        )

        # create beta schedule
        self.betas = jnp.array(cosine_beta_schedule(self.diffusion_steps))
        self.alphas = 1 - self.betas
        self.alpha_hats = jnp.array([jnp.prod(self.alphas[:i + 1]) for i in range(self.diffusion_steps)])

    def __call__(
        self,
        transformer_outputs: Dict[str, TokenGroup],
        time: Optional[ArrayLike] = None,
        noisy_actions: Optional[ArrayLike] = None,
        train: bool = True,
    ) -> jax.Array:
        """Performs a single forward pass through the diffusion model."""
        token_group = transformer_outputs[self.readout_key]
        assert token_group.tokens.ndim == 4, (
            f"Expected token_group.tokens to have shape (batch_size, window_size, num_tokens, embedding_size), "
            f"but got shape {token_group.tokens.shape}"
        )
        if self.use_map:  # Multi-head attention pooling
            embeddings = self.map_head(token_group, train=train)[:, :, 0]
        else:  # mean pooling
            embeddings = token_group.tokens.mean(axis=-2)
        # Now, embeddings is (batch_size, window_size, embedding_size)

        # time and noisy_actions are None during initialization, so we replace them with a dummy array
        if (time is None or noisy_actions is None) and not self.is_initializing():
            raise ValueError("Must provide time and noisy_actions when calling diffusion action head")
        elif self.is_initializing():
            time = jnp.zeros((*embeddings.shape[:2], 1), dtype=jnp.float32)
            noisy_actions = jnp.zeros(
                (*embeddings.shape[:2], self.action_dim * self.pred_horizon),
                dtype=jnp.float32,
            )

        pred_eps = self.diffusion_model(embeddings, noisy_actions, time, train=train)
        return pred_eps

    def loss(
        self,
        transformer_outputs: Dict[str, TokenGroup],
        actions: ArrayLike,
        pad_mask: ArrayLike,
        train: bool = True,
    ) -> Tuple[Array, Dict[str, Array]]:
        """Computes the loss for the diffusion objective.

        Args:
            transformer_ouputs: must contain self.readout_key with shape (batch_size, window_size, num_tokens,
                embedding_size)
            actions: shape (batch_size, >= window_size + pred_horizon - 1, action_dim)
            pad_mask: boolean array (batch, window_size) which is True if the timestep is not a padding timestep

        Returns:
            loss: float
            metrics: dict
        """
        batch_size, window_size = pad_mask.shape
        _check_action_window_size(actions, window_size, self.pred_horizon)
        actions_chunked = chunk_actions(actions, self.pred_horizon)
        actions_chunked = actions_chunked[:, :window_size]
        # fold action_dim and pred_horizon into one dimension
        actions_flat = rearrange(actions_chunked, "b w p a -> b w (p a)")
        actions_flat = jnp.clip(actions_flat, -self.max_action, self.max_action)

        # piggy-back on the dropout rng chain for diffusion rng
        rng = self.make_rng("dropout")
        time_key, noise_key = jax.random.split(rng)
        time = jax.random.randint(time_key, (batch_size, window_size, 1), 0, self.diffusion_steps)
        noise = jax.random.normal(noise_key, actions_flat.shape)

        alpha_hat = self.alpha_hats[time]
        alpha_1 = jnp.sqrt(alpha_hat)
        alpha_2 = jnp.sqrt(1 - alpha_hat)
        noisy_actions = alpha_1 * actions_flat + alpha_2 * noise

        pred_eps = self(transformer_outputs, train=train, time=time, noisy_actions=noisy_actions)

        loss, metrics = continuous_loss(pred_eps, noise, pad_mask[:, :, None], loss_type=self.loss_type)
        # Sum over action dimension instead of averaging
        loss = loss * self.action_dim
        metrics["loss"] = metrics["loss"] * self.action_dim
        metrics["mse"] = metrics["mse"] * self.action_dim
        return loss, metrics

    def predict_action(
        self,
        transformer_outputs: Dict[str, TokenGroup],
        rng: PRNGKey,
        train: bool = True,
        *args,
        sample_shape: tuple = (),
        **kwargs,
    ) -> jax.Array:
        """Convenience methods for predicting actions for the final timestep in the window."""
        module, variables = self.unbind()

        def scan_fn(carry, time):
            current_x, rng = carry
            input_time = jnp.broadcast_to(time, (*current_x.shape[:-1], 1))

            eps_pred = module.apply(variables, transformer_outputs, input_time, current_x, train=train)

            alpha_1 = 1 / jnp.sqrt(self.alphas[time])
            alpha_2 = (1 - self.alphas[time]) / (jnp.sqrt(1 - self.alpha_hats[time]))
            current_x = alpha_1 * (current_x - alpha_2 * eps_pred)

            rng, key = jax.random.split(rng)
            z = jax.random.normal(key, shape=current_x.shape)
            current_x = current_x + (time > 0) * (jnp.sqrt(self.betas[time]) * z)

            current_x = jnp.clip(current_x, -self.max_action, self.max_action)

            return (current_x, rng), ()

        def sample_actions(rng):
            rng, key = jax.random.split(rng)
            batch_size, window_size = transformer_outputs[self.readout_key].tokens.shape[:2]

            (actions_flat, _), () = jax.lax.scan(
                scan_fn,
                (
                    jax.random.normal(
                        key,
                        (batch_size, window_size, self.pred_horizon * self.action_dim),
                    ),
                    rng,
                ),
                jnp.arange(self.diffusion_steps - 1, -1, -1),
            )

            actions = rearrange(
                actions_flat,
                "b w (p a) -> b w p a",
                p=self.pred_horizon,
                a=self.action_dim,
            )
            # only get the last timestep in the window
            return actions[:, -1]

        n_samples = int(np.prod(sample_shape))
        actions = jax.vmap(sample_actions)(jax.random.split(rng, n_samples))
        actions = actions.reshape(sample_shape + actions.shape[1:])
        return actions


if __name__ == '__main__':
    mod = DiffusionActionHead(
        pred_horizon=5,
        action_dim=2,
        readout_key="readout_action",
    )

    input_to_model = TokenGroup(tokens=None, mask=None)

    print(111)
