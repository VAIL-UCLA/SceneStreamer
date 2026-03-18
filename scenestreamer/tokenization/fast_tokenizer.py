import json
import logging
from typing import ClassVar

import numpy as np
import torch
from scipy.fft import dct
from scipy.fft import idct
from tokenizers import ByteLevelBPETokenizer
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast
from transformers.processing_utils import ProcessorMixin

from scenestreamer.tokenization.biycle_tokenizer import BicycleModelTokenizerFixed0124
from scenestreamer.tokenization.motion_tokenizers import get_relative_velocity, \
    START_ACTION as MOTION_START_ACTION
from scenestreamer.utils import utils


class UniversalActionProcessor(ProcessorMixin):
    """
    Copied from: https://huggingface.co/physical-intelligence/fast/blob/main/processing_action_tokenizer.py
    """
    attributes: ClassVar[list[str]] = ["bpe_tokenizer"]
    bpe_tokenizer_class: str = "AutoTokenizer"

    def __init__(
        self,
        bpe_tokenizer: PreTrainedTokenizerFast,
        scale: float = 10,
        vocab_size: int = 1024,
        min_token: int = 0,
        *,
        action_dim: int | None = None,
        time_horizon: int | None = None,
    ):
        self.scale = scale
        self.vocab_size = vocab_size
        self.min_token = min_token
        assert min_token != 0
        # Action horizon and dimension needed during decoding. These can be specified
        # in three ways (in order of priority):
        # 1. passed in as kwargs to decode()
        # 2. in the constructor
        # 3. cached from the last time decode() was called
        self.time_horizon = time_horizon
        self.action_dim = action_dim
        self.called_time_horizon = time_horizon
        self.called_action_dim = action_dim

        super().__init__(bpe_tokenizer)

    def __call__(self, action_chunk: np.array) -> np.array:
        assert action_chunk.ndim <= 3, "Only 3 dimensions supported: [batch, timesteps, action_dim]"
        if action_chunk.ndim == 2:
            action_chunk = action_chunk[None, ...]

        # Cache the time horizon and action dimension for decoding
        self.called_time_horizon = action_chunk.shape[-2]
        self.called_action_dim = action_chunk.shape[-1]

        dct_coeff = dct(action_chunk, axis=1, norm="ortho")
        dct_coeff = np.around(dct_coeff * self.scale)

        # if dct_coeff.max() > 22:
        # print("MAX dct_coeff", dct_coeff.max(), "MIN dct_coeff", dct_coeff.min())

        tokens = []
        for elem in dct_coeff:
            token_str = "".join(map(chr, np.maximum(elem.flatten() - self.min_token, 0).astype(int)))
            tokens.append(self.bpe_tokenizer(token_str)["input_ids"])
        return tokens

    def decode(
        self,
        tokens: list[list[int]],
        *,
        time_horizon: int | None = None,
        action_dim: int | None = None,
    ) -> np.array:
        self.time_horizon = time_horizon or self.time_horizon or self.called_time_horizon
        self.action_dim = action_dim or self.action_dim or self.called_action_dim

        # Cache the time horizon and action dimension for the next call
        self.called_time_horizon = self.time_horizon
        self.called_action_dim = self.action_dim

        assert (
            self.time_horizon is not None and self.action_dim is not None
        ), "Tokenizer not initialized, call encode() once or pass in time_horizon and action_dim."

        decoded_actions = []
        error_rate = []

        for token in tokens:

            try:
                decoded_tokens = self.bpe_tokenizer.decode(token)
                decoded_dct_coeff = np.array(list(map(ord, decoded_tokens))) + self.min_token
                decoded_dct_coeff = decoded_dct_coeff.reshape(-1, self.action_dim)
                assert (
                    decoded_dct_coeff.shape == (
                        self.time_horizon,
                        self.action_dim,
                    )
                ), f"Decoded DCT coefficients have shape {decoded_dct_coeff.shape}, expected ({self.time_horizon}, {self.action_dim})"
                error_rate.append(0)
            except Exception as e:

                # PZH NOTE: remove error message
                # print(f"Error decoding tokens: {e}")
                # print(f"Tokens: {token}")
                error_rate.append(1)
                decoded_dct_coeff = np.zeros((self.time_horizon, self.action_dim))
            decoded_actions.append(idct(decoded_dct_coeff / self.scale, axis=0, norm="ortho"))
        assert len(error_rate) == len(decoded_actions)
        return np.stack(decoded_actions), np.stack(error_rate)

    @classmethod
    def fit(
        cls,
        action_data: list[np.array],
        scale: float = 10,
        vocab_size: int = 1024,
        *,
        time_horizon: int | None = None,
        action_dim: int | None = None,
    ) -> "UniversalActionProcessor":
        # Run DCT over all inputs
        dct_tokens = [dct(a, axis=0, norm="ortho").flatten() for a in action_data]

        # Quantize and find min token
        max_token = int(np.around(np.concatenate(dct_tokens) * scale).max())
        min_token = int(np.around(np.concatenate(dct_tokens) * scale).min())
        min_vocab_size = max_token - min_token

        assert (
            min_vocab_size <= vocab_size
        ), f"Vocab size {vocab_size} is too small for the range of tokens {min_vocab_size}"
        if min_vocab_size + 100 > vocab_size:
            logging.warning(
                f"Initial alphabet size {min_vocab_size} is almost as large as the vocab"
                f"size {vocab_size}, consider increasing vocab size"
            )

        # Make token iterator for BPE training
        def _token_iter():
            for tokens in dct_tokens:
                rounded_tokens = np.around(tokens * scale) - min_token
                rounded_tokens = rounded_tokens.astype(int)
                string = "".join(map(chr, rounded_tokens))
                yield string

        # Train BPE tokenizer
        bpe = ByteLevelBPETokenizer()

        # Set up the entire range of possible tokens as the initial alphabet
        alphabet = [chr(i) for i in range(max_token - min_token + 1)]
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=2,
            show_progress=True,
            special_tokens=[],
            initial_alphabet=alphabet,
            max_token_length=10000,
        )

        # Train the inner tokenizer (don't use ByteLevelBPETokenizer.train_from_iterator()
        # because it doesn't support custom alphabets)
        bpe._tokenizer.train_from_iterator(_token_iter(), trainer=trainer)

        return cls(
            PreTrainedTokenizerFast(tokenizer_object=bpe, clean_up_tokenization_spaces=False),
            scale=scale,
            vocab_size=vocab_size,
            min_token=min_token,
            time_horizon=time_horizon,
            action_dim=action_dim,
        )


def normalize_actions(data, lower_percentile=1, upper_percentile=99, predefined_quantiles=None):
    """
    Applies quantile normalization to each of the 15 features in the data.
    The data is assumed to have shape (N, 5, 3), corresponding to 5 time steps and 3 action dimensions,
    which yields a total of 15 features. The normalization maps the lower quantile to -1 and the upper
    quantile to 1 for each feature.

    Parameters:
        data (np.ndarray): Input array of shape (N, 5, 3).
        lower_percentile (float): Lower percentile (default 1) used if quantiles are not predefined.
        upper_percentile (float): Upper percentile (default 99) used if quantiles are not predefined.
        predefined_quantiles (dict or None): If provided, must have keys 'q_lower' and 'q_upper', each a numpy
                                             array of shape (15,) with quantile values for each feature.

    Returns:
        tuple: A tuple containing:
            - normalized_data (np.ndarray): Array of the same shape as data, with values in [-1, 1].
            - quantiles (dict): Dictionary with keys 'q_lower' and 'q_upper' used for normalization.
    """
    normalized_data = np.empty_like(data)

    if predefined_quantiles is None:
        q_lower_arr = np.empty(15)
        q_upper_arr = np.empty(15)
        # Reshape to (-1, 15) so that each column corresponds to one feature.
        data_reshaped = data.reshape(-1, 15)

        for i in range(15):
            values = data_reshaped[:, i]
            q_lower = np.percentile(values, lower_percentile)
            q_upper = np.percentile(values, upper_percentile)
            q_lower_arr[i] = q_lower
            q_upper_arr[i] = q_upper

            # Compute normalization for this feature.
            if q_upper == q_lower:
                normalized_feature = np.clip(data[..., i // 3, i % 3], -1, 1)
            else:
                scale = 2.0 / (q_upper - q_lower)
                normalized_feature = (data[..., i // 3, i % 3] - q_lower) * scale - 1
                normalized_feature = np.clip(normalized_feature, -1, 1)
            normalized_data[..., i // 3, i % 3] = normalized_feature

        quantiles = {'q_lower': q_lower_arr, 'q_upper': q_upper_arr}
    else:
        # Use the provided predefined quantiles.
        q_lower_arr = predefined_quantiles['q_lower']
        q_upper_arr = predefined_quantiles['q_upper']

        for i in range(15):
            q_lower = q_lower_arr[i]
            q_upper = q_upper_arr[i]
            if q_upper == q_lower:
                normalized_feature = np.clip(data[..., i // 3, i % 3], -1, 1)
            else:
                scale = 2.0 / (q_upper - q_lower)
                normalized_feature = (data[..., i // 3, i % 3] - q_lower) * scale - 1
                normalized_feature = np.clip(normalized_feature, -1, 1)
            normalized_data[..., i // 3, i % 3] = normalized_feature

        quantiles = predefined_quantiles

    return normalized_data, quantiles


def denormalize_actions(normalized_data, quantiles):
    """
    Reverses the quantile normalization for each of the 15 features.

    Parameters:
        normalized_data (np.ndarray): Normalized data array of shape (N, 5, 3) with values in [-1, 1].
        quantiles (dict): A dictionary with keys 'q_lower' and 'q_upper', each an array of shape (15,)
                          containing the quantile values for each feature.

    Returns:
        np.ndarray: Denormalized data array of the same shape as normalized_data.
    """
    denormalized_data = np.empty_like(normalized_data)
    q_lower_arr = quantiles['q_lower']
    q_upper_arr = quantiles['q_upper']

    for i in range(15):
        q_lower = q_lower_arr[i]
        q_upper = q_upper_arr[i]
        if q_upper == q_lower:
            denorm_feature = normalized_data[..., i // 3, i % 3]
        else:
            scale = 2.0 / (q_upper - q_lower)
            denorm_feature = (normalized_data[..., i // 3, i % 3] + 1) / scale + q_lower
        denormalized_data[..., i // 3, i % 3] = denorm_feature

    return denormalized_data


def get_norm_info(path):
    with open(path, "r") as f:
        norm_info = json.load(f)
    norm_info = {k: np.asarray(v) for k, v in norm_info.items()}
    return norm_info


class FastTokenizer(BicycleModelTokenizerFixed0124):
    def __init__(self, config):
        BicycleModelTokenizerFixed0124.__init__(self, config)

        from scenestreamer.utils import REPO_ROOT
        # import numpy as np

        self.use_type_specific_bins = False

        # cyc_tokenizer = AutoProcessor.from_pretrained("physical-intelligence/fast", trust_remote_code=True).from_pretrained(
        #     REPO_ROOT / "scenestreamer/tokenization/0305_fast_cyc_440000"
        # )
        # cyc_norm_info = get_norm_info(REPO_ROOT / "scenestreamer/tokenization/0305_fast_cyc_440000/delta_normalization_quantiles.json")
        #
        # ped_tokenizer = AutoProcessor.from_pretrained("physical-intelligence/fast", trust_remote_code=True).from_pretrained(
        #     REPO_ROOT / "scenestreamer/tokenization/0305_fast_ped_4000000"
        # )
        # ped_norm_info = get_norm_info(REPO_ROOT / "scenestreamer/tokenization/0305_fast_ped_4000000/delta_normalization_quantiles.json")
        #
        # veh_tokenizer = AutoProcessor.from_pretrained("physical-intelligence/fast", trust_remote_code=True).from_pretrained(
        #     REPO_ROOT / "scenestreamer/tokenization/0305_fast_veh_5000000"
        # )
        # veh_norm_info = get_norm_info(REPO_ROOT / "scenestreamer/tokenization/0305_fast_veh_5000000/delta_normalization_quantiles.json")

        # self.fast_tokenizers = {
        #     "cyc": cyc_tokenizer,
        #     "ped": ped_tokenizer,
        #     "veh": veh_tokenizer
        # }

        all_tokenizer = UniversalActionProcessor.from_pretrained(
            REPO_ROOT / "scenestreamer/tokenization/0305_fast_all", time_horizon=5, action_dim=3
        )
        cyc_norm_info = get_norm_info(REPO_ROOT / "scenestreamer/tokenization/0305_fast_all/norm_info_cyc.json")
        ped_norm_info = get_norm_info(REPO_ROOT / "scenestreamer/tokenization/0305_fast_all/norm_info_ped.json")
        veh_norm_info = get_norm_info(REPO_ROOT / "scenestreamer/tokenization/0305_fast_all/norm_info_veh.json")
        self.fast_tokenizers = {"cyc": all_tokenizer, "ped": all_tokenizer, "veh": all_tokenizer}
        self.fast_tokenizer = all_tokenizer

        self.norm_infos = {"cyc": cyc_norm_info, "ped": ped_norm_info, "veh": veh_norm_info}

        #
        # self.num_actions = len(all_trajs)
        #
        # self.all_trajs = torch.from_numpy(all_trajs).float()
        # self.bin_centers = self.all_trajs[:, -1].reshape(1, self.num_actions, 1, 2)
        #
        # self.config = config
        # self.all_heading = torch.from_numpy(all_head).float()
        #
        # self.default_action = 0  # We set action 0 to be all zeros.
        # self.add_noise = config.TOKENIZATION.ADD_NOISE

        self.num_actions = 1024 + 3

    # def get_motion_feature(self):
    #     # m = torch.from_numpy(self.bin_centers_flat)
    #     m = self.all_trajs[:, -1]  # (1025, 2)
    #     dist = m.norm(p=2, dim=-1).unsqueeze(-1)
    #     heading = self.all_heading[:, -1]
    #     return torch.cat([m, dist, heading], dim=-1)

    def tokenize(self, data_dict, backward_prediction=False, **kwargs):
        """

        Args:
            data_dict: Input data

        Returns:
            Discretized action in an int array with shape (num time steps for actions, num agents).
        """

        if backward_prediction:
            raise ValueError("FastTokenizer does not support backward prediction.")
            return self._tokenize_backward_prediction(data_dict, **kwargs)

        # TODO: Hardcoded here...
        assert self.config.GPT_STYLE
        start_step = 0

        # ===== Hole Filling =====
        data_dict = self.hole_filling(data_dict)

        # ===== Get initial data =====
        # If we don't clone here, the following hole-filling code will overwrite raw data.
        agent_pos = data_dict["decoder/agent_position"]  # .clone()
        agent_heading = data_dict["decoder/agent_heading"]  # .clone()
        agent_valid_mask = data_dict["decoder/agent_valid_mask"]  # .clone()
        agent_velocity = data_dict["decoder/agent_velocity"]  # .clone()
        agent_shape = data_dict["decoder/current_agent_shape"]  # .clone()
        agent_type = data_dict["decoder/agent_type"]  # .clone()
        B, T_full, N, _ = agent_pos.shape
        # assert T_full == 91
        assert agent_pos.ndim == 4

        # ===== Skip some steps =====
        # agent_pos_full = agent_pos.clone()
        # agent_heading_full = agent_heading.clone()
        # agent_velocity_full = agent_velocity.clone()
        # agent_valid_mask_full = agent_valid_mask.clone()

        agent_pos_chunk = agent_pos.unfold(dimension=1, size=6, step=5).swapaxes(-1, -2)
        agent_heading_chunk = agent_heading.unfold(dimension=1, size=6, step=5)
        agent_velocity_chunk = agent_velocity.unfold(dimension=1, size=6, step=5).swapaxes(-1, -2)

        agent_valid_mask_chunk_full = agent_valid_mask.unfold(dimension=1, size=6, step=5)
        agent_valid_mask_chunk = agent_valid_mask_chunk_full.all(dim=-1)
        # This will hold: agent_pos_chunk[0, 1:, :, 0] == agent_pos[0, :-1, :, 5]

        agent_pos = agent_pos[:, ::self.num_skipped_steps]  # [:, :-1]
        agent_heading = agent_heading[:, ::self.num_skipped_steps]  # [:, :-1]
        agent_valid_mask = agent_valid_mask[:, ::self.num_skipped_steps]  # [:, :-1]
        agent_velocity = agent_velocity[:, ::self.num_skipped_steps]  # [:, :-1]

        agent_valid_mask_chunk = torch.cat([agent_valid_mask_chunk, agent_valid_mask[:, -1:]], dim=1)

        # agent_valid_mask_chunk_all = agent_valid_mask_chunk.all(dim=-1)
        # # Add final step
        # agent_valid_mask_chunk_all = torch.cat([agent_valid_mask_chunk_all, agent_valid_mask_chunk_all.new_zeros((B, 1, N))], dim=1)
        # agent_valid_mask_chunk = torch.logical_and(agent_valid_mask, agent_valid_mask_chunk_all)
        # assert agent_valid_mask.shape == agent_valid_mask_chunk_all.shape

        # T_chunks = agent_pos.shape[1]
        # assert T_chunks == 19
        T_chunks = agent_pos.shape[1]

        # ===== Build up some variables =====
        current_pos = agent_pos[:, start_step:start_step + 1, ..., :2]
        current_heading = agent_heading[:, start_step:start_step + 1]
        current_vel = agent_velocity[:, start_step:start_step + 1, ..., :2]
        current_valid_mask = agent_valid_mask[:, start_step:start_step + 1]

        init_pos = current_pos.clone()
        init_heading = current_heading.clone()
        init_vel = current_vel.clone()
        init_valid_mask = current_valid_mask.clone()

        assert self.config.DELTA_POS_IS_VELOCITY
        init_delta = get_relative_velocity(current_vel, current_heading)

        # Select correct bins:
        bin_centers = self.get_bin_centers(agent_type)

        target_action = []
        target_action_valid_mask = []
        reconstruction_list = []
        relative_delta_pos_list = []
        pos = []
        heading = []
        vel = []

        # ===== Loop to reconstruct the scenario =====
        tokenization_state = None
        for next_step in range(start_step + 1, T_chunks):
            res = self._tokenize_a_step(
                current_pos=current_pos,
                current_heading=current_heading,
                current_vel=current_vel,
                current_valid_mask=current_valid_mask,
                next_pos=agent_pos[:, next_step:next_step + 1, ..., :2],  # (B, 1, N, 2)
                next_heading=agent_heading[:, next_step:next_step + 1],  # (B, 1, N)
                next_valid_mask=agent_valid_mask_chunk[:, next_step - 1:next_step],  # (B, 1, N)
                next_velocity=agent_velocity[:, next_step:next_step + 1, ..., :2],  # (B, 1, N, 2)
                bin_centers=bin_centers,
                add_noise=False,
                topk=self.config.TOKENIZATION.NOISE_TOPK,
                agent_shape=agent_shape,
                agent_type=agent_type,
                dt=self.dt,
                tokenization_state=tokenization_state,
                agent_pos_full=agent_pos_chunk[:, next_step - 1],
                agent_heading_full=agent_heading_chunk[:, next_step - 1],
                agent_velocity_full=agent_velocity_chunk[:, next_step - 1],
                # agent_valid_mask_full=agent_valid_mask_full[:, (next_step - 1) *
                #                                             self.num_skipped_steps:next_step * self.num_skipped_steps +
                #                                             1],
            )
            tokenization_state = res

            # best_action = res["action"]
            recon_next_pos = res["pos"]
            recon_next_heading = res["heading"]
            recon_next_vel = res["vel"]
            recon_next_valid_mask = res["mask"]
            recon_next_delta_pos = res["delta_pos"]  # The input delta for next step.

            # best_action = best_action.reshape(B, 1, N)

            # ===== Process the target action/valid mask =====
            target_action_valid_mask.append(recon_next_valid_mask.clone())
            target_action.append(res["action"])

            # Some debug asserts
            # assert (best_action[recon_next_valid_mask] >= 0).all()
            # assert (best_action[~recon_next_valid_mask] == -1).all()

            # ===== Process the "current_xxx" for next step =====
            if self.config.GPT_STYLE:
                assert self.config.TOKENIZATION.ALLOW_SKIP_STEP
            if self.config.TOKENIZATION.ALLOW_SKIP_STEP:
                # Use the next valid mask as the valid mask for next step.
                # In contrast, if this flag is False, then we will use "next valid mask & if it's not removed" for next
                # step.
                next_valid_mask = agent_valid_mask[:, next_step:next_step + 1]
                newly_added = torch.logical_and(~recon_next_valid_mask, next_valid_mask)
                if newly_added.any():
                    assert not (agent_pos[:, next_step:next_step + 1, ..., :2][newly_added] == 0.0).all(-1).any()
                    recon_next_pos[newly_added] = agent_pos[:, next_step:next_step + 1, ..., :2][newly_added]
                    recon_next_heading[newly_added] = agent_heading[:, next_step:next_step + 1][newly_added]
                    recon_next_vel[newly_added] = agent_velocity[:, next_step:next_step + 1, ..., :2][newly_added]

                    recon_next_delta_pos[newly_added] = get_relative_velocity(
                        vel=agent_velocity[:, next_step:next_step + 1, ..., :2][newly_added],
                        heading=agent_heading[:, next_step:next_step + 1][newly_added],
                    )
                    recon_next_valid_mask[newly_added] = next_valid_mask[newly_added]

            relative_delta_pos_list.append(recon_next_delta_pos)
            current_vel = recon_next_vel
            current_heading = recon_next_heading
            current_pos = recon_next_pos
            current_valid_mask = recon_next_valid_mask
            pos.append(current_pos.clone())
            heading.append(current_heading.clone())
            vel.append(current_vel.clone())

        # ===== Postprocess and prepare the "start action" =====
        # In GPT style, some agents will be added in the middle of the scene.
        # So we need to find out when they are in and add a start action before that step.
        # In non-GPT style, we only need to prepare the start action for the first step.
        max_token_len = 0
        for step_tokens in target_action:
            max_token_len = max(max_token_len, max([len(v) for v in step_tokens]))
        target_actions = torch.full((B, T_chunks - 1, N, max_token_len), -1, dtype=torch.long)
        assert B == 1
        for i, step_tokens in enumerate(target_action):
            for j, tokens in enumerate(step_tokens):
                target_actions[0, i, j, :len(tokens)] = torch.from_numpy(np.asarray(tokens))

        target_action_valid_mask = torch.cat(target_action_valid_mask, dim=1)  # (B, T_skipped, N)
        relative_delta_pos_list = torch.cat(relative_delta_pos_list, dim=1)  # (B, T_skipped, N)
        pos = torch.cat(pos, dim=1)
        heading = torch.cat(heading, dim=1)
        vel = torch.cat(vel, dim=1)

        pos = torch.cat([init_pos, pos], dim=1)
        heading = torch.cat([init_heading, heading], dim=1)
        vel = torch.cat([init_vel, vel], dim=1)
        relative_delta_pos_list = torch.cat([init_delta, relative_delta_pos_list], dim=1)

        # If not in back prediction, what will be:
        # 1. The first tokens in input_actions? START_ACTION
        # 2. The last tokens in input_actions? Just the tokens at t=18 (t=85~90)
        # 3. The first tokens in target_actions? The tokens at t=0 (t=0~5) for GPT and t=2 otherwise.
        # 4. The last tokens in target_actions? All -1 because there is no GT for t=19 (t=90~95)
        assert self.config.GPT_STYLE
        # Search for the first step that has newly added agents
        assert start_step == 0
        already_tokenized = init_valid_mask.clone()
        start_action = torch.full_like(target_actions[:, :1], -1)
        start_action[init_valid_mask] = MOTION_START_ACTION
        assert target_actions.shape[1] == T_chunks - 1
        input_action = torch.cat([start_action, target_actions], dim=1)
        input_action_valid_mask = torch.cat([init_valid_mask, target_action_valid_mask], dim=1)
        for next_step in range(start_step + 1, T_chunks):
            next_valid_mask = agent_valid_mask[:, next_step:next_step + 1]
            is_newly_added = torch.logical_and(~already_tokenized, next_valid_mask)
            if is_newly_added.any():
                input_action[:, next_step:next_step + 1][is_newly_added] = MOTION_START_ACTION
                input_action_valid_mask[:, next_step:next_step + 1][is_newly_added] = \
                    next_valid_mask[is_newly_added]
            already_tokenized = torch.logical_or(already_tokenized, is_newly_added)

        target_actions = torch.cat([target_actions, target_actions.new_full((B, 1, N, max_token_len), -1)], dim=1)
        target_action_valid_mask = torch.cat(
            [target_action_valid_mask, target_action_valid_mask.new_zeros((B, 1, N))], dim=1
        )
        data_dict["in_backward_prediction"] = False
        assert (agent_valid_mask[:, start_step:] >= target_action_valid_mask).all()
        assert (agent_valid_mask[:, start_step + 1:] >= target_action_valid_mask[:, :-1]).all()
        assert (agent_valid_mask[:, start_step:] >= input_action_valid_mask).all()

        data_dict["decoder/target_action"] = target_actions
        data_dict["decoder/target_action_valid_mask"] = target_action_valid_mask
        data_dict["decoder/input_action"] = input_action
        data_dict["decoder/input_action_valid_mask"] = input_action_valid_mask
        data_dict["decoder/modeled_agent_delta"] = relative_delta_pos_list
        data_dict["decoder/modeled_agent_position"] = pos
        data_dict["decoder/modeled_agent_heading"] = heading
        data_dict["decoder/modeled_agent_velocity"] = vel

        # Debug:
        # pos_diff = (pos - agent_pos[..., :2]).norm(dim=-1).numpy()
        # heading_diff = utils.wrap_to_pi(heading - agent_heading).abs().numpy()
        # vel_diff = (vel - agent_velocity[..., :2]).norm(dim=-1).numpy()

        # All input actions should be >0
        assert (input_action[input_action_valid_mask] >= 0).any(-1).all()
        assert (target_actions[target_action_valid_mask] >= 0).any(-1).all()
        assert (input_action[~input_action_valid_mask] == -1).all(-1).all()
        assert (target_actions[~target_action_valid_mask] == -1).all(-1).all()

        return data_dict, {"reconstruction_list": reconstruction_list}

    def _tokenize_a_step(
        self, *, current_pos, current_heading, current_valid_mask, current_vel, next_pos, next_heading, next_valid_mask,
        add_noise, agent_shape, dt, agent_pos_full, agent_heading_full, agent_velocity_full, agent_type, next_velocity,
        **kwargs
    ):
        if dt < 0:
            raise ValueError("FastTokenizer does not support backward prediction.")

        B, _, N, _ = current_pos.shape

        # Change shape. Input: (B, N, 6, 3)
        assert agent_pos_full.ndim == 4
        agent_pos_full = agent_pos_full[..., :2]  # (B, N, 6, 2)
        assert agent_heading_full.ndim == 3

        valid_mask = torch.logical_and(current_valid_mask, next_valid_mask)

        # Rotate

        static_error = (
            ((agent_pos_full[:, :, 0] - current_pos.reshape(B, N, 2)).norm(dim=-1)) * current_valid_mask.reshape(B, N)
        ).sum() / current_valid_mask.reshape(B, N).sum()
        # print("Static error: ", static_error)

        agent_pos_full_rotated = agent_pos_full - current_pos.reshape(B, N, 1, 2)
        agent_pos_full_rotated = utils.rotate(
            agent_pos_full_rotated[..., 0], agent_pos_full_rotated[..., 1],
            -current_heading.reshape(B, N, 1).expand(-1, -1, 6)
        )
        agent_heading_full = utils.wrap_to_pi(agent_heading_full - current_heading.reshape(B, N, 1))

        # Stack
        chunk = torch.cat([agent_pos_full_rotated, agent_heading_full[..., None]], dim=-1)  # (B, N, 6, 3)

        # Mask
        chunk = chunk.masked_fill_(~valid_mask.reshape(B, N, 1, 1).expand(-1, -1, 6, 3), 0)

        # Compute delta
        chunk_delta = chunk[:, :, 1:] - chunk[:, :, :-1]  # (B, N, 5, 3)

        # Swap x and y
        chunk_delta = chunk_delta[..., [1, 0, 2]]

        # Normalize
        is_ped = agent_type == 2
        assert B == 1
        tokenized_chunk = [None] * N
        if is_ped.any():
            chunk_ped, _ = normalize_actions(chunk_delta[is_ped].numpy(), predefined_quantiles=self.norm_infos["ped"])
            chunk_ped = torch.from_numpy(chunk_ped).float()
            chunk_ped = self.fast_tokenizers["ped"](chunk_ped)
            count = 0
            for i in range(N):
                if is_ped[0, i]:
                    tokenized_chunk[i] = chunk_ped[count]
                    count += 1
        is_cyc = agent_type == 3
        if is_cyc.any():
            chunk_cyc, _ = normalize_actions(chunk_delta[is_cyc].numpy(), predefined_quantiles=self.norm_infos["cyc"])
            chunk_cyc = torch.from_numpy(chunk_cyc).float()
            chunk_cyc = self.fast_tokenizers["cyc"](chunk_cyc)
            count = 0
            for i in range(N):
                if is_cyc[0, i]:
                    tokenized_chunk[i] = chunk_cyc[count]
                    count += 1

        is_veh = ~(agent_type != 1)
        chunk_veh, _ = normalize_actions(chunk_delta[is_veh].numpy(), predefined_quantiles=self.norm_infos["veh"])
        chunk_veh = torch.from_numpy(chunk_veh).float()
        chunk_veh = self.fast_tokenizers["veh"](chunk_veh)
        count = 0
        for i in range(N):
            if is_veh[0, i]:
                tokenized_chunk[i] = chunk_veh[count]
                count += 1

        detok = self._detokenize_a_step(
            current_pos=current_pos,
            current_heading=current_heading,
            current_valid_mask=current_valid_mask,
            action=tokenized_chunk,
            agent_type=agent_type
        )
        recon_pos = detok["pos"]
        recon_heading = detok["heading"]
        recon_vel = detok["vel"]
        recon_delta = detok["delta_pos"]

        # error_rate = detok["error_rate_full"]

        # recon_pos = next_pos.clone()
        # recon_heading = next_heading.clone()
        # recon_vel = next_velocity.clone()
        # recon_delta = get_relative_velocity(recon_vel, recon_heading)

        recon_pos[~valid_mask] = 0
        recon_heading[~valid_mask] = 0
        recon_vel[~valid_mask] = 0
        recon_delta[~valid_mask] = 0

        for i, v in enumerate(valid_mask[0, 0]):
            if not v:
                tokenized_chunk[i] = []

        # error = ((recon_pos[0,0,:] - next_pos[0,0,:]).norm(dim=-1) * valid_mask[0,0,:])
        # error_max = error.max().item()
        # error_argmax = error.argmax().item()
        # AID = error_argmax
        # if error_max > 5:
        #     print("CUR {}, recon Pos {}, gt pos {}, error {}, valid mask {}".format(
        #         current_pos[0,0,AID],
        #         recon_pos[0,0,AID],
        #         next_pos[0,0,AID],
        #         (recon_pos[0,0,AID] - next_pos[0,0,AID]).norm(),
        #         valid_mask[0,0,AID]
        #     ))

        return dict(
            action=tokenized_chunk,
            pos=recon_pos,
            heading=recon_heading,
            vel=recon_vel,
            mask=valid_mask,
            delta_pos=recon_delta,
        )

    def detokenize(
        self,
        data_dict,
        interpolation=True,
        detokenizing_gt=False,
        backward_prediction=False,
        flip_wrong_heading=False,
        autoregressive_start_step=2,
        **kwargs,
    ):  # actions, current_pos, current_vel, current_heading):
        # TODO: Hardcoded here...
        assert self.config.GPT_STYLE
        start_step = 0

        # ===== Get initial data =====
        agent_pos = data_dict["decoder/agent_position"].clone()
        agent_heading = data_dict["decoder/agent_heading"].clone()
        agent_valid_mask = data_dict["decoder/agent_valid_mask"].clone()
        agent_velocity = data_dict["decoder/agent_velocity"].clone()
        agent_shape = data_dict["decoder/current_agent_shape"].clone()
        agent_type = data_dict["decoder/agent_type"].clone()
        if detokenizing_gt:
            target_action_valid_mask = data_dict["decoder/target_action_valid_mask"]
        input_mask = data_dict["decoder/input_action_valid_mask"]
        B, T_full, N, _ = agent_pos.shape
        assert agent_pos.ndim == 4

        # ===== Skip some steps =====
        agent_pos = agent_pos[:, ::self.num_skipped_steps].clone()
        agent_heading = agent_heading[:, ::self.num_skipped_steps]
        agent_valid_mask = agent_valid_mask[:, ::self.num_skipped_steps]
        agent_velocity = agent_velocity[:, ::self.num_skipped_steps]
        # T_chunks = agent_pos.shape[1]

        # ===== Prepare some variables =====
        action = data_dict["decoder/output_action"]
        T_actions = action.shape[1]
        T_generated_chunks = T_actions + start_step

        current_pos = agent_pos[:, start_step:start_step + 1, ..., :2].clone()
        current_heading = agent_heading[:, start_step:start_step + 1].clone()
        current_vel = agent_velocity[:, start_step:start_step + 1, ..., :2].clone()
        current_valid_mask = agent_valid_mask[:, start_step:start_step + 1].clone()

        if detokenizing_gt:
            # Merge input mask with target mask
            input_mask = input_mask & target_action_valid_mask

        reconstructed_pos_list = [current_pos.clone()]
        reconstructed_heading_list = [current_heading.clone()]
        reconstructed_vel_list = [current_vel.clone()]

        already_interpolated = False
        reconstructed_pos_full_list = [current_pos.clone()]
        reconstructed_heading_full_list = [current_heading.clone()]
        reconstructed_vel_full_list = [current_vel.clone()]

        # Select correct bins:
        bin_centers = self.get_bin_centers(agent_type)

        kwargs["detokenization_state"] = None

        for curr_step in range(T_generated_chunks):

            # We assume that starting from start_step, the agent valid mask will not change.
            action_step = curr_step - start_step
            action_valid_mask_step = input_mask[:, action_step:action_step + 1]

            act = action[:, action_step:action_step + 1]
            assert (act[action_valid_mask_step] != -1).any(-1).all()
            res = self._detokenize_a_step(
                current_pos=current_pos,
                current_heading=current_heading,
                current_valid_mask=action_valid_mask_step,
                current_vel=current_vel,
                action=act,
                agent_shape=agent_shape,
                agent_type=agent_type,
                bin_centers=bin_centers,
                dt=self.dt,
                flip_wrong_heading=flip_wrong_heading,
                **kwargs
            )
            kwargs["detokenization_state"] = res

            next_pos, next_heading, next_vel = res["pos"], res["heading"], res["vel"]
            assert "delta_pos" in res
            next_pos = next_pos.reshape(B, 1, N, 2)
            next_heading = next_heading.reshape(B, 1, N)
            next_vel = next_vel.reshape(B, 1, N, 2)
            next_valid_mask = current_valid_mask

            # ===== A special case: fill in the info for the agents added in next step =====
            # ===== Another special case: if you are detokenizing the raw tokenized data, you need to fill in
            # the info for the agents added in the next step. =====
            if (curr_step < autoregressive_start_step) or (detokenizing_gt and curr_step < T_generated_chunks - 1):
                # Fill in the initial states of newly added agents
                action_valid_mask_next_step = input_mask[:, action_step + 1:action_step + 2]
                newly_added = torch.logical_and(~action_valid_mask_step, action_valid_mask_next_step)
                if newly_added.any():
                    next_pos[newly_added] = agent_pos[:, curr_step + 1:curr_step + 2, ..., :2][newly_added]
                    next_heading[newly_added] = agent_heading[:, curr_step + 1:curr_step + 2][newly_added]
                    next_vel[newly_added] = agent_velocity[:, curr_step + 1:curr_step + 2, ..., :2][newly_added]
                    next_valid_mask[newly_added] = action_valid_mask_next_step[newly_added]
                    if "reconstructed_position" in res:
                        # If some agents are added in the next step, the "last step" in reconstructed chunk
                        # aka the 5-th step in the chunk should be replaced by the GT states.
                        assert (agent_pos[:, curr_step + 1:curr_step + 2, ..., :2][newly_added][..., 0] != 0).all()
                        res["reconstructed_position"][-1][newly_added] = agent_pos[:, curr_step + 1:curr_step + 2,
                                                                                   ..., :2][newly_added]
                        res["reconstructed_heading"][-1][newly_added] = agent_heading[:, curr_step + 1:curr_step +
                                                                                      2][newly_added]
                        res["reconstructed_velocity"][-1][newly_added] = agent_velocity[:, curr_step + 1:curr_step + 2,
                                                                                        ..., :2][newly_added]

            if "reconstructed_position" in res:
                already_interpolated = True
                reconstructed_pos_full_list.extend(res["reconstructed_position"])
                reconstructed_heading_full_list.extend(res["reconstructed_heading"])
                reconstructed_vel_full_list.extend(res["reconstructed_velocity"])

            current_pos = next_pos
            current_heading = next_heading
            current_vel = next_vel
            current_valid_mask = next_valid_mask

            reconstructed_pos_list.append(current_pos.clone())
            reconstructed_heading_list.append(current_heading.clone())
            reconstructed_vel_list.append(current_vel.clone())

        reconstructed_pos = torch.cat(reconstructed_pos_list, dim=1)
        reconstructed_heading = torch.cat(reconstructed_heading_list, dim=1)
        reconstructed_vel = torch.cat(reconstructed_vel_list, dim=1)

        # Every input token has it's own position (before the action).
        # As we have 19 tokens, and the last one token will lead us to a new place,
        # So it's totally 20 positions.
        assert reconstructed_pos.shape[1] == T_generated_chunks + 1
        assert input_mask.shape[1] == T_generated_chunks - start_step

        # Interpolation
        reconstructed_pos = torch.cat(reconstructed_pos_full_list, dim=1)
        reconstructed_heading = torch.cat(reconstructed_heading_full_list, dim=1)
        reconstructed_vel = torch.cat(reconstructed_vel_full_list, dim=1)

        input_mask_augmented = torch.cat([agent_valid_mask[:, :start_step], input_mask], dim=1)
        assert input_mask_augmented.shape[1] == T_generated_chunks
        valid = input_mask_augmented
        valid = valid.reshape(B, -1, 1, N).expand(-1, -1, self.num_skipped_steps, -1).reshape(B, -1, N)
        valid = torch.cat([valid, input_mask[:, -1:]], dim=1)
        reconstructed_valid_mask = valid

        # Mask out:
        reconstructed_pos = reconstructed_pos * reconstructed_valid_mask.unsqueeze(-1)
        reconstructed_vel = reconstructed_vel * reconstructed_valid_mask.unsqueeze(-1)
        reconstructed_heading = reconstructed_heading * reconstructed_valid_mask

        # We ensure that the output must be 5*T_chunks+1
        assert reconstructed_pos.shape[1] == self.num_skipped_steps * T_generated_chunks + 1
        assert reconstructed_valid_mask.shape[1] == self.num_skipped_steps * T_generated_chunks + 1
        assert reconstructed_vel.shape[1] == self.num_skipped_steps * T_generated_chunks + 1
        assert reconstructed_heading.shape[1] == self.num_skipped_steps * T_generated_chunks + 1

        data_dict["decoder/reconstructed_position"] = reconstructed_pos
        data_dict["decoder/reconstructed_heading"] = reconstructed_heading
        data_dict["decoder/reconstructed_velocity"] = reconstructed_vel
        data_dict["decoder/reconstructed_valid_mask"] = reconstructed_valid_mask

        return data_dict

    def _detokenize_a_step(self, *, current_pos, current_heading, current_valid_mask, action, agent_type, **kwargs):
        # assert action.ndim == 3
        # B, T_action, N = action.shape

        device = current_pos.device

        # assert T_action == 1
        if isinstance(action, list):
            tokens = action
            B, _, N, _ = current_pos.shape

        else:
            B, _, N, _ = action.shape

            assert action.max() < self.fast_tokenizer.vocab_size

            assert action.ndim == 4
            assert action.shape[1] == 1
            action = action.squeeze(1)  # (B, N, max_num_tokens)

            # Convert action to list
            action = action.reshape(-1, action.shape[-1])
            tokens = [v[v != -1] for v in action]
            # tokens = [(v if len(v) > 0 else tokens[0].new_full((1,), 0)) for v in tokens]

        # Process pedestrian
        chunk_full = torch.zeros((B * N, 5, 3)).to(device)
        error_rate_full = np.zeros((B * N,))
        is_ped = (agent_type == 2) & current_valid_mask.reshape(B, N)
        is_ped = is_ped.reshape(-1)
        if is_ped.any():
            chunk_ped, error_rate_ped = self.fast_tokenizers["ped"].decode([v for i, v in enumerate(tokens) if is_ped[i]])

            if isinstance(is_ped, torch.Tensor):
                error_rate_full[is_ped.cpu().numpy()] = error_rate_ped
            else:
                error_rate_full[is_ped] = error_rate_ped

            chunk_ped = denormalize_actions(chunk_ped, quantiles=self.norm_infos["ped"])
            chunk_ped = torch.from_numpy(chunk_ped).float().to(device)
            count = 0
            for i in range(len(tokens)):
                if is_ped[i]:
                    chunk_full[i] = chunk_ped[count]
                    count += 1

        is_cyc = (agent_type == 3) & current_valid_mask.reshape(B, N)
        is_cyc = is_cyc.reshape(-1)
        if is_cyc.any():
            chunk_cyc, error_rate_cyc = self.fast_tokenizers["cyc"].decode([v for i, v in enumerate(tokens) if is_cyc[i]])

            if isinstance(is_cyc, torch.Tensor):
                error_rate_full[is_cyc.cpu().numpy()] = error_rate_cyc
            else:
                error_rate_full[is_cyc] = error_rate_cyc

            chunk_cyc = denormalize_actions(chunk_cyc, quantiles=self.norm_infos["cyc"])
            chunk_cyc = torch.from_numpy(chunk_cyc).float().to(device)
            count = 0
            for i in range(len(tokens)):
                if is_cyc[i]:
                    chunk_full[i] = chunk_cyc[count]
                    count += 1

        is_vel = (agent_type == 1) & current_valid_mask.reshape(B, N)
        is_vel = is_vel.reshape(-1)
        if is_vel.any():
            chunk_veh, error_rate_veh = self.fast_tokenizers["veh"].decode([v for i, v in enumerate(tokens) if is_vel[i]])

            if isinstance(is_vel, torch.Tensor):
                error_rate_full[is_vel.cpu().numpy()] = error_rate_veh
            else:
                error_rate_full[is_vel] = error_rate_veh

            chunk_veh = denormalize_actions(chunk_veh, quantiles=self.norm_infos["veh"])
            chunk_veh = torch.from_numpy(chunk_veh).float().to(device)
            count = 0
            for i in range(len(tokens)):
                if is_vel[i]:
                    chunk_full[i] = chunk_veh[count]
                    count += 1

        # Reshape back
        chunk_full = chunk_full.reshape(B, N, 5, 3)

        # Cumsum
        chunk_full = chunk_full.cumsum(dim=-2)

        # Swap x and y
        chunk_full = chunk_full[..., [1, 0, 2]]

        # Rotate
        chunk_pos = utils.rotate(
            chunk_full[..., 0], chunk_full[..., 1],
            current_heading.reshape(B, N, 1).expand(-1, -1, 5)
        )
        chunk_head = utils.wrap_to_pi(chunk_full[..., 2] + current_heading.reshape(B, N, 1).expand(-1, -1, 5))

        # Translation
        chunk_pos = chunk_pos + current_pos.reshape(B, N, 1, 2).expand(-1, -1, 5, 2)

        # Mask
        chunk_pos = chunk_pos.masked_fill_(~current_valid_mask.reshape(B, N, 1, 1).expand(-1, -1, 5, 2), 0)
        chunk_head = chunk_head.masked_fill_(~current_valid_mask.reshape(B, N, 1).expand(-1, -1, 5), 0)

        # Get output
        reconstructed_pos = chunk_pos[:, :, -1].reshape(B, 1, N, 2)
        reconstructed_heading = chunk_head[:, :, -1].reshape(B, 1, N)

        reconstructed_vel = (reconstructed_pos - current_pos) / self.dt

        relative_delta_pos = get_relative_velocity(reconstructed_vel, reconstructed_heading)

        chunk_pos_6steps = torch.cat([current_pos.reshape(B, N, 1, 2), chunk_pos], dim=-2)
        recon_vel = (chunk_pos_6steps[:, :, 1:] - chunk_pos_6steps[:, :, :-1]) / (self.dt / self.num_skipped_steps)

        # AID = 0
        # print(
        #     "POS: {}, HEAD: {}, valid {}".format(
        #         reconstructed_pos[0, 0, AID].cpu().numpy(),
        #         reconstructed_heading[0, 0, AID],
        #         current_valid_mask[0, 0, AID]
        #         # reconstructed_vel[0, 0, AID].norm(dim=-1).cpu().numpy(),
        #         # reconstructed_vel.norm(dim=-1)[0, 0, AID],
        #         # unrotated_delta_vel[0, 0, AID].cpu().numpy(),
        #         # reconstructed_vel[0, 0, AID].norm(dim=-1)
        #     )
        # )

        return dict(
            pos=reconstructed_pos,
            heading=reconstructed_heading,
            vel=reconstructed_vel,
            delta_pos=relative_delta_pos,
            reconstructed_position=[chunk_pos[:, :, i].unsqueeze(1) for i in range(5)],  # (B, N, 5, 2)
            reconstructed_heading=[chunk_head[:, :, i].unsqueeze(1) for i in range(5)],  # (B, N, 5)
            reconstructed_velocity=[recon_vel[:, :, i].unsqueeze(1) for i in range(5)],  # (B, N, 5, 2)
            error_rate_full=error_rate_full,
        )


if __name__ == '__main__':
    from scenestreamer.utils import REPO_ROOT
    all_tokenizer = UniversalActionProcessor.from_pretrained(
        REPO_ROOT / "scenestreamer/tokenization/0305_fast_all", time_horizon=5, action_dim=3
    )
    cyc_norm_info = get_norm_info(REPO_ROOT / "scenestreamer/tokenization/0305_fast_all/norm_info_cyc.json")
    ped_norm_info = get_norm_info(REPO_ROOT / "scenestreamer/tokenization/0305_fast_all/norm_info_ped.json")
    veh_norm_info = get_norm_info(REPO_ROOT / "scenestreamer/tokenization/0305_fast_all/norm_info_veh.json")

    import numpy as np

    bs = 11
    chunk_delta = np.zeros((bs, 5, 3))
    chunk_normalized, _ = normalize_actions(chunk_delta, predefined_quantiles=veh_norm_info)
    chunk_tokenized = all_tokenizer(chunk_normalized)
    print(chunk_tokenized)

    chunk_detokenized, detok_error_rate = all_tokenizer.decode(chunk_tokenized)
    chunk_denormlized = denormalize_actions(chunk_detokenized, quantiles=veh_norm_info)
    print(chunk_denormlized)

    error = np.sqrt(np.square(chunk_delta - chunk_denormlized).sum(axis=-1)).mean()
    print(error)
