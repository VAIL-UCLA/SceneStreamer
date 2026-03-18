from scenestreamer.tokenization.biycle_tokenizer import BicycleModelTokenizerFixed0124
from scenestreamer.tokenization.diffusion_tokenizer import DiffusionTokenizer, SPECIAL_INVALID, SPECIAL_START, SPECIAL_VALID
from scenestreamer.tokenization.motion_tokenizers import DeltaDeltaTokenizer, START_ACTION, END_ACTION, DeltaTokenizer


def get_tokenizer(config):
    if config.USE_DIFFUSION:
        from scenestreamer.tokenization.diffusion_tokenizer import DiffusionTokenizer
        return DiffusionTokenizer(config)

    if config.TOKENIZATION.TOKENIZATION_METHOD == "delta":
        return DeltaTokenizer(config)
    elif config.TOKENIZATION.TOKENIZATION_METHOD == "delta_delta":
        return DeltaDeltaTokenizer(config)
    elif config.TOKENIZATION.TOKENIZATION_METHOD == "precomputed_delta_delta":
        return PrecomputedDeltaDeltaTokenizer(config)
    elif config.TOKENIZATION.TOKENIZATION_METHOD == "bicycle":
        raise ValueError()
        return BicycleModelTokenizer(config)
    elif config.TOKENIZATION.TOKENIZATION_METHOD == "bicycle_noavg":
        raise ValueError()
        return BicycleModelTokenizerNoAVG(config)
    elif config.TOKENIZATION.TOKENIZATION_METHOD == "bicycle_interpolated":
        return BicycleModelInterpolatedTokenizer(config)
    elif config.TOKENIZATION.TOKENIZATION_METHOD == "BicycleModelTokenizerFixed0124":
        return BicycleModelTokenizerFixed0124(config)
    elif config.TOKENIZATION.TOKENIZATION_METHOD == "fast":
        from scenestreamer.tokenization.fast_tokenizer import FastTokenizer
        return FastTokenizer(config)
    else:
        raise ValueError("Unknown tokenizer: {}".format(config.TOKENIZATION.TOKENIZATION_METHOD))


def get_action_dim(config):
    t = get_tokenizer(config)
    return t.num_actions
