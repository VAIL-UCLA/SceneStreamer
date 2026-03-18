import random
from metadrive.policy.replay_policy import ReplayEgoCarPolicy
from metadrive.envs import ScenarioEnv
import argparse
import mediapy
import numpy as np
from scenestreamer.utils import REPO_ROOT

extra_args = dict(film_size=(900, 600), screen_size=(900, 600))


def render(input_dir, output_path):
    try:
        env = ScenarioEnv(
            {
                "manual_control": False,
                "reactive_traffic": False,
                "use_render": False,
                "agent_policy": ReplayEgoCarPolicy,
                "data_directory": REPO_ROOT / input_dir,
                "num_scenarios": 1
            }
        )
        o, _ = env.reset()
        frames = []
        for i in range(1, 100000):
            o, r, tm, tc, info = env.step([1.0, 0.])
            frame = env.render(mode="top_down", **extra_args)
            frames.append(frame)
            if tm or tc:
                break

    except Exception as e:
        raise e
    finally:
        env.close()

    imgs = np.stack([frame for frame in frames], axis=0)
    mediapy.write_video(REPO_ROOT / output_path, imgs, fps=20)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()
    render(args.input_dir, args.output_path)
