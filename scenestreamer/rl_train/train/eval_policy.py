
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import set_random_seed
from functools import partial
from IPython.display import clear_output
import os
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from metadrive.policy.env_input_policy import EnvInputPolicy
from metadrive.envs.scenario_env import ScenarioEnv
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
import logging
from collections import deque
from metadrive.policy.replay_policy import ReplayEgoCarPolicy
import torch
import random
import argparse

def set_seed(seed):
    set_random_seed(seed)
    random.seed(seed)                               # Python random
    np.random.seed(seed)                            # NumPy random
    torch.manual_seed(seed)                         # PyTorch random (CPU and GPU)&#8203;:contentReference[oaicite:2]{index=2}
    torch.cuda.manual_seed_all(seed)                # PyTorch random (all GPU devices)
    torch.backends.cudnn.deterministic = True       # Use deterministic CuDNN operations&#8203;:contentReference[oaicite:3]{index=3}
    torch.backends.cudnn.benchmark = False          # Disable CuDNN benchmark for determinism&#8203;:contentReference[oaicite:4]{index=4}


def evaluate_info(all_info):
	all_scenarios  =  list(all_info.keys())
	num_scenario = len(all_scenarios)

	_rewards = []
	_costs = []
	_completion = []
	_crash = []
	_episode_length = []

	for i in all_scenarios:
		_rewards.append(all_info[i]['episode_reward'])
		_costs.append(all_info[i]['cost'])
		_completion.append(all_info[i]['route_completion'])
		_crash.append(1 if all_info[i]['crash'] else 0)
		_episode_length.append(all_info[i]['episode_length'])


	avg_rewards = sum(_rewards) / num_scenario
	avg_costs = sum(_costs) / num_scenario
	avg_completion = sum(_completion) / num_scenario
	avg_collisions = sum(_crash) / num_scenario
	avg_episode_length = sum(_episode_length) / num_scenario


	result = {"num_scenario": num_scenario, "avg_rewards": avg_rewards, "avg_costs": avg_costs, "avg_completion": avg_completion, "avg_collisions": avg_collisions, "avg_length": avg_episode_length}
	print(result)


	return result


class CustomMonitor(Monitor):
    def __init__(self, env, buffer_size=256, info_keywords=None):
        super().__init__(env, info_keywords=info_keywords)
        self.ep_info_buffer = deque(maxlen=buffer_size)  # Initialize buffer

    def step(self, action):
        obs, reward, tm, tc, info = super().step(action)
        done = tm or tc

        if done:
            info["episode_reward"] = info["episode_reward"]
            info["episode_length"] = info["episode_length"]
            info["route_completion"] = info.get("route_completion", 0)
            info["cost"] = info.get("cost", 0)
            info["crash"] = info.get("crash", 0)

        return obs, reward, tm, tc, info
    
    

def create_env(config, need_monitor=False):
    env = ScenarioEnv(config=config)
    if need_monitor:
        info_keywords = ["episode_reward", "episode_length", "route_completion", "cost", "crash"]
        env = CustomMonitor(env, info_keywords=info_keywords)  # Pass the custom metrics

    return env

def eval_policy(config_test, checkpoint_path=None, eval_episodes=100):
	# Now, save all last step's info instead of manually calculate
    env=create_env(config_test)
    if checkpoint_path:
        model = TD3.load(checkpoint_path)
    else:
        model = TD3("MlpPolicy", 
                    env,
                    action_noise=None, 
                    learning_rate=1e-4,
                    learning_starts=200,
                    batch_size=1024,
                    tau=0.005,
                    gamma=0.99,
                    train_freq=1,
                    gradient_steps=1,
                    device="cuda",
                    seed=0,
                    verbose=2,
                    tensorboard_log="td3_rl",
                    )

    all_info = {}

    for ep_num in range(eval_episodes):

        while True:
            try:
                obs, info = env.reset()
                break

            except:
                continue

        done = False
        episode_timesteps = 0

        collision = False
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, tm, tc, info = env.step(action)

            if (env.vehicle.crash_vehicle):
                # print("collision")
                collision = True

            done = tm or tc or info['arrive_dest'] or info['max_step']
            # done = info['arrive_dest'] or info['max_step']

            episode_timesteps += 1

        # if done and episode_timesteps < 10:
        #     continue # invalid scenario

        if collision:
            info['crash'] = True

        completion = info['route_completion']
        if completion <= 0:
            info['route_completion'] = 0
            
        if completion >= 1:
            info['route_completion'] = 1

        if info['arrive_dest']:
            info['route_completion'] = 1

        all_info[ep_num] = info
        # print("info", info)
            

    results = evaluate_info(all_info)
    env.close()

    return results


def eval_policy_formal(config_test, checkpoint_path=None, eval_episodes=100, episodes_per_env=5):
    env = create_env(config_test)
    
    if checkpoint_path:
        model = TD3.load(checkpoint_path)
    else:
        model = TD3("MlpPolicy", 
                    env,
                    action_noise=None, 
                    learning_rate=1e-4,
                    learning_starts=200,
                    batch_size=1024,
                    tau=0.005,
                    gamma=0.99,
                    train_freq=1,
                    gradient_steps=1,
                    device="cuda",
                    seed=0,
                    verbose=2,
                    tensorboard_log="td3_rl",
                    )

    all_info = {}
    ep_count = 0  # Keep track of total episodes

    for env_num in range(eval_episodes):  # Each environment runs multiple episodes
        for ep in range(episodes_per_env):
            while True:
                try:
                    obs, info = env.reset()
                    break
                except:
                    continue

            done = False
            episode_timesteps = 0
            collision = False

            while not done:
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, tm, tc, info = env.step(action)

                if env.vehicle.crash_vehicle:
                    collision = True

                done = tm or tc or info['arrive_dest'] or info['max_step']
                episode_timesteps += 1

            if collision:
                info['crash'] = True

            completion = info['route_completion']
            info['route_completion'] = max(0, min(1, completion))  # Ensure range [0,1]

            if info['arrive_dest']:
                info['route_completion'] = 1

            all_info[ep_count] = info  # Store per-episode results
            ep_count += 1  # Increment episode count

    results = evaluate_info(all_info)
    env.close()

    return results


def eval_ckpt_for_seeds(config_test, ckpt_root_dir, step_num):
    seeds = [0, 100, 200, 300, 400, 500, 600, 700]
    res_all_seeds = {}

    for seed in seeds:
        ckpt_path = os.path.join(ckpt_root_dir, f"seed_{seed}_{int(step_num)}_steps.zip")
        if not os.path.exists(ckpt_path):
            print(f"Checkpoint not found: {ckpt_path}")
            continue

        print(f"Evaluating checkpoint: {ckpt_path}")
        results = eval_policy_formal(config_test=config_test, checkpoint_path=ckpt_path)
        print(f"Results: {results}")

        res_all_seeds[seed] = results

    avg_results = {}
    std_results = {}
    valid_seeds = list(res_all_seeds.keys())

    # avg results over all seeds and writ to avg_results
    for key in res_all_seeds[valid_seeds[0]]:
        values = np.array([res_all_seeds[seed][key] for seed in valid_seeds])
        avg_results[key] = sum([res_all_seeds[seed][key] for seed in valid_seeds]) / len(valid_seeds)
        std_results[key] = np.std(values, ddof=1)  # Compute standard deviation

    print(f"Average results over all seeds: {avg_results}")
    print(f"Standard deviation over all seeds: {std_results}")
    
    return {"average": avg_results, "std": std_results}



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, help="The dir for held-out envs.")
    parser.add_argument("--ckpt_dir", type=str, help="The dir checkpoints.")
    parser.add_argument("--eval_horizon", type=int, help="Eval horizon in MD.")
    parser.add_argument("--ckpt_steps", type=int, help="number of training steps of the ckpt to eval.")
    args = parser.parse_args()

    TEST_HORIZON = args.eval_horizon
    TEST_DIR= args.data_dir 
    ckpt_root_dir = args.ckpt_dir 
    ckpt_steps = args.ckpt_steps

    config_test = dict(
	    use_render=False,
        manual_control= False,
        show_interface= False,
		data_directory=TEST_DIR, # scenarionet_waymo_training_500
		start_scenario_index=0,
		num_scenarios=100,
		agent_policy=EnvInputPolicy,
        force_render_fps=10,
        reactive_traffic=False,
		sequential_seed = True,
		# force_reuse_object_name = True,
		horizon = TEST_HORIZON,
        out_of_route_done=False,
        crash_vehicle_done=False,
        crash_object_done=False,
        crash_human_done=False,
        relax_out_of_road_done=False,
    )

    config_test_CAT = dict(
            data_directory=TEST_DIR,
            start_scenario_index = 0,
            num_scenarios=100,
            sequential_seed = False,
            agent_policy=EnvInputPolicy,
            force_reuse_object_name = True,
            horizon = 50,
            no_light = True,
            no_static_vehicles = True,
            reactive_traffic = False,
            vehicle_config=dict(
                lidar = dict(num_lasers=30,distance=50, num_others=3),
                side_detector = dict(num_lasers=30),
                lane_line_detector = dict(num_lasers=12)),

            # ===== Reward Scheme =====
            success_reward=10.0,
            out_of_road_penalty=10.0,
            crash_vehicle_penalty=1,
            crash_object_penalty=1.0,
            driving_reward=1.0,
            # speed_reward=0.1,
            # use_lateral_reward=False,

            # ===== Cost Scheme =====
            crash_vehicle_cost=1.0,
            crash_object_cost=1.0,
            out_of_road_cost=1.0,

            # ===== Termination Scheme =====
            out_of_route_done=False,
            crash_vehicle_done=False,
            relax_out_of_road_done=True,        
        )


    eval_ckpt_for_seeds(config_test, ckpt_root_dir, ckpt_steps)


