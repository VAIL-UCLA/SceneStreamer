
import random

_total_timesteps = 1_000_000
min_prob = 0.5

for num_timesteps in range(0, 1000000, 10000):
    prob = max(1 - (2 * num_timesteps / _total_timesteps) * (1 - min_prob), min_prob)
    # prob = max(min(1 - (2 * num_timesteps / _total_timesteps) * (1 - min_prob), 0.5), min_prob)
    print(f"prob: {prob:.4f}")