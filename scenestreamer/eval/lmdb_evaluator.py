"""This is a fake evaluator which save the data to LMDB dataset."""
"""
Script to generate submission files for Waymo SimAgent Challenge.
Please check out the end of this file where we provide a script to merge submission files.
"""
import copy
import os
import pathlib
import uuid
import io
import numpy as np
import torch

from scenestreamer.dataset.preprocessor import centralize_to_map_center
from scenestreamer.eval.waymo_motion_prediction_evaluator import _repeat_for_modes
from scenestreamer.eval.wosac_eval import wosac_evaluation
from scenestreamer.tokenization import get_tokenizer
from scenestreamer.utils import wrap_to_pi, rotate

import json
import os
import pathlib
import pickle
import multiprocessing as mp
from functools import partial
import tqdm
import hydra
import lmdb
import omegaconf
import tqdm

from scenestreamer.dataset.dataset import SceneStreamerDataset

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent


class LMDBBulkWriter:
    def __init__(self, base_path, max_size=1e9):
        """
        Initializes the LMDBBulkWriter to save all data in batches, with map_size for each LMDB file.
        Args:
            base_path: Directory path to save LMDB files.
            max_size: Maximum size of each LMDB file in bytes.
        """
        self.base_path = base_path
        # Create the cache directory if it doesn't exist
        os.makedirs(self.base_path, exist_ok=True)

        self.max_size = int(max_size)  # Set the max LMDB file size (e.g., 1 GB)
        self.current_db_index = 0
        self.lookup = {}  # Lookup table to track which LMDB file stores which sample
        self.current_db = self._open_new_lmdb(self.current_db_index)
        self.per_shard_size = 0

        self.sample_buffer = []

    def _data_to_npz_bytes(self, data):
        """Convert data to .npz compressed bytes."""
        with io.BytesIO() as buffer:
            np.savez_compressed(buffer, **data)  # Save dictionary elements as separate arrays in .npz
            return buffer.getvalue()  # Retrieve the bytes

    def _open_new_lmdb(self, db_index):
        """Opens a new LMDB file for saving samples."""
        db_path = f"{self.base_path}/data_{db_index}.lmdb"
        return lmdb.open(db_path, map_size=self.max_size)

    def _save_a_batch(self):

        try:
            print(f"Saving {len(self.sample_buffer)} samples to data_{self.current_db_index}.lmdb")
            with self.current_db.begin(write=True) as txn:
                for key, data in self.sample_buffer:
                    npz_bytes = self._data_to_npz_bytes(data)
                    txn.put(key.encode('ascii'), npz_bytes)
                    self.lookup[key] = str((self.base_path / f"data_{self.current_db_index}.lmdb").absolute().resolve())

            self.sample_buffer.clear()

        except lmdb.MapFullError:

            # If current LMDB file is full, create a new one and retry saving
            self.current_db.close()
            self.current_db_index += 1
            print(f"Creating new LMDB file: data_{self.current_db_index}.lmdb (size: {self.per_shard_size})")
            self.current_db = self._open_new_lmdb(self.current_db_index)
            self._save_a_batch()
            self.per_shard_size = 0

    def save_sample(self, key, data):
        """Saves a sample to the current LMDB file, switching to a new file if necessary."""
        # Batch writes into a single transaction
        if self.per_shard_size % 100 == 0:
            self._save_a_batch()
        self.sample_buffer.append((key, data))
        self.per_shard_size += 1

    def close(self):
        self._save_a_batch()
        """Closes the LMDB environment and saves the lookup table as a JSON file."""
        self.current_db.close()
        # Save the lookup table to track the LMDB file where each sample is stored
        with open(f"{self.base_path}/lookup.json", "w") as f:
            json.dump(self.lookup, f)


def transform_to_global_coordinate(data_dict):
    map_center = data_dict["metadata/map_center"].reshape(-1, 1, 1, 3)
    map_heading = data_dict["metadata/map_heading"].reshape(-1, 1, 1)
    B, T, N, _ = data_dict["decoder/agent_position"].shape
    map_heading = map_heading.repeat(T, axis=1).repeat(N, axis=2)
    assert map_heading.shape == (B, T, N)
    data_dict["decoder/agent_position"] = rotate(
        x=data_dict["decoder/agent_position"][..., 0],
        y=data_dict["decoder/agent_position"][..., 1],
        angle=map_heading,
        z=data_dict["decoder/agent_position"][..., 2]
    )
    assert data_dict["decoder/agent_position"].ndim == 4
    data_dict["decoder/agent_position"] += map_center

    data_dict["decoder/agent_heading"] = wrap_to_pi(data_dict["decoder/agent_heading"] + map_heading)

    data_dict["decoder/agent_velocity"] = rotate(
        x=data_dict["decoder/agent_velocity"][..., 0],
        y=data_dict["decoder/agent_velocity"][..., 1],
        angle=map_heading,
    )

    data_dict["pred_trajs"] = [
        centralize_to_map_center(
            traj, map_center=-data_dict["expanded_map_center"][b], map_heading=-data_dict["expanded_map_heading"][b]
        ) for b, traj in enumerate(data_dict["pred_trajs"])
    ]

    return data_dict


scenario_metrics_keys = [
    # 'scenario_id',
    'metametric',
    'average_displacement_error',
    'min_average_displacement_error',
    'linear_speed_likelihood',
    'linear_acceleration_likelihood',
    'angular_speed_likelihood',
    'angular_acceleration_likelihood',
    'distance_to_nearest_object_likelihood',
    'collision_indication_likelihood',
    'time_to_collision_likelihood',
    'distance_to_road_edge_likelihood',
    'offroad_indication_likelihood'
]

aggregate_metrics_keys = [
    'realism_meta_metric', 'kinematic_metrics', 'interactive_metrics', 'map_based_metrics', 'min_ade'
]


def scenario_metrics_to_dict(scenario_metrics):
    return {k: getattr(scenario_metrics, k) for k in scenario_metrics_keys}


def aggregate_metrics_to_dict(aggregate_metrics):
    return {k: getattr(aggregate_metrics, k) for k in aggregate_metrics_keys}


import msgpack


class LMDBEvaluator:
    def __init__(self, config):
        self.config = config

        self.writer = None

    def validation_step(
        self, data_dict, batch_idx, model, log_dict_func, global_rank, logger, lightning_model, **kwargs
    ):

        if self.writer is None:
            cache_folder = REPO_ROOT / self.config.DATA.TEST_DATA_DIR / "cache"
            cache_folder.mkdir(parents=True, exist_ok=True)
            cache_folder = cache_folder / "rank_{}".format(global_rank)
            self.writer = LMDBBulkWriter(base_path=cache_folder, max_size=1e10)

        B = data_dict["decoder/input_action_valid_mask"].shape[0]
        for b in range(B):
            data_dict_b = {k: v[b] for k, v in data_dict.items()}
            new_data_dict = {}
            for k, v in data_dict_b.items():
                if isinstance(v, torch.Tensor):
                    new_data_dict[k] = v.cpu().numpy()
                elif isinstance(v, np.ndarray) and v.dtype == np.str_:
                    assert v.shape == ()
                    new_data_dict[k] = v.item()
                else:
                    new_data_dict[k] = v
            self.writer.save_sample(data_dict_b["file_name"], new_data_dict)

    def on_validation_epoch_end(self, *args, global_rank, logger, trainer, **kwargs):
        pass
        self.writer.close()
