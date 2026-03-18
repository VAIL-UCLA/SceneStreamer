"""
Only the TRAINING_DATA_DIR will be used in the code below.
Usage:

python -m scenestreamer.dataset.make_lmdb \
--config-name="1024_gpt" DATA.TEST_DATA_DIR='data/20scenarios' \
DATA.TRAINING_DATA_DIR="/data_zhenghao/datasets/scenarionet/CAT_waymo_hybrid/"

"""
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

    def _open_new_lmdb(self, db_index):
        """Opens a new LMDB file for saving samples."""
        db_path = f"{self.base_path}/data_{db_index}.lmdb"
        return lmdb.open(db_path, map_size=self.max_size)

    def _save_a_batch(self):

        try:
            # Commit the transaction if we have reached the commit interval
            # if (not hasattr(self, 'txn')) or (self.txn is None):
            #     self.txn = self.current_db.begin(write=True)  # Start a new transaction
            #
            #
            # for key, data in self.sample_buffer:
            #     self.txn.put(key.encode('ascii'), pickle.dumps(data))
            #     self.lookup[key] = f"data_{self.current_db_index}.lmdb"
            #
            #     if hasattr(self, 'txn') and self.txn:
            #         self.txn.commit()  # Commit the transaction
            print(f"Saving {len(self.sample_buffer)} samples to data_{self.current_db_index}.lmdb")
            with self.current_db.begin(write=True) as txn:
                for key, data in self.sample_buffer:
                    txn.put(key.encode('ascii'), pickle.dumps(data))
                    self.lookup[key] = f"data_{self.current_db_index}.lmdb"

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


def preprocess_and_queue_worker(worker_id, config, indices, queue):
    """
    This function runs in each worker to preprocess samples and send them to the write queue.
    The writer process will handle writing to LMDB.
    """
    print(f"Worker {worker_id} started.")
    dataset = SceneStreamerDataset(config, "training")

    print(f"Worker {worker_id} has {len(dataset)} samples.")

    # Process and queue each sample assigned to this worker
    if worker_id == 0:
        pbar = tqdm.tqdm(indices, desc="Worker %d" % worker_id)
    else:
        pbar = indices
    print(f"Worker {worker_id} has {len(indices)} samples.")

    for i in pbar:
        sample = dataset[i]  # Access the sample using its index

        # Simulate some preprocessing (replace with actual preprocessing logic)
        file_name, processed_sample = sample["file_name"], sample

        # Put the preprocessed sample into the queue to be written by the writer process
        print(f"Worker {worker_id} processed {file_name}")
        queue.put((file_name, processed_sample))

    # Signal that this worker is done
    # queue.put(None)  # 'None' signals that the worker is done


def write_process(queue, base_path, max_size):
    """
    The write process receives samples from the queue and writes them to the LMDB environment.
    """
    writer = LMDBBulkWriter(base_path=base_path, max_size=max_size)
    print("Writer process started.")

    while True:

        # Blocking if no data is available
        data = queue.get()

        if data == 100:
            print("Received 100, stopping writer process.")
            # If 'None' is received, this indicates that a worker has finished
            break

        if data is None:
            print("Received None, stopping writer process.")
            continue

        file_name, processed_sample = data
        print(f"Saved {file_name} to LMDB")
        writer.save_sample(file_name, processed_sample)

    # Close the writer once all workers are done
    writer.close()


@hydra.main(version_base=None, config_path=str(REPO_ROOT / "cfgs"), config_name="1024_gpt.yaml")
def make_lmdb(config):
    omegaconf.OmegaConf.set_struct(config, False)
    omegaconf.OmegaConf.set_struct(config, True)

    dataset = SceneStreamerDataset(config, "training")
    folder = pathlib.Path(dataset.data_dir)
    folder = folder / "cache"
    folder.mkdir(parents=True, exist_ok=False)

    # Initialize the LMDBBulkWriter
    print("Saving data to LMDB folder:", folder.absolute())

    # num_workers = mp.cpu_count()
    num_workers = 2

    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    chunk_size = dataset_size // num_workers

    # Split the indices into chunks, one for each worker
    chunked_indices = [indices[i * chunk_size:(i + 1) * chunk_size] for i in range(num_workers)]
    # The final chunk may have more samples if the dataset size is not divisible by the number of workers.
    chunked_indices[0].extend(indices[num_workers * chunk_size:])

    # Create a multiprocessing queue
    queue = mp.Queue()

    # Create and start the writer process
    writer_process = mp.Process(target=write_process, args=(queue, folder, 1e10))

    writer_process.start()

    # Create a multiprocessing pool for parallel processing (preprocessing)
    pool = mp.Pool(num_workers)

    results = []
    # Start each worker process, passing its chunk of indices
    for worker_id, worker_indices in enumerate(chunked_indices):
        print(f"Starting worker {worker_id} with {len(worker_indices)} samples.")
        result = pool.apply_async(preprocess_and_queue_worker, args=(worker_id, config, worker_indices, queue))
        results.append(result)

    # Wait for all worker processes to complete
    # for result in results:
    #     result.get()  # This will block until the worker completes its task
    # preprocess_and_queue_worker(0, config, chunked_indices[0], queue)
    pool.close()
    print("Waiting for workers to finish...")
    pool.join()
    print("All workers finished.")

    # Signal the writer process to stop (send 'None' once all workers are done)
    queue.put(100)
    # Wait for the writer process to finish
    writer_process.join()


@hydra.main(version_base=None, config_path=str(REPO_ROOT / "cfgs"), config_name="motion_default.yaml")
def debug(config):
    omegaconf.OmegaConf.set_struct(config, False)
    omegaconf.OmegaConf.set_struct(config, True)
    dataset = SceneStreamerDataset(config, "training")
    folder = pathlib.Path(dataset.data_dir)
    folder = folder / "cache"
    folder.mkdir(parents=True, exist_ok=True)
    for i, sample in enumerate(tqdm.tqdm(dataset, total=len(dataset), desc="Scenarios")):
        file_name = sample["file_name"]


if __name__ == '__main__':
    make_lmdb()
    # debug()
