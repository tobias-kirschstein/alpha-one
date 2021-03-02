from alpha_one.game.buffer import ReplayBuffer
from alpha_one.utils.io import save_pickled, load_pickled, list_file_numbering
from pathlib import Path


class ReplayDataManager:
    """
    Simple data manager to facilitate loading and storing of replay buffers.
    During a train run, arbitrarily many replay buffers may be stored to disk and the corresponding iteration can be
    indicated as well.
    """

    def __init__(self, data_store_path: str):
        self.data_store_path = data_store_path

    def list_replays(self):
        return list_file_numbering(self.data_store_path, 'replay_buffer', '.p')

    def load_replays(self, iteration: int = None) -> ReplayBuffer:
        if iteration == -1 or iteration is None and not Path(f"{self.data_store_path}/replay_buffer").exists():
            iteration = self.list_replays()[-1]
        store_path = self.get_store_path(iteration)
        return load_pickled(store_path)

    def store_replays(self, replay_buffer: ReplayBuffer, iteration: int = None):
        store_path = self.get_store_path(iteration)
        save_pickled(replay_buffer, store_path)

    def get_store_path(self, iteration: int) -> str:
        store_path = f"{self.data_store_path}/replay_buffer" \
            if iteration is None \
            else f"{self.data_store_path}/replay_buffer-{iteration}"
        return store_path
