from alpha_one.model.model_manager import ModelManager, CheckpointManager
from alpha_one.utils.io import save_pickled, load_pickled, list_file_numbering


class CFRModelManager(ModelManager):

    def __init__(self, game_name, prefix):
        super(CFRModelManager, self).__init__(game_name, prefix, CFRCheckpointManager)


class CFRCheckpointManager(CheckpointManager):

    def __init__(self, game_name, run_name):
        super(CFRCheckpointManager, self).__init__(game_name, run_name)

    def store_checkpoint(self, model, iteration):
        save_pickled(model, f"{self.model_store_path}/checkpoint-{iteration}")

    def _load_checkpoint(self, iteration, **kwargs):
        return load_pickled(f"{self.model_store_path}/checkpoint-{iteration}")

    def list_checkpoints(self):
        return list_file_numbering(self.model_store_path, "checkpoint", ".p")

    def _build_model(self, config: dict):
        raise NotImplementedError()
