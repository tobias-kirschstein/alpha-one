from pathlib import Path

from open_spiel.python.algorithms.policy_gradient import PolicyGradient

from alpha_one.model.config.policy_gradient import PolicyGradientConfig
from alpha_one.model.model_manager import CheckpointManager, ModelManager
from alpha_one.utils.io import list_file_numbering
from alpha_one.utils.tf import initialize_session


class PolicyGradientModelManager(ModelManager):

    def __init__(self, game_name):
        super(PolicyGradientModelManager, self).__init__(game_name, 'PG', PolicyGradientCheckpointManager)


class PolicyGradientCheckpointManager(CheckpointManager):

    def __init__(self, game_name, run_name):
        super(PolicyGradientCheckpointManager, self).__init__(game_name, run_name)

    def store_checkpoint(self, model: PolicyGradient, iteration):
        Path(self.model_store_path).mkdir(parents=True, exist_ok=True)
        model.save(f"{self.model_store_path}/checkpoint-{iteration}")

    def _load_checkpoint(self, iteration, player_id=None):
        model = self._build_model(self.load_config(player_id=player_id))
        model.restore(f"{self.model_store_path}/checkpoint-{iteration}")
        return model

    def load_config(self, player_id=None):
        config = super().load_config()
        if player_id is not None:
            config.player_id = player_id
        return config

    def list_checkpoints(self):
        return list_file_numbering(self.model_store_path, "checkpoint")

    def _build_model(self, config: PolicyGradientConfig):
        session = initialize_session()
        return PolicyGradient(
            session=session,
            **config
        )
