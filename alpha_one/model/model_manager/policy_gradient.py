from pathlib import Path

from python.algorithms.policy_gradient import PolicyGradient

from alpha_one.model.config.policy_gradient import PolicyGradientConfig
from alpha_one.model.model_manager import ModelManager
from alpha_one.utils.io import list_file_numbering


class PolicyGradientManager(ModelManager):

    def store_model(self, model: PolicyGradient, iteration):
        Path(f"{self.model_store_path}/PG-{iteration}").mkdir(parents=True, exist_ok=True)
        model.save_checkpoint(iteration)
        model.save(f"{self.model_store_path}/PG-{iteration}")

    def load_model(self, iteration):
        model = self._build_model(self.load_config())
        model.restore(f"{self.model_store_path}/PG-{iteration}")

    def list_models(self):
        return list_file_numbering(self.model_store_path, "PG")

    def _build_model(self, config: PolicyGradientConfig):
        return PolicyGradient(
            **config
        )
