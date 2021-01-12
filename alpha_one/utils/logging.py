import re
from abc import ABC, abstractmethod
from glob import glob

from torch.utils.tensorboard import SummaryWriter
from pathlib import Path


def generate_run_name(run_dir, run_prefix):
    """
    Assumes that runs will be stored in folder run_dir and have format "{run_prefix}-{run_id}".
    Generates a new run name by searching for existing runs and adding 1 to the one with the highest ID
    """
    regex = re.compile(f"{run_prefix}-(\d+)$")
    run_names = glob(f"{run_dir}/{run_prefix}-*")
    run_ids = [int(regex.search(Path(run_name).stem).group(1)) for run_name in run_names]

    run_id = max(run_ids) + 1 if len(run_ids) > 0 else 1
    return f"{run_prefix}-{run_id}"

class MetricsLogger(ABC):

    @abstractmethod
    def log_scalar(self, name, value, step, timestep=None, **kwargs):
        pass

    def log_scalars(self, name, values_dict: dict, step, timestep=None, **kwargs):
        pass

    def flush(self):
        pass


class TensorboardLogger(MetricsLogger):

    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir)

    def log_scalar(self, name, value, step, timestep=None, **kwargs):
        self.writer.add_scalar(name, value, global_step=step, walltime=timestep)

    def log_scalars(self, name, values_dict: dict, step, timestep=None, **kwargs):
        self.writer.add_scalars(name, values_dict, global_step=step, walltime=timestep)

    def flush(self):
        self.writer.flush()

    def log_hyperparameters(self, hyperparameters:dict):
        self.writer.add_hparams(hyperparameters, dict())
