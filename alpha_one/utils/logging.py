import re
from abc import ABC, abstractmethod
from glob import glob

from torch.utils.tensorboard import SummaryWriter
from pathlib import Path


def generate_run_name(run_dir, run_prefix, match_arbitrary_suffixes=False):
    """
    Generates a new run name by searching for existing runs and adding 1 to the one with the highest ID.
    Assumes that runs will be stored in folder run_dir and have format "{run_prefix}-{run_id}".
    If `arbitrary_suffixes` = True the assumed format is less strict, i.e., folders/files in `run_dir`will be counted if
    they have the format "{run_prefix}-{run_id}*" instead.

    Parameters
    ----------
    run_dir:
        In which folders the runs are stored
    run_prefix:
        Prefix assumed for the run names. Searches for names with format "{run_prefix}-{run_id}"
    match_arbitrary_suffixes
        If set, searches for "{run_prefix}-{run_id}*" instead

    Returns
    -------
        A run name with format "{run_prefix}_{run_id}" where run_id is one larger than what is already found in `run_dir`
    """

    if match_arbitrary_suffixes:
        regex_string = rf"{run_prefix}-(\d+)"
    else:
        regex_string = rf"{run_prefix}-(\d+)$"
    regex = re.compile(regex_string)
    run_names = glob(f"{run_dir}/{run_prefix}-*")
    run_names = [Path(run_name).stem for run_name in run_names]
    run_ids = [int(regex.search(run_name).group(1)) for run_name in run_names if regex.match(run_name)]

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

    def log_hyperparameters(self, hyperparameters: dict):
        self.writer.add_hparams(hyperparameters, dict())
