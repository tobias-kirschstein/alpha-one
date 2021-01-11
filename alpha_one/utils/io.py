import os
import pickle
from glob import glob
from pathlib import Path
import re


def save_pickled(obj, file):
    file = _file_ending(file, "p")
    create_directories(file)
    with open(f"{file}", 'wb') as f:
        pickle.dump(obj, f)


def load_pickled(file):
    file = _file_ending(file, "p")
    with open(file, 'rb') as f:
        return pickle.load(f)


def _file_ending(file, ending):
    return f"{file}.{ending}" if f".{ending}" not in file else file


def create_directories(path):
    Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)


def list_file_numbering(directory, prefix, suffix=None):
    if suffix is None:
        suffix = ""
    regex = re.compile(f"{prefix}-(-?\d+)$")
    file_names = glob(f"{directory}/{prefix}-*{suffix}")
    numbering = sorted([int(regex.search(Path(file_name).stem).group(1)) for file_name in file_names])
    return numbering