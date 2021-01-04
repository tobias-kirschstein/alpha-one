import random

class ReplayBuffer(object):
  #A fixed size buffer that keeps the newest values.

    def __init__(self, max_size):
        self.max_size = max_size
        self.data = []
        self.total_seen = 0  # The number of items that have passed through.

    def __len__(self):
        return len(self.data)

    def __bool__(self):
        return bool(self.data)

    def append(self, val):
        return self.extend([val])

    def extend(self, batch):
        batch = list(batch)
        self.total_seen += len(batch)
        self.data.extend(batch)
        self.data[:-self.max_size] = []

    def sample(self, count):
        return random.sample(self.data, count)

    def get_total_samples(self):
        return self.total_seen