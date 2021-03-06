import random
from open_spiel.python.algorithms.alpha_zero import model as model_lib


class ReplayBuffer(object):
    # A fixed size buffer that keeps the newest values.

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

    def sample(self, count, observation_key=None, n_most_recent=0):
        """
        Parameters
        ----------
        count:
            How many random samples to obtain from the buffer.
        observation_key:
            If the corresponding game trajectories were generated with `omniscient_observer = True`, then the stored
            observations will be dictionaries. By specifying an `observation_key` one can map to the corresponding value
            of the observation dictionary (e.g., "omniscient_observation" or "player_observation")
        n_most_recent:
            Whether to restrict sampling to only the most recent entries.

        Returns
        -------
            A random sample of data entries drawn from the buffer
        """
        assert (n_most_recent is None or not n_most_recent or n_most_recent == 0) or count < n_most_recent, f"n_most_recent {n_most_recent} must be larger than count {count}"

        sampled_train_inputs = random.sample(self.data[-n_most_recent:], min(count, len(self.data)))
        if observation_key is not None:
            # There are multiple observations in the game trajectory. We have to pick one to feed into the model
            sampled_train_inputs = [
                model_lib.TrainInput(sample.observation[observation_key],
                                     sample.legals_mask,
                                     sample.policy,
                                     sample.value)
                for sample in sampled_train_inputs]
        return sampled_train_inputs

    def get_total_samples(self):
        return self.total_seen
