import numpy as np


def get_state_mask(state_to_value_dict, information_set):
    state_mask = np.zeros(len(state_to_value_dict), dtype=np.bool)
    track_index = []
    for s in information_set:
        state_mask[state_to_value_dict[s.__str__()]] = 1
        track_index.append(state_to_value_dict[s.__str__()])

    return state_mask, track_index
