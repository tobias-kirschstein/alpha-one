from open_spiel.python.algorithms import get_all_states
import pyspiel


def state_to_value(game_name):
    game = pyspiel.load_game(game_name)
    states = get_all_states.get_all_states(game)
    state_to_value_dict = {}
    state_id = 0
    for key, values in states.items():
        state_to_value_dict[values.__str__()] = state_id
        state_id += 1
    return state_to_value_dict
