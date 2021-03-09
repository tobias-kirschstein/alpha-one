from alpha_one.game.information_set import InformationSetGenerator
from alpha_one.model.agent import DMCTSAgent
from alpha_one.utils.mcts import MCTSConfig
from alpha_one.utils.statemask import get_state_mask


class AlphaOneInformationSetWeightsExtractor:

    def __init__(self, observation_model, state_to_value_dict, n_previous_observations):
        self._observation_model = observation_model
        self._state_to_value_dict = state_to_value_dict
        self._n_previous_observations = n_previous_observations

    def get_information_set_weights(self, information_set_generator: InformationSetGenerator):
        obs = [information_set_generator.get_padded_observation_history(self._n_previous_observations)]

        information_set = information_set_generator.calculate_information_set()
        state_mask, state_indices = get_state_mask(self._state_to_value_dict, information_set)
        value, policy = self._observation_model.inference(obs, [state_mask])
        return policy[0][state_mask]


class HybridAlphaOneDMCTSAgent(DMCTSAgent):

    def __init__(self,
                 d_mcts_model,
                 observation_model,
                 mcts_config: MCTSConfig,
                 state_to_value_dict,
                 n_previous_observations,
                 n_rollouts=None):
        weights_extractor = AlphaOneInformationSetWeightsExtractor(observation_model,
                                                                   state_to_value_dict,
                                                                   n_previous_observations)

        super(HybridAlphaOneDMCTSAgent, self).__init__(d_mcts_model,
                                                       mcts_config,
                                                       n_rollouts=n_rollouts,
                                                       information_set_weights_fn=weights_extractor.get_information_set_weights)
