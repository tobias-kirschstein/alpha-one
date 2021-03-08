from open_spiel.python.algorithms import mcts

from alpha_one.alg.imperfect_information import BasicOmniscientMCTSEvaluator, AlphaZeroOmniscientMCTSEvaluator
from alpha_one.model.agent import MCTSAgent
from alpha_one.utils.mcts import MCTSConfig


class OmniscientAgent(MCTSAgent):

    def __init__(self, game, mcts_config: MCTSConfig, model=None):
        if model is None:
            evaluator = BasicOmniscientMCTSEvaluator(game)
        else:
            evaluator = AlphaZeroOmniscientMCTSEvaluator(game, model)

        _omniscient_bot = mcts.MCTSBot(game,
                                       mcts_config.uct_c,
                                       mcts_config.max_mcts_simulations,
                                       evaluator,
                                       solve=False)
        super(OmniscientAgent, self).__init__(game,
                                              _omniscient_bot,
                                              mcts_config.temperature,
                                              temperature_drop=mcts_config.temperature_drop,
                                              use_reward_policy=mcts_config.use_reward_policy)
