from alpha_one.utils.mcts import MCTSConfig, initialize_bot


class MCTSAgent:

    def __init__(self, game, model, mcts_config: MCTSConfig):
        self.mcts_bot = initialize_bot(game, model, mcts_config.uct_c,
                                       mcts_config.max_mcts_simulations,
                                       mcts_config.policy_epsilon, mcts_config.policy_alpha)

    def next_move(self):
        pass
