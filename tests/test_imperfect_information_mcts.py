"""
Run with `python -m unittest discover tests/`
"""

import unittest
import pyspiel

from open_spiel.python.algorithms.mcts import SearchNode

from alpha_one.alg.imperfect_information import BasicImperfectInformationMCTSEvaluator
from alpha_one.alg.mcts import ImperfectInformationMCTSBot
from alpha_one.game.dummy import DummyGame
from alpha_one.game.information_set import InformationSetGenerator


class TestImperfectInformationMCTS(unittest.TestCase):

    @staticmethod
    def child_with_action(node, action):
        for child in node.children:
            if child.action == action:
                return child
        raise ValueError(f"Could not find action {action}")

    @staticmethod
    def dg_child_with_guess_and_action(node, guess, action):
        for child in node.children:
            if len(child.state.player_1_cards_played) > len(child.state.player_2_cards_played):
                last_card_played = child.state.player_1_cards_played[-1]
            elif len(child.state.player_2_cards_played) > len(child.state.player_1_cards_played):
                last_card_played = child.state.player_2_cards_played[-1]
            else:
                if child.state.get_current_turn() % 2 == 0:
                    last_card_played = child.state.player_2_cards_played[-1]
                else:
                    last_card_played = child.state.player_1_cards_played[-1]

            if last_card_played == guess:
                return TestImperfectInformationMCTS.child_with_action(child, action)
        raise ValueError(f"Could not find child with guess {guess} and action {action}")

    @staticmethod
    def kuhn_child_with_guess_and_action(node, guess, action):
        guess_child = TestImperfectInformationMCTS.kuhn_get_guess(node, guess)
        if guess_child is not None:
            return TestImperfectInformationMCTS.child_with_action(guess_child, action)
        else:
            raise ValueError(f"Could not find child with guess {guess} and action {action}")

    @staticmethod
    def kuhn_get_guess(node, guess):
        for child in node.children:
            observation = child.state.observation_string(1 - child.children[0].player)
            if int(observation[0]) == guess:
                return child
        return None

    @staticmethod
    def create_ii_mcts_bot(game, n_simulations=500, uct_c=3, solve=False):
        ii_mcts_bot = ImperfectInformationMCTSBot(game,
                                                  uct_c,
                                                  n_simulations,
                                                  BasicImperfectInformationMCTSEvaluator(),
                                                  optimism=1.0,
                                                  solve=solve,
                                                  child_selection_fn=SearchNode.puct_value)
        return ii_mcts_bot

    def test_dummy_game_players_and_rewards(self):
        game = DummyGame()

        ii_mcts_bot = self.create_ii_mcts_bot(game)
        information_set_generator = InformationSetGenerator(game)
        state = game.new_initial_state()

        # Player 0
        action = 0
        state.apply_action(action)
        information_set_generator.register_action(action)
        information_set_generator.register_observation(state)

        ii_node, _ = ii_mcts_bot.mcts_search(information_set_generator)
        self.assertEqual(ii_node.player, 1)  # Player 1 guesses
        self.assertTrue(ii_node.is_observation_tree_node())
        self.assertEqual(ii_node.children[0].player, 1)  # Player 1 chooses
        self.assertTrue(ii_node.children[0].is_game_tree_node())
        self.assertGreaterEqual(self.dg_child_with_guess_and_action(ii_node, 0, 1).total_reward, 0)
        self.assertGreaterEqual(self.dg_child_with_guess_and_action(ii_node, 1, 2).total_reward, 0)
        self.assertGreaterEqual(self.dg_child_with_guess_and_action(ii_node, 2, 0).total_reward, 0)

        # Player 1
        action = 1
        state.apply_action(action)
        information_set_generator.register_action(action)
        information_set_generator.register_observation(state)

        ii_node, _ = ii_mcts_bot.mcts_search(information_set_generator)
        self.assertEqual(ii_node.player, 1)  # Player 1 guesses
        self.assertTrue(ii_node.is_observation_tree_node())
        self.assertEqual(ii_node.children[0].player, 1)  # Player 1 chooses
        self.assertTrue(ii_node.children[0].is_game_tree_node())
        self.assertGreaterEqual(self.dg_child_with_guess_and_action(ii_node, 1, 0).total_reward, 0)
        self.assertGreaterEqual(self.dg_child_with_guess_and_action(ii_node, 1, 2).total_reward, 0)

        # Player 1
        action = 0
        state.apply_action(action)
        information_set_generator.register_action(action)
        information_set_generator.register_observation(state)

        ii_node, _ = ii_mcts_bot.mcts_search(information_set_generator)
        self.assertEqual(ii_node.player, 0)  # Player 0 guesses
        self.assertTrue(ii_node.is_observation_tree_node())
        self.assertEqual(ii_node.children[0].player, 0)  # Player 0 chooses
        self.assertTrue(ii_node.children[0].is_game_tree_node())
        self.assertEqual(self.dg_child_with_guess_and_action(ii_node, 0, 1).total_reward, 0)  # Remis
        self.assertLess(self.dg_child_with_guess_and_action(ii_node, 0, 2).total_reward, 0)  # Defeat
        self.assertLess(self.dg_child_with_guess_and_action(ii_node, 2, 1).total_reward, 0)  # Defeat
        self.assertEqual(self.dg_child_with_guess_and_action(ii_node, 2, 2).total_reward, 0)  # Remis

        # Player 0
        action = 2
        state.apply_action(action)
        information_set_generator.register_action(action)
        information_set_generator.register_observation(state)

        ii_node, _ = ii_mcts_bot.mcts_search(information_set_generator)
        self.assertEqual(ii_node.player, 0)  # Player 0 guesses
        self.assertTrue(ii_node.is_observation_tree_node())
        self.assertEqual(ii_node.children[0].player, 0)  # Player 0 chooses
        self.assertTrue(ii_node.children[0].is_game_tree_node())
        self.assertLess(self.dg_child_with_guess_and_action(ii_node, 2, 1).total_reward, 0)  # Defeat

        # ii_node = ii_mcts_bot.mcts_search(information_set_generator)
        # print(ii_node)
        # for s1 in ii_node.children:
        #     print(f"Player {s1.player} guess action {s1.action}, reward: {s1.total_reward}")
        #     for s2 in s1.children:
        #         print("  ", s2.action, s2.explore_count, s2.total_reward)

    def test_kuhn_guess_states_correspond_info_state(self):
        game = pyspiel.load_game('kuhn_poker')
        state = game.new_initial_state()
        ii_mcts_bot = self.create_ii_mcts_bot(game, uct_c=3)
        information_set_generator = InformationSetGenerator(game)

        action = 1  # Player 0 gets card 1
        state.apply_action(action)
        information_set_generator.register(state, action)

        action = 2  # Player 1 gets card 2
        state.apply_action(action)
        information_set_generator.register(state, action)

        action = 0  # Player 0 passes
        state.apply_action(action)
        information_set_generator.register(state, action)

        root, root_root = ii_mcts_bot.mcts_search(information_set_generator)
        # Opponent should guess that player 1 has 0 or 2

        self.assertEqual(len(root.children), 2)  # Two possible guesses, opponent has card 0 or card 1

        opponent_guess = self.kuhn_child_with_guess_and_action(root, 0,
                                                               1)  # Guess that opponent has card 0 (which is not the case), and bet
        self.assertIsNotNone(self.kuhn_get_guess(opponent_guess, 2))
        self.assertIsNotNone(self.kuhn_get_guess(opponent_guess, 1))
        self.assertEqual(len(opponent_guess.children), 2)
        print([s.state.observation_string(1) for s in opponent_guess.children])

        opponent_guess = self.kuhn_child_with_guess_and_action(root, 1, 1)  # Guess that opponent has card 1, and bet
        self.assertIsNotNone(self.kuhn_get_guess(opponent_guess, 2))
        self.assertIsNotNone(self.kuhn_get_guess(opponent_guess, 0))
        self.assertEqual(len(opponent_guess.children), 2)

        opp_win = self.kuhn_child_with_guess_and_action(opponent_guess, 0, 1)
        self.assertGreater(opp_win.total_reward, 0)
        self.assertGreater(self.kuhn_get_guess(opponent_guess, 0).total_reward, 0)
        print(opp_win.state.returns())
        print(opp_win.state.observation_string(0))
        print(opp_win.state.observation_string(1))

        # Some debug output
        root.investigate()
        root_root.investigate()

    def test_leduc_poker(self):
        game = pyspiel.load_game('leduc_poker')
        state = game.new_initial_state()
        ii_mcts_bot = self.create_ii_mcts_bot(game, uct_c=5, n_simulations=500)
        information_set_generator = InformationSetGenerator(game)

        action = 1  # Player 0 gets card 1
        state.apply_action(action)
        information_set_generator.register(state, action)

        action = 4  # Player 1 gets card 4
        state.apply_action(action)
        information_set_generator.register(state, action)

        action = 1  # Player 0 passes
        state.apply_action(action)
        information_set_generator.register(state, action)

        # action = 1  # Player 1 passes
        # state.apply_action(action)
        # information_set_generator.register(state, action)
        #
        # action = 5  # Public card is 5
        # state.apply_action(action)
        # information_set_generator.register(state, action)
        #
        # action = 1  # Player 1 passes
        # state.apply_action(action)
        # information_set_generator.register(state, action)

        print(information_set_generator.current_player())
        print(state.current_player())
        root, root_root = ii_mcts_bot.mcts_search(information_set_generator)
        root.investigate()
