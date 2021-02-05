from typing import List
import pyspiel


class DummyGame:
    def __init__(self, n_cards=3):
        self.n_cards = n_cards

    def new_initial_state(self):
        return DummyGameState(self, 0, list(range(self.n_cards)), list(range(self.n_cards)), [], [])

    def get_type(self):
        return pyspiel.GameType('dummy', 'dummy', pyspiel.GameType.Dynamics.SEQUENTIAL,
                                pyspiel.GameType.ChanceMode.DETERMINISTIC,
                                pyspiel.GameType.Information.IMPERFECT_INFORMATION,
                                pyspiel.GameType.Utility.ZERO_SUM, pyspiel.GameType.RewardModel.TERMINAL,
                                2, 2,
                                False, False,
                                True, True,
                                dict())

    def max_utility(self):
        return 1


class DummyGameState:
    def __init__(self,
                 game: DummyGame,
                 current_player: int,
                 player_1_cards: List[int],
                 player_2_cards: List[int],
                 player_1_cards_played: List[int],
                 player_2_cards_played: List[int]):
        self.game = game
        self._current_player = current_player
        self.player_1_cards = player_1_cards
        self.player_2_cards = player_2_cards
        self.player_1_cards_played = player_1_cards_played
        self.player_2_cards_played = player_2_cards_played

    def legal_actions(self, player=None) -> List[int]:
        if player is None:
            player = self._current_player

        if player == 0:
            return self.player_1_cards
        else:
            return self.player_2_cards

    def get_current_turn(self) -> int:
        return min(len(self.player_1_cards_played), len(self.player_2_cards_played)) + 1

    def observation_string(self, player_id: int) -> str:
        current_turn = self.get_current_turn()
        if player_id == 0:
            player_1_cards_played = self.player_1_cards_played
            player_2_cards_played = self.player_2_cards_played[:current_turn - 1]
            player_1_cards = self.player_1_cards
            player_2_cards = []
        else:
            player_1_cards_played = self.player_1_cards_played[:current_turn - 1]
            player_2_cards_played = self.player_2_cards_played
            player_1_cards = []
            player_2_cards = self.player_2_cards
        string = f"{self._current_player}, {player_1_cards_played}, {player_2_cards_played}, {player_1_cards}, {player_2_cards}"
        return string

    def observation_tensor(self, player_id: int) -> str:
        return self.observation_string(player_id)

    def apply_action(self, action: int):
        assert action in self.legal_actions(), f"{action} not allowed"

        if self.get_current_turn() % 2 == 1:
            next_player = 1
        else:
            next_player = 0

        if self._current_player == 0:
            player_1_cards = list(self.player_1_cards)
            player_1_cards_played = list(self.player_1_cards_played)
            player_1_cards.remove(action)
            player_1_cards_played.append(action)
            player_2_cards = self.player_2_cards
            player_2_cards_played = self.player_2_cards_played
        else:
            player_1_cards = self.player_1_cards
            player_1_cards_played = self.player_1_cards_played
            player_2_cards = list(self.player_2_cards)
            player_2_cards_played = list(self.player_2_cards_played)
            player_2_cards.remove(action)
            player_2_cards_played.append(action)

        self._current_player = next_player
        self.player_1_cards = player_1_cards
        self.player_2_cards = player_2_cards
        self.player_1_cards_played = player_1_cards_played
        self.player_2_cards_played = player_2_cards_played

    def is_terminal(self) -> bool:
        return len(self.player_1_cards) == 0 and len(self.player_2_cards) == 0

    def is_chance_node(self) -> bool:
        return False

    def current_player(self) -> int:
        return self._current_player

    def clone(self):
        return DummyGameState(self.game, self._current_player, list(self.player_1_cards), list(self.player_2_cards),
                              list(self.player_1_cards_played), list(self.player_2_cards_played))

    def returns(self) -> List[float]:
        assert self.is_terminal(), "Only terminal nodes have a return"
        points_p1 = 0
        points_p2 = 0
        for p1_card, p2_card in zip(self.player_1_cards_played, self.player_2_cards_played):
            if p1_card > p2_card:
                points_p1 += 1
            elif p2_card > p1_card:
                points_p2 += 1

        if points_p1 > points_p2:
            return [1, -1]
        elif points_p2 > points_p1:
            return [-1, 1]
        else:
            return [0, 0]

    def to_str(self):
        return f"{self._current_player}, {self.player_1_cards_played}, {self.player_2_cards_played}, {self.player_1_cards}, {self.player_2_cards}"

    def __str__(self):
        return self.to_str()

