from abc import ABC, abstractmethod


class RatingSystem(ABC):

    @abstractmethod
    def update_ratings(self, match_outcomes):
        pass

    @abstractmethod
    def get_rating(self, player_id):
        pass

    @abstractmethod
    def get_players(self):
        pass

    @abstractmethod
    def add_player(self, player_id, initial_rating=None):
        pass

    def get_ratings(self):
        return {player_id: self.get_rating(player_id) for player_id in self.get_players()}


class MatchOutcome:

    def __init__(self, players, ranks, rewards=None):
        self.players = players
        self.ranks = ranks
        self.rewards = rewards

    @staticmethod
    def win(player1, player2, rewards=None):
        if rewards is None:
            rewards = [1, -1]
        return MatchOutcome([player1, player2], ranks=[1, 2], rewards=rewards)

    @staticmethod
    def defeat(player1, player2, rewards=None):
        if rewards is None:
            rewards = [-1, 1]
        return MatchOutcome([player1, player2], ranks=[2, 1], rewards=rewards)

    @staticmethod
    def draw(player1, player2):
        return MatchOutcome([player1, player2], ranks=[1, 1], rewards=[0, 0])

    def rename_player(self, old_player_name, new_player_name):
        self.players[self.players.index(old_player_name)] = new_player_name

    def with_renamed_players(self, player_name_mapping: dict):
        self.players = [player_name_mapping[player_id]
                        if player_id in player_name_mapping
                        else player_id
                        for player_id
                        in self.players]
        return self
