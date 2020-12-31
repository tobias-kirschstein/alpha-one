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

    def get_ratings(self):
        return {player_id: self.get_rating(player_id) for player_id in self.get_players()}


class MatchOutcome:
    
    def __init__(self, players, ranks):
        self.players = players
        self.ranks = ranks
        
    @staticmethod
    def win(player1, player2):
        return MatchOutcome([player1, player2], ranks=[1, 2])
    
    @staticmethod
    def defeat(player1, player2):
        return MatchOutcome([player1, player2], ranks=[2, 1])
    
    @staticmethod
    def draw(player1, player2):
        return MatchOutcome([player1, player2], ranks=[1, 1])
        