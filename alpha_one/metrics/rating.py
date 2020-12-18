from abc import ABC, abstractmethod

class RatingSystem(ABC):
    
    @abstractmethod
    def calculate_ratings(self, match_outcomes):
        pass


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
        