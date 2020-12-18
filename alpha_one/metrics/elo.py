from collections import defaultdict
from .rating import RatingSystem

class EloRatingSystem(RatingSystem):
    
    def __init__(self, k_factor, initial_elo = 0, elo_width=400):
        self.k_factor = k_factor
        self.elo_width = elo_width
        self.players = defaultdict(lambda: initial_elo)
    
    def calculate_ratings(self, match_outcomes):
        for match_outcome in match_outcomes:
            winning_player = match_outcome.players[match_outcome.ranks.index(1)]
            losing_player = 1 - winning_player
            winner_elo, loser_elo = self._update_elo(self.players[winning_player], self.players[losing_player])
            self.players[winning_player] = winner_elo
            self.players[losing_player] = loser_elo
        return self.players
    
    def _update_elo(self, winner_elo, loser_elo):
        """
        https://en.wikipedia.org/wiki/Elo_rating_system#Mathematical_details
        """
        expected_win = self._expected_result(winner_elo, loser_elo)
        change_in_elo = self.k_factor * (1-expected_win)
        winner_elo += change_in_elo
        loser_elo -= change_in_elo
        return winner_elo, loser_elo
    
    def _expected_result(self, elo_a, elo_b):
        """
        https://en.wikipedia.org/wiki/Elo_rating_system#Mathematical_details
        """
        expect_a = 1.0/(1+10**((elo_b - elo_a)/self.elo_width))
        return expect_a
        