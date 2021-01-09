from collections import defaultdict
from .rating import RatingSystem


class EloRatingSystem(RatingSystem):

    def __init__(self, k_factor, initial_elo=0, elo_width=400):
        self.k_factor = k_factor
        self.elo_width = elo_width
        self.initial_elo = initial_elo
        self.players = defaultdict(lambda: initial_elo)

    def update_ratings(self, match_outcomes):
        for match_outcome in match_outcomes:
            winning_rank = match_outcome.ranks.index(1)
            winning_player = match_outcome.players[winning_rank]
            losing_player = match_outcome.players[1 - winning_rank]
            winner_elo, loser_elo = self._update_elo(self.players[winning_player], self.players[losing_player])
            self.players[winning_player] = winner_elo
            self.players[losing_player] = loser_elo

    def get_rating(self, player_id):
        return self.players[player_id]

    def get_players(self):
        return list(self.players.keys())

    def add_player(self, player_id, initial_rating: float = None):
        assert player_id not in self.players, f"Player {player_id} is already registered!"
        if initial_rating is None:
            self.players[player_id] = self.initial_elo
        else:
            self.players[player_id] = initial_rating

    def calculate_win_probability(self, elo_a, elo_b):
        """
        https://en.wikipedia.org/wiki/Elo_rating_system#Mathematical_details
        """
        return 1.0 / (1 + 10 ** ((elo_b - elo_a) / self.elo_width))

    def _update_elo(self, winner_elo, loser_elo):
        """
        https://en.wikipedia.org/wiki/Elo_rating_system#Mathematical_details
        """
        expected_win = self.calculate_win_probability(winner_elo, loser_elo)
        change_in_elo = self.k_factor * (1 - expected_win)
        winner_elo += change_in_elo
        loser_elo -= change_in_elo
        return winner_elo, loser_elo
