from collections import defaultdict
from trueskill import Rating, rate
from .rating import RatingSystem


class TrueSkillRatingSystem(RatingSystem):

    def __init__(self):
        self.players = defaultdict(lambda: Rating())

    def update_ratings(self, match_outcomes):
        for match_outcome in match_outcomes:
            new_ratings = rate([[self.players[player_id]] for player_id in match_outcome.players],
                               ranks=match_outcome.ranks)
            for player_id, new_rating in zip(match_outcome.players, new_ratings):
                self.players[player_id] = new_rating[0]

    def get_rating(self, player_id):
        return self.players[player_id]

    def get_players(self):
        return list(self.players.keys())

    def add_player(self, player_id, initial_rating: Rating = None):
        assert player_id not in self.players, f"Player {player_id} is already registered!"
        if initial_rating is None:
            self.players[player_id] = Rating()
        else:
            self.players[player_id] = initial_rating
