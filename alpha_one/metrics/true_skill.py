from collections import defaultdict
from trueskill import Rating, rate
from .rating import RatingSystem


class TrueSkillRatingSystem(RatingSystem):
    
    def __init__(self):
        self.players = defaultdict(lambda: Rating())

    def calculate_ratings(self, match_outcomes):
        for match_outcome in match_outcomes:
            new_ratings = rate([[self.players[player_id]] for player_id in match_outcome.players], ranks=match_outcome.ranks)
            for player_id, new_rating in zip(match_outcome.players, new_ratings):
                self.players[player_id] = new_rating[0]
        return dict(self.players)