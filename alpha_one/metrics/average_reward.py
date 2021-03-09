from collections import defaultdict
from typing import List

from alpha_one.metrics import RatingSystem, MatchOutcome


class AverageRewardRatingSystem(RatingSystem):

    def __init__(self):
        self.players = defaultdict(lambda: {
            'total_reward': 0,
            'games_played': 0
        })

    def update_ratings(self, match_outcomes: List[MatchOutcome]):
        for match_outcome in match_outcomes:
            for player, reward in zip(match_outcome.players, match_outcome.rewards):
                self.players[player]['total_reward'] += reward
                self.players[player]['games_played'] += 1

    def get_rating(self, player_id):
        player_rating = self.players[player_id]
        if player_rating['games_played'] == 0:
            return player_rating['total_reward']
        else:
            return float(player_rating['total_reward']) / player_rating['games_played']

    def get_players(self):
        return list(self.players.keys())

    def add_player(self, player_id, initial_rating=0):
        self.players[player_id] = {
            'total_reward': initial_rating,
            'games_played': 0
        }
