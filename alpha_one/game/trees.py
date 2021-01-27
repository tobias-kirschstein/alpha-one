import pyspiel
from open_spiel.python.observation import make_observation

class player_observation_tree:
    
    def __init__(self, game, player_id):
        self.player = player_id
        
        self.observation = make_observation(
        game,
        pyspiel.IIGObservationType(
                                perfect_recall=False,
                                public_info=False,
                                private_info=pyspiel.PrivateInfoType.SINGLE_PLAYER))
        
        
        
    def get_private_observations(self, state):
        self.observation.set_from(state, player=self.player)
        return self.observation.tensor
    
    
class game_tree:
    
    def __init__(self, game):
        self.observation = make_observation(
        game,
        pyspiel.IIGObservationType(
                                perfect_recall=False,
                                public_info=True,
                                private_info=pyspiel.PrivateInfoType.ALL_PLAYERS))
        
        
    def get_all_childs(self, state):
        
        if (state.current_player() == 0):
            player = 0
        else:
            player = 1
        
        actions_child_states = {}
        for action in state.legal_actions():
            child_state = state.child(action)
            self.observation.set_from(child_state, player=player)
            actions_child_states[action] = self.observation.tensor
            
        return actions_child_states
    
    
class public_observation_tree:
    
    def __init__(self, game):
        
        self.observation = make_observation(
        game,
        pyspiel.IIGObservationType(
                                perfect_recall=False,
                                public_info=True,
                                private_info=pyspiel.PrivateInfoType.NONE))
        
        
        
    def get_public_observations(self,state):
        
        if (state.current_player() == 0):
            self.observation.set_from(state, player=0)
            return self.observation.tensor
        
        self.observation.set_from(state, player=1)
        return self.observation.tensor