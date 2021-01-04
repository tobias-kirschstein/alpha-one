class OpenSpielModelConfig(object):

    def __init__(self,
                 game,
                 model_type: str,
                 nn_width: int,
                 nn_depth: int,
                 weight_decay: float,
                 learning_rate: float):
        self.game = game
        self.model_type = model_type
        self.nn_width = nn_width
        self.nn_depth = nn_depth
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
