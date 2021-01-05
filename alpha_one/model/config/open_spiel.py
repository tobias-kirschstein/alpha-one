from .base import ModelConfig


class OpenSpielModelConfig(ModelConfig):

    def __init__(self,
                 game,
                 model_type: str,
                 nn_width: int,
                 nn_depth: int,
                 weight_decay: float,
                 learning_rate: float):
        super(OpenSpielModelConfig, self).__init__(
            game=game,
            model_type=model_type,
            nn_width=nn_width,
            nn_depth=nn_depth,
            weight_decay=weight_decay,
            learning_rate=learning_rate)
