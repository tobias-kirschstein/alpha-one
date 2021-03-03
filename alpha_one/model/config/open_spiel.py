from .base import ModelConfig


class OpenSpielModelConfig(ModelConfig):

    def __init__(self,
                 game,
                 model_type: str,
                 input_shape: list,
                 nn_width: int,
                 nn_depth: int,
                 weight_decay: float,
                 learning_rate: float,
                 output_shape: int = None,
                 omniscient_observer: bool = False):
        super(OpenSpielModelConfig, self).__init__(
            game=game,
            model_type=model_type,
            input_shape=input_shape,
            nn_width=nn_width,
            nn_depth=nn_depth,
            weight_decay=weight_decay,
            learning_rate=learning_rate,
            output_shape=output_shape,
            omniscient_observer=omniscient_observer)
