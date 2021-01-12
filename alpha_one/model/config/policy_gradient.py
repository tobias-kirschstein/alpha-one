from alpha_one.model.config.base import ModelConfig


class PolicyGradientConfig(ModelConfig):

    def __init__(self,
                 player_id,
                 info_state_size,
                 num_actions,
                 loss_str="a2c",
                 loss_class=None,
                 hidden_layers_sizes=(128,),
                 batch_size=16,
                 critic_learning_rate=0.01,
                 pi_learning_rate=0.001,
                 entropy_cost=0.01,
                 num_critic_before_pi=8,
                 additional_discount_factor=1.0,
                 max_global_gradient_norm=None,
                 optimizer_str="sgd"
                 ):
        super(PolicyGradientConfig, self).__init__(
            player_id=player_id,
            info_state_size=info_state_size,
            num_actions=num_actions,
            loss_str=loss_str,
            loss_class=loss_class,
            hidden_layers_sizes=hidden_layers_sizes,
            batch_size=batch_size,
            critic_learning_rate=critic_learning_rate,
            pi_learning_rate=pi_learning_rate,
            entropy_cost=entropy_cost,
            num_critic_before_pi=num_critic_before_pi,
            additional_discount_factor=additional_discount_factor,
            max_global_gradient_norm=max_global_gradient_norm,
            optimizer_str=optimizer_str
        )
