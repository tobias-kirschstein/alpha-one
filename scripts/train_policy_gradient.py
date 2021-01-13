import pyspiel
import numpy as np
from open_spiel.python import rl_environment

from alpha_one.model.model_manager import PolicyGradientModelManager, PolicyGradientConfig
from alpha_one.utils.logging import TensorboardLogger
from env import LOGS_DIR

game_name = "connect_four"
num_episodes = 1000000  # Number of train episodes for policy gradient
eval_interval = 1000  # After how many iterations to print the loss
n_evaluations = 50  # How often to evaluate the agents  to perform

loss_str = 'qpg'
hidden_layers_size = [50, 50, 50, 50, 50]
batch_size = 256
entropy_cost = 0.001
critic_learning_rate = 0.01
pi_learning_rate = 0.01
num_critic_before_pi = 4
optimizer = 'adam'


def _eval_agent(env, agents, num_episodes):
    """Evaluates `agent` for `num_episodes`."""
    rewards = 0.0
    rewards_by_player = np.array([0, 0])
    for _ in range(num_episodes):
        time_step = env.reset()
        episode_reward = 0
        episode_reward_by_player = np.array([0, 0])
        while not time_step.last():
            player_id = time_step.observations["current_player"]
            agent_output = agents[player_id].step(time_step, is_evaluation=True)
            time_step = env.step([agent_output.action])
            episode_reward += time_step.rewards[0]
            episode_reward_by_player[player_id] += time_step.rewards[player_id]
        rewards += episode_reward
        rewards_by_player += episode_reward_by_player
    return rewards / num_episodes, rewards_by_player / num_episodes


if __name__ == '__main__':
    # load the game
    game = pyspiel.load_game(game_name)

    # Create new checkpoint manager
    pg_checkpoint_manager = PolicyGradientModelManager(game_name).new_run()
    run_name = pg_checkpoint_manager.get_run_name()

    print("===========================")
    print(f"Starting run {run_name}")
    print("===========================")
    tensorboard = TensorboardLogger(f"{LOGS_DIR}/{game_name}/{run_name}")

    # RL environment configurations for policy gradient
    num_players = 2
    env_configs = {"players": num_players}
    env = rl_environment.Environment(game, **env_configs)
    info_state_size = env.observation_spec()["info_state"][0]
    num_actions = env.action_spec()["num_actions"]

    pg_configs = [
        PolicyGradientConfig(
            player_id=player_id,
            info_state_size=info_state_size,
            num_actions=num_actions,
            loss_str=loss_str,
            hidden_layers_sizes=hidden_layers_size,
            batch_size=batch_size,
            entropy_cost=entropy_cost,
            critic_learning_rate=critic_learning_rate,
            pi_learning_rate=pi_learning_rate,
            num_critic_before_pi=num_critic_before_pi,
            optimizer_str=optimizer)
        for player_id in [0, 1]
    ]

    pg_checkpoint_manager.store_config(pg_configs[0])

    agents = np.array([pg_checkpoint_manager.build_model(config) for config in pg_configs])

    for ep in range(num_episodes):
        time_step = env.reset()
        while not time_step.last():
            player_id = time_step.observations["current_player"]
            agent_output = agents[player_id].step(time_step)
            action_list = [agent_output.action]
            time_step = env.step(action_list)

        # Episode is over, step all agents with final info state.
        for agent in agents:
            agent.step(time_step)

        if ep and ep % eval_interval == 0:
            print(f"Evaluation at iteration {ep}")
            print("============================")
            losses = {
                "Agent 1": agents[0].loss,
                "Agent 2": agents[1].loss}
            print("Losses: ", losses)
            tensorboard.log_scalars("Loss", {key: value[0] for key, value in losses.items()}, ep)

            avg_return, avg_return_by_player = _eval_agent(env, agents, n_evaluations)
            print("Avg return: ", avg_return)
            print("Avg return by player: ", avg_return_by_player)
            tensorboard.log_scalar("Avg return", avg_return, ep)

            tensorboard.flush()
            print()

    # Store trained Policy Gradient
    pg_checkpoint_manager.store_checkpoint(agents[0], 0)
    pg_checkpoint_manager.store_checkpoint(agents[1], 1)
