import numpy as np
import torch
from environment.scenarios.get_to_landmark import GetToLandmark
from environment.scenarios.predator_prey_dense import PredatorPreyDense
from replay_buffer import ReplayBuffer
from agents import DDPGAgent
from datetime import datetime


num_iterations = 1000000
train_interval = 16
batch_size = 64

hidden_layer_size = 100
learning_rate = 1e-3
tau = 0.02
gamma = 0.97
exploration_decay = 0.998

replay_buffer_max_len = 100000

log_interval = 5000
gif_on_eval = True
gif_path = "gifs/"


# n_agents = 3
# env = GetToLandmark(n_agents)
# eval_env = GetToLandmark(n_agents)

n_predators = 1
n_preys = 1
n_agents = n_predators + n_preys
env = PredatorPreyDense(n_predators, n_preys, vel_scale_preys=10.0, vel_scale_predators=15.0)
eval_env = PredatorPreyDense(n_predators, n_preys, vel_scale_preys=10.0, vel_scale_predators=15.0)

critic_input_size = env.obs_size * n_agents + env.act_size * n_agents
agents = [DDPGAgent(id, env.obs_size, env.act_size, critic_input_size, hidden_layer_size, learning_rate, gamma, tau, exploration_decay) for id in range(n_agents)]

replay_buffer = ReplayBuffer(replay_buffer_max_len)


def main():
    obs = env.reset()

    for i in range(num_iterations):
        actions = [agent.act(obs[a]).detach().numpy() for a, agent in enumerate(agents)]
        next_obs, rewards, done = env.step(actions)
        replay_buffer.add_experience(obs, actions, rewards, next_obs, done)
        obs = next_obs

        if i % train_interval == 0:
            if len(replay_buffer) >= batch_size:
                obs_batch, acts_batch, rews_batch, next_obs_batch, dones_batch = replay_buffer.sample(batch_size)
                # Generate next actions for every agent
                next_acts_batch = np.ndarray(shape=acts_batch.shape, dtype="float32")
                for b in range(batch_size):
                    next_acts = np.array([agent.actor_target.forward(torch.from_numpy(o)).detach().numpy() for agent, o in zip(agents, next_obs_batch[b])])
                    # print(next_acts.reshape((-1, env.act_size)))
                    next_acts_batch[b] = next_acts.reshape((-1, env.act_size))
                for a, agent in enumerate(agents):
                    agent.train(obs_batch, acts_batch, rews_batch[:, a], next_obs_batch, next_acts_batch, dones_batch)

        if i % log_interval == 0:
            avg_r = avg_return(iteration=i)
            print('Episode {}: Average Return = {}'.format(i, avg_r))


def avg_return(n_episodes=10, iteration=None):
    rewards_sum = np.zeros((n_agents,))
    obs = eval_env.reset()
    for _ in range(n_episodes):
        done = False
        while not done:
            actions = [agent.act(obs[i], explore=False).detach().numpy() for i, agent in enumerate(agents)]
            obs, rewards, done = eval_env.step(actions)
            rewards_sum += rewards

    if gif_on_eval:
        imgs = []
        done = False
        while not done:
            imgs.append(eval_env.render())
            actions = [agent.act(obs[i], explore=False).detach().numpy() for i, agent in enumerate(agents)]
            obs, rewards, done = eval_env.step(actions)
        now = datetime.now()
        if iteration is not None:
            filename = '{}_iteration_{}.gif'.format(now.strftime('%Y%m%d_%H%M%S'), iteration)
        else:
            filename = '{}.gif'.format(now.strftime('%Y%m%d_%H%M%S'))
        imgs[0].save(gif_path + filename, save_all=True, append_images=imgs[1:])

    return rewards_sum / n_episodes


if __name__ == '__main__':
    # avg_return(n_episodes=0)
    main()
