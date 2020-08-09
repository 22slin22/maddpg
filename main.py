import numpy as np
from environment.scenarios.get_to_landmark import GetToLandmark
from replay_buffer import ReplayBuffer
from agents import DDPGAgent
from datetime import datetime


num_iterations = 1000000
train_interval = 16
batch_size = 64

learning_rate = 1e-3
tau = 1e-2
gamma = 0.97
exploration_decay = 0.998

replay_buffer_max_len = 100000

log_interval = 5000
gif_on_eval = True
gif_path = "gifs/"


n_agents = 3
env = GetToLandmark(n_agents)
eval_env = GetToLandmark(n_agents)
agents = [DDPGAgent(env.obs_size, env.act_size, 4, 32, learning_rate, gamma, tau, exploration_decay)] * env.n_agents

replay_buffer = ReplayBuffer(replay_buffer_max_len)


def avg_return(n_episodes=10, iteration=None):
    rewards_sum = np.zeros((n_agents,))
    obs = eval_env.reset()
    for _ in range(n_episodes):
        done = False
        while not done:
            actions = [agent.act(obs[i], explore=False).detach().numpy() for i, agent in enumerate(agents)]
            obs, rewards, done = eval_env.step(actions)
            # print(rewards)
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


def main():
    obs = env.reset()

    for i in range(num_iterations):
        actions = [agent.act(obs[i]).detach().numpy() for i, agent in enumerate(agents)]
        next_obs, rewards, done = env.step(actions)
        replay_buffer.add_experience(obs, actions, rewards, next_obs, done)
        obs = next_obs

        if i % train_interval == 0:
            if len(replay_buffer) >= batch_size:
                obs_batch, acts_batch, rews_batch, next_obs_batch, dones_batch = replay_buffer.sample(batch_size)
                for i, agent in enumerate(agents):
                    agent.train(obs_batch[:, i], acts_batch[:, i], rews_batch[:, i], next_obs_batch[:, i], dones_batch)

        if i % log_interval == 0:
            avg_r = avg_return(iteration=i)
            print('Episode {}: Average Return = {}'.format(i, avg_r))


if __name__ == '__main__':
    main()
