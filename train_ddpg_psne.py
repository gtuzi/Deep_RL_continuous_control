import torch
import numpy as np
from collections import deque

from unityagents import UnityEnvironment
from udacity.ddpg_psne import Agent
from tensorboardX import SummaryWriter
import os

cwd = os.getcwd()

def get_env(agents = 'multi'):
    from sys import platform as _platform

    if _platform == "linux" or _platform == "linux2":
       # linux
       if agents == 'single':
            env = UnityEnvironment(file_name="./Reacher_Linux/Reacher.x86_64", no_graphics = False)
       elif agents == 'multi':
           env = UnityEnvironment(file_name="./Reacher_Linux_Multi/Reacher.x86_64", no_graphics=False)
       else:
           raise Exception('Unrecognized environment type')
    elif _platform == "darwin":
       # MAC OS X
       if agents == 'single':
           env = UnityEnvironment(file_name="Reacher.app", no_graphics = False)
       elif agents == 'multi':
           env = UnityEnvironment(file_name="Reacher_Multi.app", no_graphics=False)
       else:
           raise Exception('Unrecognized environment type')
    else:
        raise Exception('Unsupported OS')

    return env

def welcome(agents):
    env = get_env(agents)

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # number of agents
    num_agents = len(env_info.agents)

    # size of each action
    action_size = brain.vector_action_space_size

    # examine the state space
    states = env_info.vector_observations
    state_size = states.shape[1]
    print('Number of agents:', num_agents)
    print('Size of each action:', action_size)
    print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))

    return env, state_size, action_size


def save_nets(agent, subdir):
    if not os.path.isdir(cwd + '/models/{}'.format(subdir)):
        os.makedirs(cwd + '/models/{}'.format(subdir))
    torch.save(agent.actor_local.state_dict(), 'local_actor.pth'.format(subdir))
    torch.save(agent.critic_local.state_dict(), 'local_critic.pth'.format(subdir))
    torch.save(agent.actor_target.state_dict(), 'target_actor.pth'.format(subdir))
    torch.save(agent.critic_target.state_dict(), 'target_critic.pth'.format(subdir))



def ddpg_single_agent(env, agent, n_episodes=2000, max_t=int(10000), subdir=''):
    scores_deque = deque(maxlen=100)
    episode_horizons = deque(maxlen=100)
    scores = []
    max_score = -np.Inf
    brain_name = env.brain_names[0]

    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations
        agent.reset()
        score = 0
        for t in range(max_t):
            action = agent.act(state)
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations
            reward = np.array(env_info.rewards)
            done = np.array(env_info.local_done)
            if t + 1 == max_t:
                done = np.ones_like(done, dtype=np.bool)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                episode_horizons.append(t)
                break
        scores_deque.append(score)
        scores.append(score)
        print(
            '\rEpisode {}\tAverage Score: {:.2f}\tScore: {:.2f}\tTime Step: {}'.format(i_episode, np.mean(scores_deque),
                                                                                       float(score), agent.total_steps),
            end="")

        if i_episode % 50 == 0:
            print('\tAvg. Horizon: {:.2f}'.format(np.mean(episode_horizons)))

        if np.mean(scores_deque) >= 30.:
            print('The environment was solved in {} episodes'.format(i_episode))
            break

    save_nets(agent, subdir)
    return scores


def ddpg_multi_agent(env, agent, n_episodes=2000, max_t=int(10000), subdir=''):
    scores_deque = deque(maxlen=100)
    scores = []
    episode_horizons = deque(maxlen=100)
    max_score = -np.Inf
    brain_name = env.brain_names[0]
    solved = False
    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations
        n_agents = state.shape[0]
        agent.reset()
        score = np.zeros((n_agents, 1), dtype=np.float32)
        for t in range(max_t):
            action = agent.act(state)
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations
            reward = np.array(env_info.rewards)[..., None]
            done = np.array(env_info.local_done)[..., None]
            if t + 1 == max_t:
                done = np.ones_like(done, dtype=np.bool)

            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if np.all(done):
                episode_horizons.append(t)
                break

        scores_deque.append(score)
        scores.append(score)

        _mu_score_moving = np.mean(np.mean(scores_deque, axis=1))
        print('\rEpisode {}\t100-episode avg score: {:.2f}\tScore: {:.2f}\tTime Step: {}'.format(i_episode,
                                                                                                 _mu_score_moving,
                                                                                                 float(np.mean(score)),
                                                                                                 agent.total_steps),
              end="")

        if i_episode % 50 == 0:
            print(
                '\rEpisode {}\t100-episode avg score: {:.2f}\tAvg. Horizon: {:.2f}'.format(i_episode, _mu_score_moving,
                                                                                           np.mean(episode_horizons)))

        if (np.mean(scores_deque) >= 30.) and (i_episode > 99) and (not solved):
            print('\nThe environment was solved in {} episodes'.format(i_episode))
            solved = True

    save_nets(agent, subdir)

    return scores


import sys, getopt
if __name__== "__main__":
    # Defaults
    agents = 'multi'
    max_t = 1000
    n_episodes = 2000
    prefix = 'ddpg_psne_'
    _prefix = None
    # Parse arguments and options
    try:
        opts, args = getopt.getopt(sys.argv[1:], shortopts="a:e:f:")
    except getopt.GetoptError as ex:
        print('train_ddpg_psne.py -a <string: single / multi> -e <int: number of episodes> -f <string: subdir in <~/models> for saved nets, <~/logs> for logs')
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-a",):
            agents = str(arg)
            assert (agents == 'single') or (agents == 'multi'), 'Agents must be either <single> or <multi>'
        if opt in ("-e",):
            n_episodes = int(str(arg))
        if opt in ("-f",):
            _prefix = str(arg)

    prefix = _prefix if _prefix is not None else prefix + agents

    logpath = cwd + '/logs/{}'.format(prefix)
    if not os.path.isdir(logpath):
        os.makedirs(logpath)

    writer = SummaryWriter(log_dir=logpath)
    env, state_size, action_size = welcome(agents)
    agent = Agent(state_size=state_size, action_size=action_size, random_seed=0, writer=writer)

    if agents == 'multi':
        ddpg_multi_agent(env, agent, n_episodes=n_episodes, max_t=1000, subdir=prefix)
    elif agents == 'single':
        ddpg_single_agent(env, agent, n_episodes=n_episodes, max_t=1000, subdir=prefix)



