import torch
import numpy as np
from unityagents import UnityEnvironment
from udacity.ddpg_psne import Agent
import os
import time

cwd = os.getcwd()

def get_env():
    from sys import platform as _platform
    if _platform == "linux" or _platform == "linux2":
       # linux
       env = UnityEnvironment(file_name="./Reacher_Linux/Reacher.x86_64")
    elif _platform == "darwin":
       # MAC OS X
       env = UnityEnvironment(file_name="Reacher.app")
    else:
        raise Exception('Unsupported OS')

    return env

def welcome():
    env = get_env()
    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode=False)[brain_name]

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

def file_exists(dir):
    return os.path.isfile(dir)

def load_agent(agent, path):
    agent.actor_target.load_state_dict(torch.load(path, map_location='cpu'))


def run(env, agent):
    brain_name = env.brain_names[0]
    agent.reset()
    done = False
    env_info = env.reset(train_mode=False)[brain_name]
    state = env_info.vector_observations

    while not done:
        action = agent.test_act(state)
        env_info = env.step(action)[brain_name]
        state = env_info.vector_observations
        reward = np.array(env_info.rewards)
        done = np.array(env_info.local_done)
        print('\r Reward: {:.1f}'.format(float(reward.squeeze())), end="")
        if bool(done):
            break


import sys, getopt
if __name__== "__main__":
    # Parse arguments and options
    netdir = None
    try:
        opts, args = getopt.getopt(sys.argv[1:], shortopts="p:")
    except getopt.GetoptError as ex:
        print('{}: test_ddpg_psne.py -p <string: path of actor net>'.format(str(ex)))
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-p",):
            netdir = str(arg)
            netdir = cwd + netdir
            assert file_exists(netdir), 'Cannot find: <{}>'.format(netdir)

    if netdir is None:
        print('test_ddpg_psne.py -p <string: path of actor net>')
        sys.exit(2)


    env, state_size, action_size = welcome()
    agent = Agent(state_size=state_size, action_size=action_size, random_seed=0)
    load_agent(agent, netdir)
    run(env, agent)


