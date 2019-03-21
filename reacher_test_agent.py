import os
import torch
import numpy as np
from agents.ppo_agent import Agent as ppo_agent
from agents.td3_agent import Agent as td3_agent
from agents.ddpg_agent import Agent as ddpg_agent
from agents.utils.RunConfig import RunConfig
from unityagents import UnityEnvironment

cwd = os.getcwd()
def file_exists(dir):
    return os.path.isfile(dir)

def get_env(seed):
    from sys import platform as _platform
    if _platform == "linux" or _platform == "linux2":
       # linux
        env = UnityEnvironment(file_name="./unity_envs/Reacher_Linux/Reacher.x86_64", seed=seed)
    elif _platform == "darwin":
       # MAC OS X
       env = UnityEnvironment(file_name="./unity_envs/Reacher.app", seed=seed)
    return env


def welcome(seed):
    env = get_env(seed)

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
    return env, state_size, action_size, num_agents



# ------ PPO -------- #
n_episodes = 10
rollout = 1000
seed = 10

ppo_config = RunConfig()
ppo_config.rollout = rollout
ppo_config.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_ppo_model(agent, dir, prefix=''):
    if not os.path.exists(dir + prefix):
        raise Exception('{} : does not exist'.format(dir + prefix))
    agent.net.load_state_dict(torch.load(dir + prefix + 'net.pth', map_location=ppo_config.device))


def run_ppo(env):
    log_dir = cwd + '/models/{}/'.format(algo) + 'reacher/'
    agent = ppo_agent(state_size=state_size, action_size=action_size,
                      random_seed=seed, writer=None, config=ppo_config,
                      n_agents=1)

    load_ppo_model(agent, log_dir)

    brain_name = env.brain_names[0]
    env_info = env.reset(train_mode=False)[brain_name]  # reset the environment
    states = env_info.vector_observations  # get the current state (for each train_agent)
    num_agents = len(env_info.agents)
    G = np.zeros(num_agents)  # Undiscounted return for each train_agent

    t = 0
    while True:
        actions = agent.test_act(states)                        # select an action from train_agent
        env_info = env.step(actions)[brain_name]                # send all actions to tne environment
        next_states = env_info.vector_observations              # get next state (for each train_agent)
        rewards = np.array(env_info.rewards)                    # get reward (for each train_agent)
        dones = np.array(env_info.local_done)                   # see if episode finished
        G += rewards                                            # update the score (for each train_agent)
        if np.any(dones):                                       # exit loop if episode finished
            break
        else:
            states = next_states                                # roll over states to next time step
            t += 1
    return G


# ----- TD3 ------ #
td3_config = RunConfig()
td3_config.rollout = rollout
td3_config.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_td3_model(agent, dir, prefix=''):
    if not os.path.exists(dir + prefix):
        raise Exception('{} : does not exist'.format(dir + prefix))
    agent.actor_local.load_state_dict(torch.load(dir + prefix + 'actor.pth', map_location=td3_config.device))
    agent.critics_local[0].load_state_dict(torch.load(dir + prefix + 'critic_0.pth', map_location=td3_config.device))
    agent.critics_local[1].load_state_dict(torch.load(dir + prefix + 'critic_1.pth', map_location=td3_config.device))

def run_td3(env):
    log_dir = cwd + '/models/{}/'.format(algo) + 'reacher/'
    agent = td3_agent(state_size=state_size,
                      action_size=action_size,
                      random_seed=seed,
                      n_agents=n_agents,
                      writer=None,
                      config=td3_config)
    load_td3_model(agent, log_dir)
    brain_name = env.brain_names[0]
    env_info = env.reset(train_mode=False)[brain_name]  # reset the environment
    s = env_info.vector_observations                    # get the current state (for each train_agent)
    num_agents = len(env_info.agents)
    G = np.zeros(num_agents)                            # undiscounted return for each train_agent

    t = 0
    while True:
        a = agent.act(s, add_noise = False)               # select an action from train_agent
        env_info = env.step(a)[brain_name]                # send all actions to tne environment
        sp = env_info.vector_observations                 # get next state (for each train_agent)
        rewards = np.array(env_info.rewards)              # get reward (for each train_agent)
        dones = np.array(env_info.local_done)             # see if episode finished
        G += rewards                                      # update the score (for each train_agent)
        if np.any(dones):                                 # exit loop if episode finished
            break
        else:
            s = sp                                # roll over states to next time step
            t += 1
    return G



# --------- DDPG ------------- #
ddpg_config = RunConfig()
ddpg_config.rollout = rollout
ddpg_config.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_ddpg_model(agent, dir, prefix=''):
    if not os.path.exists(dir + prefix):
        raise Exception('{} : does not exist'.format(dir + prefix))
    agent.actor_local.load_state_dict(torch.load(dir + prefix + 'actor.pth', map_location=ddpg_config.device))
    agent.critic_local.load_state_dict(torch.load(dir + prefix + 'critic.pth', map_location=ddpg_config.device))

def run_ddpg(env):
    log_dir = cwd + '/models/{}/'.format(algo) + 'reacher/'
    agent = ddpg_agent(state_size=state_size,
                        action_size=action_size,
                        random_seed =seed,
                        n_agents=n_agents,
                        writer=None,
                        config=ddpg_config)

    load_ddpg_model(agent, log_dir)
    brain_name = env.brain_names[0]
    env_info = env.reset(train_mode=False)[brain_name]  # reset the environment
    s = env_info.vector_observations                    # get the current state (for each train_agent)
    num_agents = len(env_info.agents)
    G = np.zeros(num_agents)                            # undiscounted return for each train_agent

    t = 0
    while True:
        a = agent.act(s, add_noise = False)               # select an action from train_agent
        env_info = env.step(a)[brain_name]                # send all actions to tne environment
        sp = env_info.vector_observations                 # get next state (for each train_agent)
        rewards = np.array(env_info.rewards)              # get reward (for each train_agent)
        dones = np.array(env_info.local_done)             # see if episode finished
        G += rewards                                      # update the score (for each train_agent)
        if np.any(dones):                                 # exit loop if episode finished
            break
        else:
            s = sp                                # roll over states to next time step
            t += 1
    return G



# --------- DDPG_PSNE ------------- #
ddpg_psne_config = RunConfig()
ddpg_psne_config.rollout = rollout
ddpg_psne_config.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_ddpg_psne_model(agent, dir, prefix=''):
    if not os.path.exists(dir + prefix):
        raise Exception('{} : does not exist'.format(dir + prefix))
    agent.actor_local.load_state_dict(torch.load(dir + prefix + 'actor.pth', map_location=ddpg_psne_config.device))
    agent.critic_local.load_state_dict(torch.load(dir + prefix + 'critic.pth', map_location=ddpg_psne_config.device))


def run_ddpg_psne(env):
    log_dir = cwd + '/models/{}/'.format(algo) + 'reacher/'
    agent = ddpg_agent(state_size=state_size,
                       action_size=action_size,
                       random_seed=seed,
                       n_agents=n_agents,
                       writer=None,
                       config=ddpg_psne_config)

    load_ddpg_psne_model(agent, log_dir)
    brain_name = env.brain_names[0]
    env_info = env.reset(train_mode=False)[brain_name]  # reset the environment
    s = env_info.vector_observations  # get the current state (for each train_agent)
    num_agents = len(env_info.agents)
    G = np.zeros(num_agents)  # undiscounted return for each train_agent

    t = 0
    while True:
        a = agent.act(s, add_noise=False)  # select an action from train_agent
        env_info = env.step(a)[brain_name]  # send all actions to tne environment
        sp = env_info.vector_observations  # get next state (for each train_agent)
        rewards = np.array(env_info.rewards)  # get reward (for each train_agent)
        dones = np.array(env_info.local_done)  # see if episode finished
        G += rewards  # update the score (for each train_agent)
        if np.any(dones):  # exit loop if episode finished
            break
        else:
            s = sp  # roll over states to next time step
            t += 1
    return G



# --------- Main ------------ #
import sys, getopt

if __name__== "__main__":
    algo = None
    use_str = 'reacher_test_agent.py -a <string: [ppo|ddpg|ddpg_psne|td3]>'
    try:
        opts, args = getopt.getopt(sys.argv[1:], shortopts="a:")
    except getopt.GetoptError as ex:
        print(str('{}: ' + use_str).format(str(ex)))
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-a",):
            algo = str(arg)
            assert algo in ['ppo', 'td3', 'ddpg', 'ddpg_psne'], '-a: algorithm <{}> not recognized'.format(algo)


    env, state_size, action_size, n_agents = welcome(seed)
    if algo == 'ppo':
        run_ppo(env)
    elif algo == 'ddpg':
        run_ddpg(env)
    elif algo == 'ddpg_psne':
        run_ddpg_psne(env)
    elif algo == 'td3':
        run_td3(env)
