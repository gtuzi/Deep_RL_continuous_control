[actor-critic]: assets/actor-critic.png

[image1]: assets/reachers.gif
[image2]: assets/crawler.png
[discounted_state_visitation]: assets/discounted_state_visitation.png
[Reacher]: https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher 
[Crawler]: https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#crawler

[PPO]:   https://arxiv.org/pdf/1707.06347.pdf
[A3C]:   https://arxiv.org/pdf/1602.01783.pdf
[D4PG]: https://openreview.net/pdf?id=SyZipzbCb
[DQN]:  https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf

[DPG]:  http://proceedings.mlr.press/v32/silver14.pdf
[DDPG]: https://arxiv.org/pdf/1509.02971.pdf
[PSNE]:  https://arxiv.org/pdf/1706.01905.pdf
[TD3]: https://arxiv.org/pdf/1802.09477.pdf

[karpathy_rl_blog]: http://karpathy.github.io/2016/05/31/rl/
[lilian_weng_policy_gradient]:https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html#policy-gradient

# Continuous Control
### Introduction

For this project, I explore the application of [DDPG], [DDPG] with parameter-space noise variation [PSNE], [TD3], and [PPO].
 
These application are applied to [Reacher] and [Crawler] environments. Two versions of [Reacher] are tried: one with a single agent, the other with multiple agents. 

Algorithms such as [A3C] and [D4PG] (based on [DDPG]) take a distributed approach to environments with multiple agents. However, in this work I focus on the combinations of [DDPG] and variations [PSNE], with small adaptations for multi-agent environments.


### Environment Description

#### Reacher

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of the agent is to maintain its position at the target location for as many time steps as possible.

<div style="text-align: center"><img src="assets/reachers_frozen.png" alt="Reacher" width="400" height="200" ></div>

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector is a number between -1 and 1.

#### Training variants

For this project, I explore two separate versions of the Unity environment for [Reacher]:
- The first version contains a *single* agent.
- The second version is a *multi*-agent where there are several identical agents each with its own copy of the environment: 
    - *Reacher* contains 20 agents
    - *Crawler* contains 13 agents

Only the multi-agent [Crawler] enviroment is used here. 

### Solving the Environments
The tasks in this project are episodic, that is, the agent/s run for a finite number of steps on the environment.
For purposes of Udacity's Deep RL class the following variations determine when an environment is solved: 

#### Option 1: Solve the single agent version
The task is episodic, and in order to solve the environment,  your agent must get an average score of +30 over 100 consecutive episodes.

#### Option 2: Solve the Second Version

The barrier for solving the second version of the environment is slightly different, to take into account the presence of many agents.  In particular, your agents must get an average score of +30 (over 100 consecutive episodes, and over all agents).  Specifically,
- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent.  This yields 20 (potentially different) scores.  We then take the average of these 20 scores. 
- This yields an **average score** for each episode (where the average is over all 20 agents).

The environment is considered solved, when the average (over 100 episodes) of those average scores is at least +30. 


## Deep Deterministic Policy Gradients [DDPG] variants for Continuous Control

[DDPG], [DDPG] + [PSNE] are implemented. The explanation for these algorithms is found
[here](ddpg.md)


## [TD3] for Continuous Control
[TD3] is implemented. Explanation is [here](td3.md)

#### Dependencies
* python: 3.5
* tensorboardX: 1.4
* tensorboard: 1.7.0
* pytorch: 0.4.1
* numpy: 1.15.2
* Linux or OSX environment


##### Test the algorithms the actor
Run `test_ddpg_psne.py` with the following option:
* -a : algorithm of choice: ['TD3'|PPO|DDPG|DDPG_PSNE]


###### Pre-trained Models
The pre-trained models are located under
`/models/<algorithm>`

###### Video
Coming so so soon ... 










