[karpathy_rl_blog]: http://karpathy.github.io/2016/05/31/rl/
[lilian_weng_policy_gradient]:https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html#policy-gradient


[DDPG_algo]: assets/DDPG_algo.png

[DPG]:  http://proceedings.mlr.press/v32/silver14.pdf
[DDPG]: https://arxiv.org/pdf/1509.02971.pdf
[PSNE]:  https://arxiv.org/pdf/1706.01905.pdf




# Deep Deterministic Policy Gradients [DDPG]  for Continuous Control

For [DDPG], we use an actor-critic model and build on the insights gained from [DQN], which was able to achieve human level performance on Atari games, to tackle continuous control problems. While [DQN] handles complex high dimensional systems, it has a significant limitation as it operates on discrete action space. One, of course, can approach a continuous control problem by discretizing the action (i.e. control) space at the cost of the loss of precision (quantization errors), and more importantly, for multiple control dimensions problems, face an explosion in the number of discrete outputs, making learning very difficult or intractable.

### Background

This subsection draws heavily from [this excellent article][lilian_weng_policy_gradient]

The goal in reinforcement learning is to learn a policy which maximizes the expected return from the _start_ distribution:

`J(θ) = E[G0|s0, a~π(s|θ) r,s~env] =`

`∑{s,s0}∈𝒮(d(s|s0,θ)V(s0|θ)) = `

`∑{s,s0}∈𝒮(d(s|s0, θ)∑{a}∈𝒜(π(a|s, θ)Q(s,a|θ))`


where `V(s|θ)` denotes the value function if the parametrized policy `π(·|·, θ)` is followed. `d(s|s0, θ)` is the stationary state distribution of the underlying Markov process defined as `d(s|s0,θ) = lim(t->∞)P(S = st|S = s0, θ)`.

The stationary distribution denotes the probability distribution after the model takes an infinite number of states and follows the policy. So, it's values depend on the policy (known via θ), and on the environment (unknown).

Since we're trying to _maximize_ our objective function, we follow the direction of the gradient `∇J(θ)`, to perform _gradient ascent_ in order to find its maximum value.

The formulation of the objective function as the sum and product of known and unknown functions whose input is θ, we need to know their derivatives w.r.t to θ. in order to compute ∇J(θ). While that is known for a policy, we cannot take the derivative the the state distribution. Alternatively, how  can we know the direction of change a change in policy causes on the distribution of the unknown environment ?

##### Policy Gradient Theorem
According to the _policy gradient theorem_, we have:


`∇J(θ) = ∇(∑{s,s0}∈𝒮(d(s|s0,θ)∑{a}∈𝒜(π(a|s,θ)Q(s,a|θ)))`

thus:

`∇J(θ)∝ ∑{s,s0}∈𝒮(d(s|s0, θ)∑{a}∈𝒜(∇π(a|s,θ)Q(s,a|θ))`

where the `∇` are always with respect to `θ`. As we can see, the gradient of the objective function does not involve the gradient of the stationary state distribution, `d`, as mentioned earlier, we couldn't compute the effects a change in `θ` would have (i.e. we don't know how to compare two policies and their affect the state distributions). Therefore, all that is required for the computation of the gradient of the objective function, is that for each sample obtained from the distributions - `d` and `π` we obtain the gradient of the policy. The theorem guarantees that the expectation of the gradients of the samples will be proportional to the gradient of the objective function itself.


##### Deterministic Policy Gradient (DPG)
Let the probability of _k_-step state visitations, starting from `s0` to `s`, following a policy `𝜋` be denoted as:

`𝜌(s0 -> s, k)`

where:

`𝜌(s -> s, 0) = 1`

`𝜌(s0 -> s', 1) = 𝛴{a}p(s'|s,a)𝜋(a|s)`

`𝜌(s0 -> s', k+1) = 𝛴{s*}𝜌(s0 -> s*, k)𝛴{a}p(s'|s*,a)𝜋(a|s*) = 𝛴{s*}𝜌(s0 -> s*, k)𝜌(s* -> s', 1)`

So the state visitation probability denotes the probability of all possible routes from `s0` to `s` in the MDP graph.

With this interpretation, the stationary distribution `d(s|s0)` above can also be written as:

`d(s | s0) = 𝛴{k: 0 -> ∞}𝜌(s0 -> s, k) / 𝛴{{s0, s*}∈𝒮, k: 0 -> ∞}𝜌(s0 -> s, k)`


So far, the policy function `π(.|s)` is always modeled as a probability distribution over actions given the current state and thus it is _stochastic_. Deterministic policy gradient (DPG) instead models the policy as a deterministic decision: `a=μ(s)`

Let:

`𝜌0(s)`: initial distribution over states
`𝜌(s)`: discounted state visitation distribution under the _deterministic_ policy `μ`:

The objective function under this framework is defined as:

`J(θ)=∫ρ(s)Q(s,μ(s;θ))ds`

where the stochastic sources are the state visitations, not the actions. The gradient of this objective function is:

`∇J(θ) = E[∇Q(s,μ(s;θ)∇μ(s;θ) | ρ(s)]`, again, all the gradients are with respect to the parameters of the policy.

The deterministic policy is a special case of the stochastic policy. Stochastic policy `μ` can be re-parametrized by a deterministic policy and a variational term `𝜎`. Compared to the deterministic policy, we expect the stochastic policy to require more samples as it integrates the data over the whole state and action space

#### DDPG

As mentioned above, DDPG combines DPG and DQN.  The original DQN works in discrete space, and DDPG extends it to continuous space with the actor-critic framework while learning a deterministic policy.


##### Actor - Critic

<div style="text-align: center"><img src="assets/actor-critic.png" alt="Actor-Critic schema" width="300" height="250" align="middle"></div>

Under the actor(`μ` - deterministic/`π` - stochastic) + critic(`Q`/`V`) framework, the critic evaluates the expectations of returns for a given state. Its knowledge is used for optimizing the next action (i.e. the actor). The actor, in turn explores and assists in improving the critics values (a bootstrap approach).

##### Deep Q Network (DQN)
[DQN] is able to learn value functions using function approximators in a stable and robust way due to two innovations:

1. a _behavioral_ network is trained off-policy with samples from a replay buffer to minimize correlations between samples
2. this network is trained with a _target_ Q network to give consistent targets during temporal difference backups. In return, the _target_ network is slowly updated with the weights of the _behavioral_ network


#### Algorithm
[DDPG] takes a similar approach to DQN, in that it uses a behavioral and a target network type for each of the actor and critic. The behavioral actor uses the gradient of the behavioral critic to compute the objective function's gradient.
The behavioral critic minimizes the TD(0) error between its estimations and target estimations derived from the target critic. However, unlike in DQN, in DDPG the target networks are very slowly updated every episodic iteration (via the `𝜏` hyper-parameter)

<div style="text-align: center"><img src="assets/DDPG_algo.png" alt="DDPG algorithm" width="640" height="480" ></div>

where the `'` denote the _targets_.

But since we're using a deterministic policy, exploration is achieved by sampling from a noise process. Following the authors, the Ornstein-Uhlenbeck process was used, for the vanilla implementation of DDPG


#### Parameter Noise Space for Exploration
An alternative approach to finding optimal solutions to RL problems are evolutionary methods. In these methods, the search is performed in parameter space. Analogously, in [PSNE] the authors perturb the parameters of the Actor at the beginning of the episode.

Adding action space noise, while in a fixed state, by definition, we obtain different actions when encountering the same space during the rollout. By perturbing the actor parameters once per episode, we obtain repeatable actions for each state encountered (during the episode). This ensures consistency in actions, and directly introduces a dependence
between the state and the exploratory action taken.

However, perturbing deep NN is not straightforward, as the effects of noise would vary by layer. By using _layer normalization_, the same noise scale can be used across layers, even if different layers varying sensitivities to noise.

##### Adaptive noise scaling
In order to pick a consistent noise scale `σ` across different topologies, and over the training process (as the network becomes more sensitive as learning progresses), the following formulation is used:

<div style="text-align: center"><img src="assets/psne_noise_formula.png" alt="Adaptive Noise Scale" width="220" height="60" ></div>

where `d()` denotes the distance in the actions produced between a policy and it's perturbed version:

<div style="text-align: center"><img src="assets/psne_distance_fun.png" alt="PSNE Distance Function" width="340
0" height="100" ></div>

The distance metric is estimated on a batch sampled on the replay memory.

In this approach, the exploratory actions are taken by the _perturbed_ actor, while the rest of the learning algorithm remains the same.

#### Number of agents
These aproaches were tried with single and multiple agent environments. A radical speedup, in terms of epoch count, was noticed in solving the environment.


<div style="text-align: center"><img src="assets/DDPG_PSNE_SingleAgent.png" alt="Signle Agent: DDPG + PSNE" width="300" height="200" ></div>

<div style="text-align: center"><img src="assets/DDPG_PSNE_MultiAgent.png" alt="Signle Agent: DDPG + PSNE" width="300" height="200" ></div>

Using DDPG + PSNE, the single agent reacher environment was solved (score +30 over 100 consecutive episodes) in 493 episodes, while the multi agent environment was solved in 131 episodes.
