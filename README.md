# ML-Based-Inverted-Pendulum-Control
# Deep-Q-Learning-in-OpenAI-Gym

An implementation `cartpole_dqn.py` of a reinforcement learning technique for one of OpenAI's "Gym" environments, [CartPole](https://github.com/openai/gym/wiki/CartPole-v0). This is a simulation of the [inverted pendulum problem](https://en.wikipedia.org/wiki/Inverted_pendulum), where a pendulum is mounted by its pivot point to a cart, which must be moved horizontally to keep the pole upright. Reward is given for every frame the pole is upright and the cart has remained under a maximum distance from the starting point. 

The agent uses a modified [Q-Learning](https://en.wikipedia.org/wiki/Q-learning) strategy called Deep Reinforcement Learning as presented by Minh et al. in [Playing Atari with Deep Reinforcement Learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf). Supplemental methods of this technique include using a deep neural network as a Q-Value function approximator in the place of a Q-Table, and an exerience replay module where the agent records experiences at different time-steps into a data-set, sampled at random to perform minibatch updates on the network.

The agent recieves information only in the form of a vector of length three containing values representing state, reward, and whether the game has terminated or not. The state contains values for the position and speed of the cart, and the angle and angular velocity of the rod's pivot to the horizontal. The agent has no strategy or knowledge of the environment and must "learn" something about the relationship of these values to play the game. The general nature of this approach means that this agent could easily be adapted to other games and environments, which I intend to do in the future. 

I have also uploaded a presentation I created on Q-Learning in general and as it applies to this problem, and how the use of deep learning is appropriate in allowing the Q-Learning approach to work in problem domains, such as this one, where the size of the action-state space renders the traditional method of Q-Tables unfeasible. 

Another, more basic Q-Learning implementation `q-learn-graph.py` is included to show the other technique (using a Q-Table) on a very simple problem, details of which can be found in the presentation. 

As inspired by the original [DQN Paper](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf) and [Keon Kim](https://github.com/keon).

