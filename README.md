# Deep-Q-Learning-for-OpenAI-Gym

An implementation `cartpole_dqn.py` of reinforcement learning to navigate one of OpenAI's "Gym" environments, [CartPole](https://github.com/openai/gym/wiki/CartPole-v0). 

The agent uses Reinforcement Learning, implementing Keras to train a deep neural network as the Q-Value function approximator, so as to optimize performance in this environment (to keep the pole upright for the longest amount of time). The general nature of this approach means that this agent could easily be adapted to other games and environments, which I intend to do in the future. 

I have also uploaded a presentation on Q-Learning in general and as applied to this problem, and how the use of deep learning is appropriate in allowing the Q-Learning approach to work in problem domains where the action-state space renders the traditional method of Q-Tables unfeasible. 

Another, more basic Q-Learning implementation `q-learn-graph.py` is included to show the technique in a very simple form (using a Q-Table) on a very simple problem, details of which can be found in the presentation. 

As inspired by [Keon Kim](https://github.com/keon).
