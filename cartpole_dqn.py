import gym
import numpy as np
import random
import os
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

output_dir = 'model_output/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

    
time = 5000
alpha = .5
gamma = .8
learning_rate = .05
batch_size = 32 
#episodes = 300
n_episodes = 500

env = gym.make('CartPole-v0') #our environment (the pendulum simulation)

state_size = env.observation_space.shape[0] #4
action_size = env.action_space.n #2

observation_space = env.observation_space.shape[0]
action_space = env.action_space.n


    
class Agent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000) # double-ended queue; acts like list, but elements can be added/removed from either end
        self.gamma = 0.95 # decay or discount rate: enables agent to take into account future actions in addition to the immediate ones, but discounted at this rate
        self.epsilon = 1.0 # exploration rate: how much to act randomly; more initially than later due to epsilon decay
        self.epsilon_decay = 0.995 # decrease number of random explorations as the agent's performance (hopefully) improves over time
        self.epsilon_min = 0.01 # minimum amount of random exploration permitted
        self.learning_rate = 0.001  
        self.model = self._build_model()
        
    def _build_model(self):
        # neural net to approximate Q-value function:
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='sigmoid'))
        model.add(Dense(self.action_size, activation='linear')) 
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # list of previous experiences, enabling re-training later

    def act(self, state):
        if np.random.rand() <= self.epsilon: # if acting randomly, take random action
            return random.randrange(self.action_size)
        act_values = self.model.predict(state) # if not acting randomly, predict reward value based on current state
        return np.argmax(act_values[0]) # pick the action that will give the highest reward (i.e., go left or right?)

    def replay(self, batch_size): # method that trains NN with experiences sampled from memory
        minibatch = random.sample(self.memory, batch_size) # sample a minibatch from memory
        for state, action, reward, next_state, done in minibatch: # extract data for each minibatch sample
            target = reward # if done (boolean whether game ended or not, i.e., whether final state or not), then target = reward
            if not done: # if not done, then predict future discounted reward
                target = (reward + self.gamma * # (target) = reward + (discount rate gamma) * 
                          np.amax(self.model.predict(next_state)[0])) # (maximum target Q based on future action a')
            target_f = self.model.predict(state) # approximately map current state to future discounted reward
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0) # single epoch of training with x=state, y=target_f; fit decreases loss btwn target_f and y_hat
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)



agent = Agent(state_size,action_size)
done = False

max_score = 0
print("Episode \t Score \t Epsilon")
for e in range(n_episodes): # iterate over new episodes of the game
    state = env.reset() # reset state at start of each new episode of the game
    state = np.reshape(state, [1, state_size])

    avg_reward = np.array([])
    avgr = 0
    for time in range(5000):  # time represents a frame of the game; goal is to keep pole upright as long as possible up to range, e.g., 500 or 5000 timesteps

        env.render()

        action = agent.act(state) # action is either 0 or 1 (move cart left or right); decide on one or other here
        next_state, reward, done, _ = env.step(action) # agent interacts with env, gets feedback; 4 state data points, e.g., pole angle, cart position        
        reward = reward if not done else -10 # reward +1 for each additional frame with pole upright        
        next_state = np.reshape(next_state, [1, state_size])
        agent.remember(state, action, reward, next_state, done) # remember the previous timestep's state, actions, reward, etc.        
        state = next_state # set "current state" for upcoming iteration to the current next state
        
        
        if done: # episode ends if agent drops pole or we reach timestep 5000
            np.append(avg_reward,time)
            if e % 10 == 0:
##                print("episode: {}/{}, score: {}, e: {:.2}" # print the episode's score and agent's epsilon
##                      .format(e, n_episodes, time, agent.epsilon))
                
                print(e,"\t",time,"\t",agent.epsilon)
            if time > max_score:
                if time > 198:
                    agent.save(output_dir + "weights_" + '{:04d}'.format(e) + ".hdf5")
                    max_score = time
                    best_weights = output_dir + "weights_" + '{:04d}'.format(e) + ".hdf5"
                    break
            break # exit loop
        
    if len(agent.memory) > batch_size:
        agent.replay(batch_size) # train the agent by replaying the experiences of the episode
        
    #if e % 500 == 0:
        #agent.save(output_dir + "weights_" + '{:04d}'.format(e) + ".hdf5")
    
    
'weights_0253.hdf5'
