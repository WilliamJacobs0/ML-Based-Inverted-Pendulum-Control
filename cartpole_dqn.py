import gym
import numpy as np
import random
import os
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

#save the weights of our models, if we want
output_dir = 'model_output/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

#hyperparameters
time = 5000
alpha = .5
gamma = .8
learning_rate = .05
batch_size = 32 
#episodes = 300
n_episodes = 500


#Thank you OpenAI Gym for providing us with these easy to use simulations!
#Make our environment
env = gym.make('CartPole-v0') 
state_size = env.observation_space.shape[0] #4 values for state. Cart position, cart velocity, pivot angle, pivot angular velocity 
action_size = env.action_space.n #2 possible actions... move cart left, or move cart right

observation_space = env.observation_space.shape[0]
action_space = env.action_space.n

    
class Agent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000) # Memory module for recording previous experiences 
        self.gamma = 0.95 # discount rate: used to discount future reward over closer rewards... as the Doors say, the future is uncertain 
        self.epsilon = 1.0 # exploration rate: the probability that the agent takes random actions for exploration purposes 
        self.epsilon_decay = 0.995 # decay exploration rate
        self.epsilon_min = 0.01 # minimum amount of random exploration
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
        # Remember experiences for further training
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        # Impose an action upon the environment
        if np.random.rand() <= self.epsilon: # if acting randomly, take random action
            return random.randrange(self.action_size)
        predictions = self.model.predict(state) # if not acting randomly, predict reward value based on current state
        return np.argmax(predictions[0]) # Compare the predicted reward of going left vs going right and choose the higher one
    
    def replay(self, batch_size): 
        # Experience Replay module as inspired by Minh et al. 
        minibatch = random.sample(self.memory, batch_size) # sample a minibatch from memory
        for state, action, reward, next_state, done in minibatch: # unpack data from memory
            target = reward # if done (whether the game ended here or not), then target = reward
            if not done: # if not done, then predict future discounted reward using the classic Q-Learning equation
                target = (reward + self.gamma * # (target) = reward + (discount rate gamma, to discount future rewards) * 
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
for e in range(n_episodes): # Begin training loop, iterating over game episodes
    state = env.reset() # Reset state after each iteration
    state = np.reshape(state, [1, state_size])

    avg_reward = np.array([])
    avgr = 0
    
    for time in range(5000):  
        # the upper limit on time is the amount of frames we want one iteration to last, assuming the game does not end beforehand
        env.render()

        action = agent.act(state) # go left or go right - action is 0 or 1 
        next_state, reward, done, _ = env.step(action) # agent interacts with environment    
        reward = reward if not done else -10 # reward +1 for each additional frame with pole upright        
        next_state = np.reshape(next_state, [1, state_size])
        agent.remember(state, action, reward, next_state, done) # engage memory module    
        state = next_state # set "current state" to the next state
                
        if done: # episode ends if agent drops pole or we reach max timestep
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
        agent.replay(batch_size) # engage experience-replay module
        
    #if e % 500 == 0:
        #agent.save(output_dir + "weights_" + '{:04d}'.format(e) + ".hdf5")
        # save weights if you get a good model trained :)
