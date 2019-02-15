import gym
import numpy as np

class envr:
    def __init__(self):
        self.states = [0,1,2,3,4,5]
        self.R = np.array(\
               [[-1,-1,-1,-1,0,-1],
                [-1,-1,-1,0,-1,100],
                [-1,-1,-1,0,-1,-1],
                [-1,0,0,-1,0,-1],
                [0,-1,-1,0,-1,100],
                [-1,0,-1,-1,0,100]])
        
    def actions(self,s):
        #return possible actions given state
        re = []
        c = 0
        for action in self.R[s]:
            if action > -1:
                re.append(c)
            c+=1
        return re
    
time = 5000
alpha = .5
gamma = .8

env = envr()

q = np.zeros_like(env.R)

episodes = 100

for e in range(episodes):
    done = False
    #random_init_state = 1
    random_init_state = np.random.randint(0,6)
    count = 0
    while not done:
        
        '''
Select one among all possible actions for the current state.
Using this possible action, consider going to the next state.
Get maximum Q value for this next state based on all possible actions.
Compute: Q(state, action) = R(state, action) + Gamma * Max[Q(next state, all actions)]
Set the next state as the current state.
End Do
'''
        random_action = np.random.choice(env.actions(random_init_state))
        next_state_from_random_action = random_action
        
        next_state_actions = env.actions(next_state_from_random_action)

        next_state_q = []
        for action in next_state_actions:
            next_state_q.append(q[next_state_from_random_action,action])

            
        max_q = max(next_state_q)

        #update q table
        first_move_reward = env.R[random_init_state][random_action]
        q[random_init_state][random_action] = (1-alpha)*q[random_init_state][random_action]+ \
                                              alpha*(env.R[random_init_state][random_action]+ \
                                                     gamma*max_q)
        
        count+=1

        if e % 25 == 0:
            print("- 25 episodes later -")
            print(q)
        if first_move_reward > 1 or count > 5:
            done = True


ex = False
actions = []


initial = int(input("starting state?"))
    
state = initial
while state != 5:
    best_action = np.argmax(q[state])
    state = best_action
    actions.append(best_action)
    
print(actions)
