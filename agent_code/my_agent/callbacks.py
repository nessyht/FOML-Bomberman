
import numpy as np
import pickle
from functions import create_state_vector


def setup(self):
    np.random.seed()
    moves = ['UP','DOWN','LEFT','RIGHT','WAIT','BOMB']
    self.regressors = []
    self.generation = 1 # Let this be externally fixed
    for move in moves:
        self.regressors.append(pickle.loads(open('Training_data/' + f'{self.generation:03}' + '_' + move + '.txt', 'wb')))
        

def MB_probs(rewards, T=100):
    """
    returns probabilities for exploring based on a Max-Boltzman-distribution
    rewards: array/list with expected rewards
    """
    Q = np.array(rewards)
    denom = np.sum(np.exp(np.divide(Q,T)))
    return np.divide(np.exp(np.divide(Q,T)),denom)
    
def choose_action(regressor_list, state, exploring=False, epsilon=0.3):
    """
    regressor_list: list of regressors in order: up,down,left,right,wait,bomb
    state: the state for which to find the best action
    exploring: whether agent should explore different actions
    epsilon: probability with which we will explore
    """
    
    # list of actions in the same order as corresponding actions in regressor_list
    actions = ['UP','DOWN','LEFT','RIGHT','WAIT','BOMB']
    
    predicted_reward = []
    
    # predict expected reward for each action
    for reg in regressor_list:
        predicted_reward.append(reg.predict(state))
    
    exploit = np.argmax(predicted_reward) # Index of action with highest reward
    
    if not exploring:
        # Choose action with highest expected reward
        return actions[exploit]
    
    # Will we explore?
    explore = np.random.choice([True, False], p=[epsilon, 1-epsilon])
    
    if not explore:
        # Choose action with highest expected reward
        return actions[exploit]
    
    if explore:
        # Remove highest reward from selection
        actions.pop([exploit])
        predicted_reward.pop([exploit])
        
        mean_reward_size = np.mean(np.abs(predicted_reward))
        probabilities = MB_probs(predicted_reward, T=mean_reward_size)
        
        return np.random.choice(actions, p=probabilities)
    
def act(self):
    self.logger.info('Pick action from trees')


    self.next_action = choose_action(regressors, create_state_vector(self), True)
    print(self.next_action)
    
def reward_update(agent):
    pass

def end_of_episode(agent):
    pass
