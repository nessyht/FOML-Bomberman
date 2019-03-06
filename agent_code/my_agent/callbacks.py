
import numpy as np
import pickle
from functions import create_state_vector
from settings import e

def setup(self):
    np.random.seed()
    moves = ['UP','DOWN','LEFT','RIGHT','WAIT','BOMB']
    self.regressors = []
    #self.generation = 0 # Let this be externally fixed
    if self.train_flag.is_set():
        generation = self.generation - 1
    else:
        generation = self.generation
    
    print('Agent now training with generation ', generation)
    for move in moves:
        self.regressors.append(pickle.load(open('agent_code/my_agent/Training_data/trees/' + f'{generation:03}' + '_' + move + '.txt', 'rb')))
        

def MB_probs(rewards, T=100):
    """
    returns probabilities for exploring based on a Max-Boltzman-distribution
    rewards: array/list with expected rewards
    """
    Q = np.array(rewards)
    denom = np.sum(np.exp(np.divide(Q,T)))
    return np.divide(np.exp(np.divide(Q,T)),denom)
    
def choose_action(regressor_list, state, exploring=False, epsilon=0.25):
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
        predicted_reward.append(reg.predict(state.reshape(1, -1)))
        
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
        actions.pop(exploit)
        predicted_reward.pop(exploit)
        
        mean_reward_size = np.mean(np.abs(predicted_reward))
        probabilities = MB_probs(predicted_reward, T=mean_reward_size).flatten()
        return np.random.choice(actions, p=probabilities)
    
def act(self):
    self.logger.info('Pick action from trees')
    
    self.next_action = choose_action(self.regressors, create_state_vector(self), exploring=self.train_flag.is_set())
    print(self.next_action)
    
def reward_update(self):
    """Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occured during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state. In
    contrast to act, this method has no time limit.
    """
    
    self.logger.debug(f'Encountered {len(self.events)} game event(s)')
    
    # what to do when interrupted or when round survived?
    # CHANGED KT
    reward = 0
    for event1, event2 in self.events:

        if event1 == e.MOVED_LEFT:
            reward -= 1
        if event1 == e.MOVED_RIGHT:
            reward -= 1
        if event1 == e.MOVED_UP:
            reward -= 1
        if event1 == e.MOVED_DOWN:
            reward -= 1
        if event1 == e.MOVED_WAIT:
            reward -= 10
        if event1 == e.BOMB_DROPPED:
            reward -= 1
        if event1 == e.INVALID_ACTION:
            reward -= 100
        if event1 == e.CRATE_DESTROYED:
            reward += 10            
        if event1 == e.COIN_FOUND:
            reward += 20
        if event1 == e.BOMB_EXPLODED and not event2 == e.KILLED_SELF:
            reward += 20
        if event1 == e.COIN_COLLECTED:
            reward += 100
        if event1 == e.KILLED_OPPONENT:
            reward += 500
        
        
    self.rewards.append(reward)
    # CHANGED KT
def end_of_episode(self):
    """Called at the end of each game to hand out final rewards and do training.

    This is similar to reward_update, except it is only called at the end of a
    game. self.events will contain all events that occured during your agent's
    final step. You should place your actual learning code in this method.
    """
    self.logger.debug(f'Encountered {len(self.events)} game event(s) in final step')

    # CHANGED KT
    reward = 0
    for event1, event2 in self.events:
        
        if event1 == e.MOVED_LEFT:
            reward -= 1
        if event1 == e.MOVED_RIGHT:
            reward -= 1
        if event1 == e.MOVED_UP:
            reward -= 1
        if event1 == e.MOVED_DOWN:
            reward -= 1
        if event1 == e.MOVED_WAIT:
            reward -= 10
        if event1 == e.BOMB_DROPPED:
            reward -= 1
        if event1 == e.GOT_KILLED:
            reward -= 500
        if event1 == e.KILLED_SELF:
            reward -= 400
        if event1 == e.INVALID_ACTION:
            reward -= 100
        if event1 == e.CRATE_DESTROYED:
            reward += 10            
        if event1 == e.COIN_COLLECTED:
            reward += 100
        if event1 == e.CRATE_DESTROYED:
            reward += 10            
        if event1 == e.COIN_FOUND:
            reward += 20
        if event1 == e.BOMB_EXPLODED and not event2 == e.KILLED_SELF:
            reward += 20
        if event1 == e.COIN_COLLECTED:
            reward += 100
        if event1 == e.KILLED_OPPONENT:
            reward += 500
    
    self.rewards.append(reward)
    # CHANGED KT