
import numpy as np
import pickle
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
        

def create_state_vector(self):
    """
    This function is called in agents.py after act(self) was called.
    This function stores/returns the current state of the game in a numpy array.
    The current state of the game is stored in 'self'.
    """
    # CHANGED KT
    # Import available data
    arena = self.game_state['arena'].copy()    
    
    ''' 
    Create 3 arena shaped arrays for state information to merge into state vector with field information:
    [Agent present: -1 for enemy, 0 for none, 1 for self;
     Coin or crate present: -1 for crate, 0 for none, 1 for coin;
     bomb or explosion present: 5 for none, t for timer]
    '''
    agent_state = np.zeros((arena.shape))
    #loot_state = np.zeros((arena.shape))
    #bomb_state = np.zeros((arena.shape))

    x, y, _, bombs_left = self.game_state['self']
    bombs = self.game_state['bombs'].copy()
    others = [(x,y) for (x,y,n,b) in self.game_state['others']]
    coins = self.game_state['coins']
    step = self.game_state['step']
    explosions = self.game_state['explosions'].copy()
    
    # May need to be extended dependent on the changes in state definition
    extras = np.zeros((4))
    
    # For every cell: 

    # Agent on cell, self
    agent_state[x, y] = 1
    
    # Agent on cell, enemy
    for o in others:
        agent_state[o[0], o[1]] = -1
        
    # Crate on cell? 
    loot_state = np.where(arena == 1, -1, 0)
            
    # Coin on cell?
    for coin in coins:
        loot_state[coin[0], coin[1]] = 1
    
    # Bomb radius on cell?    
    bomb_state = np.ones(arena.shape) * 6
    
    for xb,yb,t in bombs:
        for (i,j) in [(xb+h, yb) for h in range(-3,4)] + [(xb, yb+h) for h in range(-3,4)]:
            if (0 < i < bomb_state.shape[0]) and (0 < j < bomb_state.shape[1]):
                bomb_state[i,j] = min(bomb_state[i,j], t)
    # Danger level
    extras[1] = 6 - bomb_state[x, y]
                    
    # Exlosions on cell?
      
    bomb_state[np.where(explosions == 1)] = 0
    bomb_state[np.where(explosions == 2)] = 1

    # Only once:
    # Current step
    extras[0] = step
    

    # Bomb action possible?
    extras[2] = bombs_left
    
    # Touching enemy
    if len(others) > 0:
        if (min(abs(xy[0] - x) + abs(xy[1] - y) for xy in others)) <= 1:
            extras[3] = 1
    
    # Reward for step (later)
    
    # Reward for episode (added later)
    
    # State of each cell
    # combine state maps and flatten into 1D-array
    state = np.stack((agent_state[arena != -1].flatten(), \
                      loot_state[arena != -1].flatten(), \
                      bomb_state[arena != -1].flatten()), \
                      axis=-1).flatten()
    
    vector = np.concatenate((state, extras)) # combine extras and state vector
    #print('Shape of a single state vector:',vector.shape)
    
    # Confirmed: vector has the right shape and content
    # END OF CHANGED KT
    # return final state vector
    return vector

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
    
    if e.MOVED_LEFT in self.events:
        reward -= 1
    if e.MOVED_RIGHT in self.events:
        reward -= 1
    if e.MOVED_UP in self.events:
        reward -= 1
    if e.MOVED_DOWN in self.events:
        reward -= 1
    if e.WAITED in self.events:
        reward -= 5
    if e.BOMB_DROPPED in self.events:
        reward -= 1
    if e.INVALID_ACTION in self.events:
        reward -= 100
    if e.CRATE_DESTROYED in self.events:
        reward += 10            
    if e.COIN_FOUND in self.events:
        reward += 20
    if (e.BOMB_EXPLODED in self.events) and not (e.KILLED_SELF in self.events):
        reward += 50
    if e.COIN_COLLECTED in self.events:
        reward += 2000
    if e.KILLED_OPPONENT in self.events:
        reward += 10000
    if e.GOT_KILLED in self.events:
        reward -= 2000
    if e.KILLED_SELF in self.events:
        reward -= 1500
        
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
    if e.MOVED_LEFT in self.events:
        reward -= 1
    if e.MOVED_RIGHT in self.events:
        reward -= 1
    if e.MOVED_UP in self.events:
        reward -= 1
    if e.MOVED_DOWN in self.events:
        reward -= 1
    if e.WAITED in self.events:
        reward -= 5
    if e.BOMB_DROPPED in self.events:
        reward -= 1
    if e.INVALID_ACTION in self.events:
        reward -= 100
    if e.CRATE_DESTROYED in self.events:
        reward += 10            
    if e.COIN_FOUND in self.events:
        reward += 20
    if e.BOMB_EXPLODED in self.events and not e.KILLED_SELF in self.events:
        reward += 20
    if e.COIN_COLLECTED in self.events:
        reward += 100
    if e.KILLED_OPPONENT in self.events:
        reward += 500
    if e.GOT_KILLED in self.events:
        reward -= 500
    if e.KILLED_SELF in self.events:
        reward -= 400
    if e.KILLED_OPPONENT in self.events:
        reward += 10000
    if e.GOT_KILLED in self.events:
        reward -= 2000
    if e.KILLED_SELF in self.events:
        reward -= 1500
        
    self.rewards.append(reward)
    # CHANGED KT