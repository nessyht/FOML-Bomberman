
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
import pickle


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
    print(vector.shape)
    
    # Confirmed: vector has the right shape and content
    # END OF CHANGED KT
    # return final state vector
    return vector
    

def training(states, actions, rewards):
    """
    states: a flattened numpy array representing the occurred states
    actions: a list of actions performed after respective state occurred
    rewards: a list of rewards received after respective action was performed
    """
    
    # Check whether all arguments have the same number of entries
    n = states.shape[0]
    if len(actions) != n or len(rewards) != n:
        print('ERROR in training(): No matching number of states, rewards, actions.')
        print('Number of entries:')
        print('states:', n)
        print('actions:', len(actions))
        print('rewards:', len(rewards))
        return # stop training
    
    moves = ['UP','DOWN','LEFT','RIGHT','WAIT','BOMB']
    
    for move in moves:        
        regressor = RandomForestRegressor() # Initialize Random Forest Regressor  
        #regressor = MLPRegressor(max_iter=500) # Initialize MLP Regressor (Neural Network)
        
        print('Fitting ', move,'.') # Inform user about progress
        # Fit regressor on respective states/rewards
        regressor.fit(states[actions==move], rewards[actions==move])
        pickle.dump(regressor, open(move + '.txt', 'wb')) # Store regressor in file e.g. 'UP.txt'
        
    print('Regressors stored.')

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
    