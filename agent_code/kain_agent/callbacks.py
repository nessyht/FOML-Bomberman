
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

from settings import e

def setup(self):
    np.random.seed()
    moves = ['UP','DOWN','LEFT','RIGHT','WAIT','BOMB']
    self.regressors = []

    for move in moves:
        self.regressors.append(pickle.load(open('013' + '_' + move + '.txt', 'rb')))
        #print('013' + '_' + move + '.txt')
        

def training(states, actions, rewards, generation):
    """
    states: a flattened numpy array representing the occurred states
    actions: a list/numpy array of actions performed after respective state occurred
    rewards: a list/numpy array of rewards received after respective action was performed
    generation: int, generation number
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
        
        print('Fitting ', f'{generation:03}', move) # Inform user about progress
        
        # Fit regressor on respective states/rewards
        regressor.fit(states[actions==move], rewards[actions==move])
        
         # Store regressor in file e.g. '000_UP.txt', '001_UP.txt', ...
        pickle.dump(regressor, open('agent_code/my_agent/Training_data/trees/' + f'{generation:03}' + '_' + move + '.txt', 'wb'))
        
    print('Regressors stored.')
    

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
    
    x, y, _, bombs_left, s = self.game_state['self']
    bombs = self.game_state['bombs'].copy()
    others = [(x,y) for (x,y,n,b,s) in self.game_state['others']]
    coins = self.game_state['coins']
    step = self.game_state['step']
    explosions = self.game_state['explosions'].copy()
    
    
    # May need to be extended dependent on the changes in state definition
    extras = np.zeros((4))
    
    # For every cell: 

    # Agent on cell: 1
    agent_state[x, y] = 1
    
    # Opponent on cell: 2
    for o in others:
        agent_state[o[0], o[1]] = 2
        
    # Crate on cell: 3 
    agent_state += np.where(arena == 1, 3, 0)
            
    # Coin on cell: 4
    for coin in coins:
        agent_state[coin[0], coin[1]] = 4
    
    # Bomb radius on cell?    
    bomb_state = np.ones(arena.shape) * 6
    
    for xb,yb,t in bombs:
        for (i,j) in [(xb+h, yb) for h in range(-3,4)] + [(xb, yb+h) for h in range(-3,4)]:
            if (0 < i < bomb_state.shape[0]) and (0 < j < bomb_state.shape[1]):
                bomb_state[i,j] = min(bomb_state[i,j], t)
                
    # Exlosions on cell?      
    bomb_state[np.where(explosions == 1)] = 0
    bomb_state[np.where(explosions == 2)] = 1
    
    bomb_state = 11 - bomb_state 
    # 6 - bomb_state + 5 because values 0 to 4 have other meanings in agent_state
    
    
    
    # Overwrite values of dangerous cells; 5 in bomb_state means no danger
    # -> stored values are 6 to 11
    # -> 5 is still without meaning
    agent_state[bomb_state != 5] = bomb_state[bomb_state != 5]
    
    # Danger level
    extras[0] = bomb_state[x, y]
    

    # Bomb action possible?
    extras[1] = bombs_left
    
    # Touching enemy
    if len(others) > 0:
        if (min(abs(xy[0] - x) + abs(xy[1] - y) for xy in others)) <= 1:
            extras[2] = 1
    
    # Position of agent:
    extras[3] = 17*x + y
    # Reward for step (later)
    
    # Reward for episode (added later)
    
    agent_state = agent_state.flatten()
    
    state = agent_state[(arena != -1).flatten()]
    
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
    
    explore = False
    #if self.train_flag.is_set():  
    #    explore = np.random.choice([True, False], p=[0.25, 0.75])
    self.next_action = choose_action(self.regressors, create_state_vector(self), exploring=explore)  
    
    arena = self.game_state['arena'].copy() 
    x, y, _, bombs_left = self.game_state['self']
    
    invalid = False
    if (self.next_action == 'UP' and arena[x,y-1] != 0) or (self.next_action == 'Down' and arena[x,y+1] != 0) or (self.next_action == 'LEFT' and arena[x-1,y] != 0) or (self.next_action == 'RIGHT' and arena[x+1,y] != 0) or (self.next_action == 'BOMB' and not bombs_left):
        invalid = True
                        
    #print(self.next_action, ('-Valid' if not invalid else '-Invalid'), '(', x, y, ')')
    
    
    
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
        reward += 50
    if e.COIN_COLLECTED in self.events:
        reward += 2000
    if e.KILLED_OPPONENT in self.events:
        reward += 10000
    if e.GOT_KILLED in self.events:
        reward -= 2000
    if e.KILLED_SELF in self.events:
        reward -= 1500
    
    Gamma = 0.9    
    self.rewards.append(reward)
    rewards = np.array(self.rewards)
    gammas = Gamma*np.arange(len(self.rewards))
    
    for y in range(rewards.shape[0]):
        output.append(np.sum(rewards[y:]*gammas[:rewards.shape[0]-y]))
    self.rewards = output
    # CHANGED KT