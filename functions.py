import numpy as np

def create_state_vector(self):
    """
    This function is called in agents.py after act(self) was called.
    This function stores/returns the current state of the game in a numpy array.
    The current state of the game is stored in 'self'.
    """
    # CHANGED KT-26.02/27-02
    # Import available data
    arena = self.game_state['arena'].copy()    
    
    ''' Create 3 arena shaped arrays for state information to merge into state vector with field information:
        [Agent present: -1 for enemy, 0 for none, 1 for self;
         Coin or crate present: -1 for crate, 0 for none, 1 for coin;
         bomb or explosion present: 5 for none, t for timer]
    '''
    agent_state = np.zeros((arena.shape))
    #loot_state = np.zeros((arena.shape))
    #bomb_state = np.zeros((arena.shape))
    x, y, _, bombs_left = self.game_state['self']
    bombs = self.game_state['bombs']
    others = [(x,y) for (x,y,n,b) in self.game_state['others']]
    coins = self.game_state['coins']
    step = self.game_state['step']
    explosions = self.game_state['explosions']
                
    # May need to be extended dependent on the changes in state definition
    extras = np.zeros((2))
    
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
    bomb_state = np.ones(arena.shape) * 5
    for xb,yb,t in bombs:
        for (i,j) in [(xb+h, yb) for h in range(-3,4)] + [(xb, yb+h) for h in range(-3,4)]:
            if (0 < i < bomb_state.shape[0]) and (0 < j < bomb_state.shape[1]):
                bomb_state[i,j] = min(bomb_state[i,j], t)
    
    # Explosion on cell? Important?
    

    # Only once:
    # Current step
    
    # Danger level
    extras[0] = 5 - bomb_state[x, y]
    
    # Bomb action possible?
    
    # Reward for step
    # extras[1] = self.rewards[-1]
    
    # Step number
    
    
    # Reward for episode (added later)
    
    
    # State of each cell
    # combine state maps and flatten into 1D-array
    state = np.stack((agent_state[arena != -1].flatten(), \
                      loot_state[arena != -1].flatten(), \
                      bomb_state[arena != -1].flatten()), \
                      axis=-1).flatten()
    
    vector = np.concatenate((extras, state)) # combine extras and state vector
 
    
    # Confirmed: vector has the right shape and content
    # CHANGED KT-26.02/27.02
    # return final state vector
    return vector
    
def store_next_action(self):
    """
    This function is called in agents.py after act(self) was called.
    This function stores/returns the action chosen by act(self).
    The chosen action is stored in self.next_action
    """
    
    return self.next_action