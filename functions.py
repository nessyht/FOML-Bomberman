import numpy as np

def create_state_vector(self):
    """
    This function is called in agents.py after act(self) was called.
    This function stores/returns the current state of the game in a numpy array.
    The current state of the game is stored in 'self'.
    """
    
    # Import available data
    arena = self.game_state['arena'].copy()    
    
    ''' Create empty arena with 3D vector per cell for state information
        [Agent present: -1 for enemy, 0 for none, 1 for self;
         Coin or crate present: -1 for crate, 0 for none, 1 for coin;
         bomb or explosion present: 0 for none, t for timer]
    '''
    state = np.zeros((arena.shape[0], arena.shape[1], 3))
    
    x, y, _, bombs_left = self.game_state['self']
    bombs = self.game_state['bombs']
    bomb_xys = [(x,y) for (x,y,t) in bombs]
    others = [(x,y) for (x,y,n,b) in self.game_state['others']]
    coins = self.game_state['coins']
    step = self.game_state['step']
    explosions = self.game_state['explosions']

                
    extras = np.array([])
    
    # For every cell: 

    # Agent on cell, self
    state[x, y, 0] = 1
    
    # Agent on cell, enemy
    for o in others:
        state[o[0], o[1], 0] = -1
        
    # Crate on cell? 
    state[:, :, 1] = np.where(arena == 1, -1, 0)
            
    # Coin on cell?
    for coin in coins:
        state[coin[0], coin[1], 1] = 1
   
    # Not functional yet
    
    # Bomb radius on cell?    

    for xb,yb,t in bombs:
        for (i,j) in [(xb+h, yb) for h in range(-3,4)] + [(xb, yb+h) for h in range(-3,4)]:
            if (0 < i < state.shape[0]) and (0 < j < state.shape[1]):
                state[i,j, 2] = min(state[i,j, 2], t)
    # not yet functional
    
    # Explosion on cell? Important?
    
    # Danger level
    
    # Only once:
    # Current step
    
    
    # Bomb action possible?
    
    
    # Danger level for the agent
    
    
    # Agent touches opponent
    
    # State of each cell
    # state = np.reshape(arena, (arena.shape[0]*arena.shape[1],)) # reshape arena into 1D-array
    state.flatten()
    walls = np.argwhere(arena == -1)
    offset = arena.shape[0]
    indexes = np.concatenate((3*np.add(walls[:,0]*offset, walls[:,1]), 3*np.add(walls[:,0]*offset, walls[:,1]+1), 3*np.add(walls[:,0]*offset, walls[:,1]+2)))
    # Indexes correct confirmed
    state = np.mgrid[0:17, 0:17, 0:3].flatten()
    vector = np.delete(state, indexes, axis=0).flatten() # delete walls. Those are the same everytime. Thus, we save memory
    vector = np.concatenate((extras, vector)) # combine extras and state vector
    print(state[0:17], state[state%17==0])
    print(vector)
    print(np.argwhere(arena == -1))
    print(3*np.add(walls[18,0]*offset, walls[18,1]))
    print(state.shape, vector.shape, 17*17*3, 176*3)
 
    
    # Confirmed: vector has the right shape
    # Not confirmed: vector has right content
    
    # return final state vector
    return vector
    
def store_next_action(self):
    """
    This function is called in agents.py after act(self) was called.
    This function stores/returns the action chosen by act(self).
    The chosen action is stored in self.next_action
    """
    
    return self.next_action