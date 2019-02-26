import numpy as np

def create_state_vector(self):
    """
    This function is called in agents.py after act(self) was called.
    This function stores/returns the current state of the game in a numpy array.
    The current state of the game is stored in 'self'.
    """
    
    # Import available data
    arena = self.game_state['arena'].copy()
    x, y, _, bombs_left = self.game_state['self']
    bombs = self.game_state['bombs']
    bomb_xys = [(x,y) for (x,y,t) in bombs]
    others = [(x,y) for (x,y,n,b) in self.game_state['others']]
    coins = self.game_state['coins']
    step = self.game_state['step']
    explosions = self.game_state['explosions']
    
    vector = np.array([])
    
    # For every cell:
    
    # State of each cell
    arena = np.reshape(arena, (arena.shape[0]*arena.shape[1],)) # reshape arena into 1D-array
    arena = np.delete(arena, np.where(arena==-1)) # delete walls. Those are the same everytime. Thus, we save memory
    vector = np.concatenate((vector,arena)) # combine vector and arena
    # Confirmed: vector has the right shape
    # Not confirmed: vector has right content
    
    
    # Agent on cell?
    
    
    # Opponent on cell?
    
    
    # Coin on cell?
    
    
    # Explosion on cell?
    
    
    # Danger level
    
    
    # Only once:
    # Current step
    
    
    # Bomb action possible?
    
    
    # Danger level for the agent
    
    
    # Agent touches opponent
    
    
    # return final state vector
    return vector
    
def store_next_action(self):
    """
    This function is called in agents.py after act(self) was called.
    This function stores/returns the action chosen by act(self).
    The chosen action is stored in self.next_action
    """
    
    return self.next_action