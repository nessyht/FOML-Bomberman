
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
import pickle 

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
    