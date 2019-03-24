
import numpy as np

from settings import s, e

def setup(agent):
    np.random.seed()

def act(agent):
    agent.logger.info('Pick action at random')
    agent.next_action = np.random.choice(['RIGHT', 'LEFT', 'UP', 'DOWN', 'WAIT', 'BOMB'], p=[.18, .18, .18, .18, .18, .1])

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
    
    output = []
    for y in range(rewards.shape[0]):
        output.append(np.sum(rewards[y:]*gammas[:rewards.shape[0]-y]))
    self.rewards = output
    # CHANGED KT