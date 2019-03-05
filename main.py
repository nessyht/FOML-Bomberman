
from time import time, sleep
import contextlib

with contextlib.redirect_stdout(None):
    import pygame
from pygame.locals import *
import numpy as np
import multiprocessing as mp
import threading
import pickle

from environment import BombeRLeWorld, ReplayWorld
from settings import s
from functions import training

# Function to run the game logic in a separate thread
def game_logic(world, user_inputs):
    last_update = time()
    while True:
        # Game logic
        if (s.turn_based and len(user_inputs) == 0):
            sleep(0.1)
        elif (s.gui and (time()-last_update < s.update_interval)):
            sleep(s.update_interval - (time() - last_update))
        else:
            last_update = time()
            if world.running:
                try:
                    world.do_step(user_inputs.pop(0) if len(user_inputs) else 'WAIT')
                except Exception as e:
                    world.end_round()
                    raise

def main():
    agents = [('my_agent', True),
              ('simple_agent', False),
              ('simple_agent', False),
              ('simple_agent', False)]
    
    train_main(agents, 5,[0,1,2,3])
    
def train_main(agents, episodes, generations_list):
    '''
    agents: array of tuples of agent dir names and training flag
    episodes: number of episodes to be played per generation
    generations_list: list of the generations that should be used to generate training data for (later with input to train from in callbacks to select correct trees); 0 = 4 simples
    '''
    pygame.init()
    start_time = time()

    # Emulate Windows process spawning behaviour under Unix (for testing)
    # mp.set_start_method('spawn')
    for generation in generations_list:
        # Initialize environment and agents
        # 0th generation is trained on 4 simples, after it is users choice
        gen_time = time()
        
        if generation == 0:
            world = BombeRLeWorld([
                    ('simple_agent', True),
                    ('simple_agent', True),
                    ('simple_agent', True),
                    ('simple_agent', True)
                ], generation)
        else:
            world = BombeRLeWorld(agents, generation)
            
        # Pass generation to world in order to pass it to agents
        #world.generation = generation

        # world = ReplayWorld('Replay 2019-01-30 16:57:42')
        user_inputs = []
        
        # Start game logic thread
        t = threading.Thread(target=game_logic, args=(world, user_inputs))
        t.daemon = True
        t.start()
        
        # Run one or more games
        for i in range(episodes):
            
            if not world.running:
                world.ready_for_restart_flag.wait()
                world.ready_for_restart_flag.clear()
                world.new_round()
              
            round_finished = False
            last_update = time()
            last_frame = time()
            user_inputs.clear()
        
            # Main game loop
            while not round_finished:
                # Grab events
                key_pressed = None
                for event in pygame.event.get():
                    if event.type == QUIT:
                        world.end_round()
                        world.end()
                        return
                    elif event.type == KEYDOWN:
                        key_pressed = event.key
                        if key_pressed in (K_q, K_ESCAPE):
                            world.end_round()
                        if not world.running:
                            round_finished = True
                        # Convert keyboard input into actions
                        if s.input_map.get(key_pressed):
                            if s.turn_based:
                                user_inputs.clear()
                            user_inputs.append(s.input_map.get(key_pressed))
        
                if not world.running:
                    round_finished = True

                sleep_time = 1/s.fps - (time() - last_frame)
                if sleep_time > 0:
                    sleep(sleep_time)
                last_frame = time()
            
            # CHANGED KT
            # End of a round occurres here
            # This is what happens after round_finished was set True
            
            # Add data of the current round to the data of the entire season
            # print('Passing current round states to world states.')
            # print('Current Round States Shape:', world.current_round_states.shape)
        
            if world.states is None:
                world.states = world.current_round_states
            else:
                world.states = np.concatenate((world.states, world.current_round_states))            
                # print('Shape of world states:', world.states.shape)
            world.actions.extend(world.current_round_actions)   
            
            print('Ending round ' + str(i) + ' of ' + str(episodes) + ' of generation ' + str(generation))
        
            # print('World States Shape:', world.states.shape)        
            # print('Length of world actions:', len(world.actions))         
            #world.rewards.extend(world.current_round_rewards)
            
            # END OCHANGED KT
        
        # CHANGED:
        # Arrays which contain the training data generated by the entire season (n rounds):
        
        # print('Passing wold states to global states.')
        states = world.states  # All states occurred during the season
        actions = world.actions # All actions chosen after respective state occurred
        #rewards = world.rewards # All cummulated rewards received after respective state occurred
        
        print('Shape of final state vector:',states.shape)
        # print('Shape of final actions:',len(actions))
        # print('Shape of final rewards', states[:, -2].shape)
        
        # Train regressor
        training(states[:,:-2], np.array(actions), states[:, -1], generation)
        
        # Store training data
        pickle.dump([states, actions], open('agent_code/my_agent/Training_data/data/' + f'{generation:03}' + '_data.txt', 'wb'))
        print('States stored')
        print('Time taken to gather training data for generation ' + str(generation) + ' was:', time()-gen_time,
              '\nAverage episode time =', (time()-gen_time)/episodes)
        
        world.end()
    print('Total time taken was:', time()-start_time)
    # END OF CHANGED


def game_main(agents, episodes, get_stats=False, save_replays=False):
    '''
    agents: array of tuples of agent dir names and training flag
    episodes: number of episodes to be played
    get_stats: flag to indicate whether stats should be gathered
    save_replays: flag to indicate whether replays should be saved
    '''
    pygame.init()
    
    # Emulate Windows process spawning behaviour under Unix (for testing)
    # mp.set_start_method('spawn')

    # Initialize environment and agents
    world = BombeRLeWorld(agents)
    
    # world = ReplayWorld('Replay 2019-01-30 16:57:42')
    user_inputs = []

    # Start game logic thread
    t = threading.Thread(target=game_logic, args=(world, user_inputs))
    t.daemon = True
    t.start()

    # Run one or more games
    for i in range(s.n_rounds):
        if not world.running:
            world.ready_for_restart_flag.wait()
            world.ready_for_restart_flag.clear()
            world.new_round()

        # First render
        if s.gui:
            world.render()
            pygame.display.flip()

        round_finished = False
        last_update = time()
        last_frame = time()
        user_inputs.clear()

        # Main game loop
        while not round_finished:
            # Grab events
            key_pressed = None
            for event in pygame.event.get():
                if event.type == QUIT:
                    world.end_round()
                    world.end()
                    return
                elif event.type == KEYDOWN:
                    key_pressed = event.key
                    if key_pressed in (K_q, K_ESCAPE):
                        world.end_round()
                    if not world.running:
                        round_finished = True
                    # Convert keyboard input into actions
                    if s.input_map.get(key_pressed):
                        if s.turn_based:
                            user_inputs.clear()
                        user_inputs.append(s.input_map.get(key_pressed))

            if not world.running and not s.gui:
                round_finished = True

            # Rendering
            if s.gui and (time()-last_frame >= 1/s.fps):
                world.render()
                pygame.display.flip()
                last_frame = time()
            else:
                sleep_time = 1/s.fps - (time() - last_frame)
                if sleep_time > 0:
                    sleep(sleep_time)
                if not s.gui:
                    last_frame = time()

    world.end()

if __name__ == '__main__':
    main()
