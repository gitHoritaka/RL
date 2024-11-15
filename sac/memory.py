import random
import numpy as np


class ReplayMemory:

    def __init__(self, memory_size):
        self.memory_size = memory_size
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, mask):
        if len(self.buffer) < self.memory_size:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, mask)
        self.position = (self.position + 1) % self.memory_size

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state_list, action_list, reward_list, next_state_list, done_list = zip(*batch)
        
        # Ensure all states and actions have the same shape
        # If they don't, you need to handle this before stacking
        state = np.stack(state_list)
        action = np.stack(action_list)
        next_state = np.stack(next_state_list)
        
        # Use np.array for scalars
        reward = np.array(reward_list)
        done = np.array(done_list)
    
        return state, action, reward, next_state, done


    def __len__(self):
        return len(self.buffer)