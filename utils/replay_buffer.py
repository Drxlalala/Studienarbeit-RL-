import torch
import numpy as np
import random
from collections import deque
from copy import deepcopy as dc


class ReplayBuffer:
  

    def __init__(self, capacity: int) -> None:
       

            :param capacity: (int) 
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=self.capacity)

    def push(self, transitions: tuple):
     

            :param transitions: (tuple) transition
        """
        self.buffer.append(transitions)

    def sample(self, batch_size: int = None, sequential: bool = True, with_log=True):
       

            :param batch_size: (int) 
            :param sequential: (bool) 
            :param with_log: (bool) 
            :return: (tuple)
        """
       
        if batch_size is None or batch_size > len(self.buffer):
            batch_size = len(self.buffer)

        if sequential:  #  sequential sampling
            rand = random.randint(0, len(self.buffer) - batch_size)
            batch = [self.buffer[i] for i in range(rand, rand + batch_size)]
        else:  # random
            batch = random.sample(self.buffer, batch_size)

        a_logprob = None
        if with_log:
            s, a, a_logprob, s_, r, dw, done = zip(*batch)
        else:
            s, a, s_, r, dw, done = zip(*batch)

        s = torch.tensor(np.asarray(s), dtype=torch.float)
        a = torch.tensor(np.asarray(a), dtype=torch.float)
        if with_log:
            a_logprob = torch.tensor(np.asarray(a_logprob), dtype=torch.float)
        s_ = torch.tensor(np.asarray(s_), dtype=torch.float)
        r = torch.tensor(np.asarray(r), dtype=torch.float).view(batch_size, 1)
        dw = torch.tensor(np.asarray(dw), dtype=torch.float).view(batch_size, 1)
        done = torch.tensor(np.asarray(done), dtype=torch.float).view(batch_size, 1)
        if with_log:
            return s, a, a_logprob, s_, r, dw, done
        else:
            return s, a, s_, r, dw, done

    def clear(self):
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)


class ReplayBufferDiscreteAction(ReplayBuffer):
    

    def __init__(self, capacity: int) -> None:
        

            :param capacity: (int) 经验回放池的容量
        """
        super().__init__(capacity)

    def sample(self, batch_size: int = None, sequential: bool = True, with_log=True):
     
        if batch_size is None or batch_size > len(self.buffer):
            batch_size = len(self.buffer)

        if sequential: 
            rand = random.randint(0, len(self.buffer) - batch_size)
            batch = [self.buffer[i] for i in range(rand, rand + batch_size)]
        else:  # 随机采样
            batch = random.sample(self.buffer, batch_size)

        a_logprob = None
        if with_log:
            s, a, a_logprob, s_, r, dw, done = zip(*batch)
        else:
            s, a, s_, r, dw, done = zip(*batch)

        s = torch.tensor(np.asarray(s), dtype=torch.float)
        a = torch.tensor(np.asarray(a), dtype=torch.int64)
        if with_log:
            a_logprob = torch.tensor(np.asarray(a_logprob), dtype=torch.float)
        s_ = torch.tensor(np.asarray(s_), dtype=torch.float)
        r = torch.tensor(np.asarray(r).reshape((batch_size, 1)), dtype=torch.float)
        dw = torch.tensor(np.asarray(dw).reshape((batch_size, 1)), dtype=torch.float)
        done = torch.tensor(np.asarray(done).reshape((batch_size, 1)), dtype=torch.float)
        if with_log:
            return s, a, a_logprob, s_, r, dw, done
        else:
            return s, a, s_, r, dw, done


class Trajectory:
  

    def __init__(self) -> None:
        self.buffer = []

    def push(self, transitions: tuple):
        

            :param transitions: (tuple)
        """
        self.buffer.append(transitions)

    def __len__(self):
        return len(self.buffer)


class HERReplayBuffer(ReplayBuffer):
    """ Hindisght Experience Replay Buffer """

    def __init__(self, capacity: int, k_future: int, env) -> None:
        super().__init__(capacity)
        self.env = env  
        self.future_p = 1 - (1. / (1 + k_future))

    def push(self, trajectory: Trajectory):
       

            :param trajectory: (Trajectory)
        """
        self.buffer.append(trajectory)

    def sample(self, batch_size: int = 256, sequential: bool = True, with_log=True, device='cpu'):
    
        ep_indices = np.random.randint(0, len(self.buffer), batch_size)
        time_indices = np.random.randint(0, len(self.buffer[0]), batch_size)
        states = []
        actions = []
        desired_goals = []
        next_states = []
        next_achieved_goals = []

     
        for episode, timestep in zip(ep_indices, time_indices):
            states.append(dc(self.buffer[episode].buffer[timestep][0]))
            actions.append(dc(self.buffer[episode].buffer[timestep][1]))
            next_states.append(dc(self.buffer[episode].buffer[timestep][2]))
            desired_goals.append(dc(self.buffer[episode].buffer[timestep][5]))
            next_achieved_goals.append(dc(self.buffer[episode].buffer[timestep][6]))

       
        states = np.vstack(states)
        actions = np.vstack(actions)
        desired_goals = np.vstack(desired_goals)
        next_achieved_goals = np.vstack(next_achieved_goals)
        next_states = np.vstack(next_states)

      
        her_indices = np.where(np.random.uniform(size=batch_size) < self.future_p)
        future_offset = np.random.uniform(size=batch_size) * (len(self.buffer[0]) - time_indices)
        future_offset = future_offset.astype(int)
        future_t = (time_indices + future_offset)[her_indices]

        future_ag = []
        for epi, f_offset in zip(ep_indices[her_indices], future_t):
            future_ag.append(dc(self.buffer[epi].buffer[f_offset][4]))
        future_ag = np.vstack(future_ag)

        desired_goals[her_indices] = future_ag
        rewards = np.expand_dims(self.env.unwrapped.compute_reward(next_achieved_goals, desired_goals, None), 1)

        s = torch.tensor(np.asarray(states), dtype=torch.float).to(device)
        a = torch.tensor(np.asarray(actions), dtype=torch.float).to(device)
        s_ = torch.tensor(np.asarray(next_states), dtype=torch.float).to(device)
        r = torch.tensor(np.asarray(rewards), dtype=torch.float).to(device)
        g = torch.tensor(np.asarray(desired_goals), dtype=torch.float).to(device)

        return s, a, s_, r, g
