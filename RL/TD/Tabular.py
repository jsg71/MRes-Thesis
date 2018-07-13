#import matplotlib.pyplot as plt
import numpy as np
#from collections import namedtuple,  defaultdict
from abc import ABC, abstractmethod

class TabularAgent(ABC):
  def __init__(self, number_of_states, number_of_actions, initial_state, 
               SARSA= True, double= False, eps=0.1, step_size=0.1):

    self._q = np.zeros((number_of_states, number_of_actions))
    if double:
      self._q2 = np.zeros((number_of_states, number_of_actions))
    
    self._s = initial_state
    self._number_of_actions = number_of_actions
    self._step_size = step_size
    
    self._last_action = np.random.randint(number_of_actions)
    self._double = double
    self._eps = eps

    if SARSA:
        self._target_policy = self._sarsa_target_policy
    else:
        self._target_policy = self._q_target_policy

  def initial_action(self):
     action = np.random.randint(self._number_of_actions)
     self._last_action = action
     return self._last_action
  
  @property
  def q_values(self):
    if self._double:
      return (self._q + self._q2)/2
    else:
      return self._q

  def _behaviour_policy(self,q):
      return self.epsilon_greedy(q)

  def _sarsa_target_policy(self,q, a):
      return np.eye(len(q))[a]

  def _q_target_policy(self, q, a):
      #print(len(q))
      #print(q)
      #print(np.argmax(a))
      return np.eye(len(q))[np.argmax(q)]

  def epsilon_greedy(self, q_values):
    if self._eps < np.random.random():
      return np.argmax(q_values)
    else:
      return np.random.randint(self._number_of_actions)

  def behaviour(self,state):
     if not self._double:
         return self._behaviour_policy(self._q[state])
     else:
         return self._behaviour_policy(self._q[state]+self._q2[state])

  def done(self, state):
     self._last_action = self.behaviour(state)
     self._s = state

  @abstractmethod
  def step(self, r, g, s):
    pass


class ExpTabAgent(TabularAgent):

  def __init__(self, number_of_states, number_of_actions, initial_state,
               SARSA= True, double= False, model= None, eps=0.1, step_size=0.1
               , num_offline_updates=0):

    super().__init__(number_of_states, number_of_actions, initial_state,
               SARSA, double, eps, step_size)

    self._num_offline_updates = num_offline_updates
    self._replay = []
    self.has_model = bool(model)
    
    if self.has_model:
        self._model = model(number_of_states, number_of_actions)


  def step(self, reward, discount, next_state):
    s = self._s
    a = self._last_action
    r = reward
    g = discount
    next_s = next_state
    self._replay.append((s,a,r,g,next_s))
    
    if self.has_model:
       self._model.update(s,a,r,g,next_s)

    if not self._double:

        a_next = self._behaviour_policy(self.q_values[next_s,:])
        act_targ_prob = self._target_policy(self._q[next_s], a_next)
        q_target = np.sum(act_targ_prob*self._q[next_s,:])
        self._q[s,a] += self._step_size*  (r + g*q_target - self._q[s,a])

        for i in range(self._num_offline_updates):
            ind = np.random.randint(0,len(self._replay))
            
            if self.has_model:
                s,a,_,_,_ = self._replay[ind]
                r,g,next_s = self._model.transition(s,a)
                next_s = int(next_s)
            else:
                s,a,r,g,next_s = self._replay[ind]
      
            a_next = self._behaviour_policy(self.q_values[next_s,:])
            act_targ_prob = self._target_policy(self._q[next_s], a_next)
            q_target = np.sum(act_targ_prob*self._q[next_s,:])
            self._q[s,a] += self._step_size*  (r + g*q_target - self._q[s,a])

        #make sure we are taking the correct action from experience not memory
        a_next = self._behaviour_policy(self.q_values[next_state,:])
    else:
        a_next = self._behaviour_policy(self._q[next_s]+self._q2[next_s])
        
        act_targ_prob = self._target_policy(self._q[next_s], a_next)
        act_targ_prob2 = self._target_policy(self._q2[next_s], a_next)

        q_target = np.sum(act_targ_prob*self._q2[next_s,:])
        q2_target = np.sum(act_targ_prob2*self._q[next_s,:])

        if np.random.uniform() <0.5:
            self._q[s,a] += self._step_size* (r + g*q_target - self._q[s,a])

        else:
            self._q2[s,a] += self._step_size*(r + g*q2_target - self._q2[s,a])

        for i in range(self._num_offline_updates):
            ind = np.random.randint(0,len(self._replay))

            if self.has_model:
                s,a,_,_,_ = self._replay[ind]
                r,g,next_s = self._model.transition(s,a)
                next_s = int(next_s)
            else:
                s,a,r,g,next_s = self._replay[ind]

            a_next = self._behaviour_policy(self._q[next_s]+self._q2[next_s])
            act_targ_prob = self._target_policy(self._q[next_s], a_next)
            act_targ_prob2 = self._target_policy(self._q2[next_s], a_next)

            q_target = np.sum(act_targ_prob*self._q2[next_s,:])
            q2_target = np.sum(act_targ_prob2*self._q[next_s,:])

            if np.random.uniform() <0.5:
               self._q[s,a] += self._step_size* (r + g*q_target - self._q[s,a])
            else:
                self._q2[s,a] += self._step_size*(r + g*q2_target - self._q2[s,a])

        #again use real experience not from memory to output next action
        a_next = self._behaviour_policy(self._q[next_state]+self._q2[next_state])

    self._s = next_state
    self._last_action = a_next #next_actio
    return self._last_action


