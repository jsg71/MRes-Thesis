#import matplotlib.pyplot as plt
import numpy as np
#from collections import namedtuple,  defaultdict
from abc import ABC, abstractmethod
from Tabular import TabularAgent, ExpTabAgent

class FeatureExp(ExpTabAgent):

  def __init__(
      self, number_of_features, number_of_actions, double = False, model=None, *args, **kwargs):
    super(FeatureExp, self).__init__(
        number_of_actions=number_of_actions, *args, **kwargs)

    #Experience replay with linear function approximation, states are now
    #vectors of features
    self._number_of_features = number_of_features
    self._replay = []
    #q values are parameterised weights now - thus dimension features x actions
    self._double = double
    if self._double:
        self._Q2theta = np.zeros((self._number_of_features, self._number_of_actions))

    self._Qtheta = np.zeros((self._number_of_features,self._number_of_actions))
    self.model =  model# kwargs['model']
    if self.model:
        self._model = self.model(self._number_of_features, self._number_of_actions)
    
  def behaviour(self,state):
      return self._behaviour_policy(np.squeeze(self.q(next_state)))

  def done(self, state):
      self._last_action = self.behaviour(state)
      self._s = state

  def q(self, state):
    #extra careful about dimensions - return predicted q value for the state
    #using dot product with learned q values - note multiple actions so we return
    #a vector of q values for each action
    state = state.reshape(-1,1)
    if self._double:
        return (np.dot(state.T, self._Qtheta) + np.dot(state.T, self._Q2theta))/2
    else:
        return np.dot(state.T,self._Qtheta)

  def step(self, reward, discount, next_state):
    #Same careful setup with numpy dimensions
    s = self._s.reshape(-1,1)
    a = self._last_action
    r = reward
    g = discount
    next_s = next_state.reshape(-1,1)

    self._replay.append((s,a,r,g,next_s))
    if self.model:
        self._model.update(s,a,r,g,next_s)


    if not self._double:
        a_next = self._behaviour_policy(np.squeeze(self.q(next_s)))
        act_targ_prob = self._target_policy(np.squeeze(self.q(next_s)), a_next)
        q_target = np.sum(act_targ_prob*np.squeeze(self.q(next_s)))

        delta = r + g*q_target - np.dot(s.T,self._Qtheta[:,a])
    
        #Calculate the error for Q_theta our parameterised q for that action against max
        #delta = r + g*np.max(np.dot(next_s.T,self._Qtheta)) - np.dot(s.T,self._Qtheta[:,a])

        #Update Q_theta parameters online from real experience
        self._Qtheta[:,a] = self._Qtheta[:,a] + (self._step_size*delta*s).reshape(-1)

        #Add experience to the replay buffer
        #self._replay.append((s,a,r,g,next_s))
        #if self.model:
        #    self._model.update(s, a, r, g, next_s) #, self._step_size)

        #Begin updating qtheta parameters from replay experiences
        for i in range(self._num_offline_updates):
            #Uniform random sample of prior experiences
            ind = np.random.randint(0,len(self._replay))
            #Grab actual previous experiences
            if self.model:
                s,a,_,_,_ = self._replay[ind]
                r,g,next_s = self._model.transition(s,a)
                r = float(r)
                g = float(g)
            else:
                s,a,r,g,next_s = self._replay[ind]

            #Qtheta update - from prior experiences
            a_next = self._behaviour_policy(np.squeeze(self.q(next_s)))
            act_targ_prob = self._target_policy(np.squeeze(self.q(next_s)), a_next)
            q_target = np.sum(act_targ_prob*np.squeeze(self.q(next_s)))

            delta = r + g*q_target - np.dot(s.T,self._Qtheta[:,a])
            #delta = r + g*np.max(np.dot(next_s.T,self._Qtheta)) - np.dot(s.T,self._Qtheta[:,a])
            self._Qtheta[:,a] = self._Qtheta[:,a] + (self._step_size*delta*s).reshape(-1)

        a_next = self._behaviour_policy(np.squeeze(self.q(next_state)))
        self._s = next_state
        self._last_action = a_next #next_action
    else:
        a_next = self._behaviour_policy(np.squeeze(self.q(next_s)))

        act_targ_prob = self._target_policy(np.squeeze(np.dot(next_s.T,self._Qtheta)), a_next)
        act_targ_prob2 = self._target_policy(np.squeeze(np.dot(next_s.T,self._Q2theta)), a_next)
        q_target = np.sum(act_targ_prob*np.squeeze(np.dot(next_s.T,self._Q2theta)))
        q2_target = np.sum(act_targ_prob2*np.squeeze(np.dot(next_s.T,self._Qtheta)))

        if np.random.uniform() <0.5:
            delta = r + g*q_target - np.dot(s.T,self._Qtheta[:,a])
            #delta = r + g*np.max(np.dot(next_s.T,self._Qtheta)) - np.dot(s.T,self._Qtheta[:,a])
            self._Qtheta[:,a] = self._Qtheta[:,a] + (self._step_size*delta*s).reshape(-1)
        else:
            delta = r + g*q2_target - np.dot(s.T,self._Q2theta[:,a])
            #delta = r + g*np.max(np.dot(next_s.T,self._Qtheta)) - np.dot(s.T,self._Qtheta[:,a])
            self._Q2theta[:,a] = self._Q2theta[:,a] + (self._step_size*delta*s).reshape(-1)
    

        #Begin updating qtheta parameters from replay experiences
        for i in range(self._num_offline_updates):
            #Uniform random sample of prior experiences
            ind = np.random.randint(0,len(self._replay))
            #Grab actual previous experiences
            if self.model:
                s,a,_,_,_ = self._replay[ind]
                r,g,next_s = self._model.transition(s,a)
                r = float(r)
                g = float(g)
            else:
                s,a,r,g,next_s = self._replay[ind]


            a_next = self._behaviour_policy(np.squeeze(self.q(next_s)))

            act_targ_prob = self._target_policy(np.squeeze(np.dot(next_s.T,self._Qtheta)), a_next)
            act_targ_prob2 = self._target_policy(np.squeeze(np.dot(next_s.T,self._Q2theta)), a_next)
            q_target = np.sum(act_targ_prob*np.squeeze(np.dot(next_s.T,self._Q2theta)))
            q2_target = np.sum(act_targ_prob2*np.squeeze(np.dot(next_s.T,self._Qtheta)))

            if np.random.uniform() <0.5:
                delta = r + g*q_target - np.dot(s.T,self._Qtheta[:,a])
                #delta = r + g*np.max(np.dot(next_s.T,self._Qtheta)) - np.dot(s.T,self._Qtheta[:,a])
                self._Qtheta[:,a] = self._Qtheta[:,a] + (self._step_size*delta*s).reshape(-1)
            else:
                delta = r + g*q2_target - np.dot(s.T,self._Q2theta[:,a])
                #delta = r + g*np.max(np.dot(next_s.T,self._Qtheta)) - np.dot(s.T,self._Qtheta[:,a])
                self._Q2theta[:,a] = self._Q2theta[:,a] + (self._step_size*delta*s).reshape(-1)

        a_next = self._behaviour_policy(np.squeeze(self.q(next_state)))
        #next_action = np.random.randint(self._number_of_actions)
        self._s = next_state
        self._last_action = a_next #next_action

    return self._last_action
