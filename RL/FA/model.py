import matplotlib.pyplot as plt
import numpy as np
from collections import namedtuple


#Note my home environment is Python3 so I did slight trivial alterations to enable this
class TabularModel(object):

  def __init__(self, number_of_states, number_of_actions):
    self.number_of_states = number_of_states
    self.number_of_actions = number_of_actions
    
    #this is all deterministic - so learning simply means populating tables - thus for _next_state 
    #we can somewhat unusually use a single number rather than a probability dist over states etc.
    #However, there are different ways of doing this - taking an average will be slower to react to 
    #a changing environment than just taking the last deterministic values for example.
    
    self._next_state = np.zeros((self.number_of_states, self.number_of_actions))
    self._reward = np.zeros((self.number_of_states, self.number_of_actions))
    self._discount = np.zeros((self.number_of_states, self.number_of_actions))

  def next_state(self, s, a):
    #Self explanatory - returning next state, given a state and action
    return self._next_state[s,a]
  
  def reward(self, s, a):
    #Returning reward, given a state and action
    return self._reward[s,a]

  def discount(self, s, a):
    #Returning discount, given a state and action
    return self._discount[s,a]
  
  def transition(self, state, action):
    return (
        self.reward(state, action), 
        self.discount(state, action),
        self.next_state(state, action))
  
  def update(self, state, action, reward, discount, next_state):
    #Note the update here is the agent wondering around and populating a lookup table
    #an alternative would be actually learning or using an average...this would be slower 
    #to react to the non-stationary environment at the end
    self._next_state[state,action] = next_state
    self._reward[state,action] = reward
    self._discount[state,action] = discount

class LinearModel(object):

  def __init__(self, number_of_features, number_of_actions):
    #No states - we represent state by its exposure to features
    #Thus we now model a vector for reward and discount to enable a dot product with
    #the feature vector
    self.number_of_features = number_of_features
    self.number_of_actions = number_of_actions
    #We are doing q values so I will have a state transtion matrix for each action
    self._trans = np.zeros((number_of_features,number_of_features,number_of_actions))
    self._r = np.zeros((number_of_features,number_of_actions))
    self._g = np.zeros((number_of_features,number_of_actions))

  def next_state(self, s, a):
    #return the state transition matrix for the state and action
    return np.dot(self._trans[:,:,a],s)
  
  def reward(self, s, a):
    #return the expected reward for the state and action
    return np.dot(self._r[:,a].T,s)

  def discount(self, s, a):
    #return the learned discount for the state and action
    return np.dot(self._g[:,a].T,s)

  def transition(self, state, action):
    return (
        self.reward(state, action),
        self.discount(state, action),
        self.next_state(state, action))

  def update(self, state, action, reward, discount, next_state, step_size=0.1):
    #Update does stochastic gradient descent to learn the transition matrix, reward vector
    #and discount vector given experiences, feels a bit odd learning a discount vector, 
    #I would expect logically this is not really a part of the model environment, but anyway it works
    #Being extra careful with numpy and reshaping things...for the correct format...

    s = state.reshape(-1,1)
    a = action
    r = reward
    g = discount
    next_s = next_state.reshape(-1,1)

    #The learning bit...
    self._trans[:,:,a] = self._trans[:,:,a] + step_size*(next_s - np.dot(self._trans[:,:,a],s))*s.T
    self._r[:,a] = self._r[:,a] + (step_size*(r - np.dot(self._r[:,a].T,s))*s).reshape(-1)
    self._g[:,a] = self._g[:,a] + (step_size*(g - np.dot(self._g[:,a].T,s))*s).reshape(-1)


