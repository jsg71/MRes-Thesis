import matplotlib.pyplot as plt
import numpy as np
from collections import namedtuple,  defaultdict

class RandomTD(object):
  """Implements TD - basically just policy evaluation using TD under a random set of actions"""
  def __init__(self, number_of_states, number_of_actions, initial_state, step_size=0.1):
    self._values = np.zeros(number_of_states)
    self._state = initial_state
    self._number_of_actions = number_of_actions
    self._step_size = step_size
    
  def get_values(self):
    return self._values

  def step(self, r, g, s):
    #TD update
    self._values[self._state] = self._values[self._state] + self._step_size \
                                *(r + g*self._values[s] - self._values[self._state])
    #update state
    self._state = s
    #return a random action
    next_action = np.random.randint(self._number_of_actions)
    return next_action

class GeneralQ(object):
  """Implementation of SARSA, Q-learning, Double Q-learning but now all subsumed under General Q (double Q)
     learning"""

  def __init__(self, number_of_states, number_of_actions, initial_state, 
               SARSA=True, double=False, step_size=0.1):
        
    self._q = np.zeros((number_of_states, number_of_actions))
    if double:
      self._q2 = np.zeros((number_of_states, number_of_actions))
    self._s = initial_state
    self._number_of_actions = number_of_actions
    self._step_size = step_size
    self._last_action = 0 


    if SARSA:
        self._target_policy = self.sarsa_target_policy
    else:
        self._target_policy = self.q_target_policy
        
    self._behaviour_policy = self.behaviour_policy
        
    #self._target_policy = target_policy
    self._double = double
    #self._last_action = 0
    
  def behaviour_policy(self,q):
      return self.epsilon_greedy(q, 0.1)
  
  def sarsa_target_policy(self,q, a):
      return np.eye(len(q))[a]
    
  def q_target_policy(self, q, a):
      return np.eye(len(q))[np.argmax(q)]
    
  @property
  def q_values(self):
    if self._double:
      return (self._q + self._q2)/2
    else:
      return self._q

  def step(self, r, g, s):  
    #store our last action and state
    a_old = self._last_action
    s_old = self._s
        
    if not self._double:
        
        #Work out the next behavioural action under the behavour policy
        act_behav_next = self._behaviour_policy(self._q[s])
        
        #Work out the probabilities of our actions under the target policy
        #In the case of Q-Learning this will be a max, under Sarsa it will
        #be our behaviour policy, hence this been passed in
        act_targ_prob = self._target_policy(self._q[s], act_behav_next)
        
        #For general q-learning take the expectation under our policy probabilities
        #under our q action values - Note that Sarsa and Q-learning are subsumed in
        #this also as long as the policy functions passed are implemented
        q_target = np.sum(act_targ_prob*self._q[s,:])

        #Update our Q-values - For general q this is an expectation
        #For SARSA it will be the actual q value of the behaviour taken eps-greedy
        #For Q-learning it will be the evaluation under the greedy policy
        self._q[s_old,a_old] = self._q[s_old,a_old] + self._step_size* \
                        (r + g*q_target - self._q[s_old,a_old])
        
    else:
        #Work out the next behavioural action under the behavour policy
        #but this time over both sets of q-values
        act_behav_next = self._behaviour_policy(self._q[s]+self._q2[s])
        
        #Returns target probabilities under the target policy for each
        #set of q-values
        act_targ_prob = self._target_policy(self._q[s], act_behav_next)
        act_targ_prob2 = self._target_policy(self._q2[s], act_behav_next)
        
        #For General Q learning we now take expectations - note that for double we
        #now evaluate under the other Q-action values - so the q-target is evaluated
        #under the q2 action values and vice versa
        q_target = np.sum(act_targ_prob*self._q2[s,:])
        q2_target = np.sum(act_targ_prob2*self._q[s,:])
        
        #Randomly switch between which network of q-action values we update
        if np.random.uniform() <0.5:
            self._q[s_old,a_old] = self._q[s_old,a_old] + self._step_size* \
                        (r + g*q_target - self._q[s_old,a_old])
            
        else:
            self._q2[s_old,a_old] = self._q2[s_old,a_old] + self._step_size* \
                        (r + g*q2_target - self._q2[s_old,a_old])
        
    #update our state
    self._s = s
    #update our action - behavioural action and return
    self._last_action = act_behav_next
    return self._last_action   

  def epsilon_greedy(self, q_values, epsilon):
    if epsilon < np.random.random():
      return np.argmax(q_values)
    else:
      #print(q_values)
      #print(np.array(q_values).shape[-1])
      return np.random.randint(np.array(q_values).shape[-1])


class ExperienceQ(object):
  def __init__(
    self, number_of_states, number_of_actions, initial_state, 
     num_offline_updates=0, step_size=0.1, eps=0.1):

    #Tabular form here - initialise q values, initial state and action
    #self._q  = defaultdict(lambda: np.zeros(number_of_actions))
    self._q = np.zeros((number_of_states, number_of_actions))
    self._s = initial_state
    self._last_action = 0
    
    self._number_of_actions = number_of_actions
    self._step_size = step_size
    
    #Experience replay will story a buffer of experiences and replay to update the q values
    #as well as the online learning bit
    self._num_offline_updates = num_offline_updates
    self._replay = []
    self._eps = eps

  @property
  def q_values(self):
    return self._q

  def behaviour_policy(self,q):
      return self.epsilon_greedy(q)

  def epsilon_greedy(self, q_values):
    if self._eps  < np.random.random():
      return np.argmax(q_values)
    else:
      #print(q_values)
      #print(np.array(q_values).shape[-1])
      return np.random.randint(np.array(q_values).shape[-1])

  def step(self, reward, discount, next_state):
    s = self._s
    a = self._last_action
    r = reward
    g = discount
    next_s = next_state
    
    #Online update of q values - note random policy but max evaluation for q values estimation
    self._q[s,a] = self._q[s,a] + self._step_size*(r + g*np.max(self._q[next_s,:]) - self._q[s,a])
    
    #add experience to replay buffer
    self._replay.append((s,a,r,g,next_s))

    #Repay memories from the buffer to update our q values more times
    #Not sure I have the setup for experience replay right for episodic?
    for i in range(self._num_offline_updates):
        
      #uniform sampling
      ind = np.random.randint(0,len(self._replay))
      s,a,r,g,next_s = self._replay[ind]
      
      #relay the sample and update q values
      self._q[s,a] = self._q[s,a] + self._step_size*(r + g*np.max(self._q[next_s,:]) - self._q[s,a])
    #take a random next action and update the state, action 
    explore = bool(np.random.random() < self._eps)
    if explore:
      a_next = np.random.randint(self._number_of_actions)
    else:
      a_next = np.argmax(self.q_values[next_s,:])
    next_action = np.random.randint(self._number_of_actions)
    #a_next = self.behaviour_policy(self.q_values[next_state,:])
    
    self._s = next_state
    self._last_action = a_next #next_actio
    return self._last_action

class DynaQ(object):
  def __init__(
    self, number_of_states, number_of_actions, initial_state, 
     model=None, num_offline_updates=0, step_size=0.1, eps=0.1):
    
    self._q = np.zeros((number_of_states, number_of_actions))
    self._s = initial_state
    #self.initial_action = 0
    self._last_action = 0 #self.initial_action
    
    self._number_of_actions = number_of_actions
    self._step_size = step_size
    
    self._num_offline_updates = num_offline_updates
    self._replay = []
   

    #Look at the experience replay parts as it doesnt seem to learn
    #Very similar to experience replay but now we are going to learn a model
    self._model = model(number_of_states, number_of_actions)
    self._eps = eps

  @property
  def q_values(self):
    return self._q

  def behaviour_policy(self,q):
      return self.epsilon_greedy(q)

    #quickly put in 11/06/2018 no test

  def behaviour(self,q):
      return self.epsilon_greedy(q)


  def epsilon_greedy(self, q_values):
    if self._eps < np.random.random():
      return np.argmax(q_values)
    else:
      return np.random.randint(np.array(q_values).shape[-1])

  def step(self, reward, discount, next_state):
    s = self._s
    a = self._last_action
    r = reward
    g = discount
    next_s = next_state
    
    #Update q value in an online fashion
    self._q[s,a] = self._q[s,a] + self._step_size*(r + g*np.max(self._q[next_s,:]) - self._q[s,a])
    #add experience to buffer
    self._replay.append((s,a,r,g,next_s))
    
    #Learning part of the model, learning from real online experience
    self._model.update(s, a, r, g, next_s)

    #Replay experiences
    for i in range(self._num_offline_updates):
      #Uniform random sampling of previous state actions
      #note unlike experience replay I do not want the r,g,next_s from memory
      ind = np.random.randint(0,len(self._replay))
      s,a,_,_,_ = self._replay[ind]
        
      #Use the learned model in a generative fashion to generate next r,g,next_s
      r,g, next_s = self._model.transition(s,a)
      next_s = int(next_s)

      #Update q values from model generated experiences
      self._q[s,a] = self._q[s,a] + self._step_size*(r + g*np.max(self._q[next_s,:]) - self._q[s,a])
    
    #take random action and update state, action
    #take a random next action and update state and action
    explore = bool(np.random.random() < self._eps)
    if explore:
      a_next = np.random.randint(self._number_of_actions)
    else:
        a_next = np.argmax(self.q_values[next_state,:])
    
    #a_next = self.behaviour_policy(self._q[next_state,:])

    self._s = next_state
    self._last_action = a_next #next_action

    return self._last_action


#The next agents inherit from the above but use feature approximation

class FeatureExperienceQ(ExperienceQ):

  def __init__(
      self, number_of_features, number_of_actions, *args, **kwargs):
    super(FeatureExperienceQ, self).__init__(
        number_of_actions=number_of_actions, *args, **kwargs)
    
    #Experience replay with linear function approximation, states are now
    #vectors of features
    self._number_of_features = number_of_features
    self._replay = []
    #q values are parameterised weights now - thus dimension features x actions
    self._Qtheta = np.zeros((self._number_of_features,self._number_of_actions))
    self._eps = kwargs['eps']


  def q(self, state):
    #extra careful about dimensions - return predicted q value for the state
    #using dot product with learned q values - note multiple actions so we return
    #a vector of q values for each action
    state = state.reshape(-1,1)
    return np.dot(state.T,self._Qtheta)

  def step(self, reward, discount, next_state):
    #Same careful setup with numpy dimensions
    s = self._s.reshape(-1,1)
    a = self._last_action
    r = reward
    g = discount
    next_s = next_state.reshape(-1,1)
    
    #Calculate the error for Q_theta our parameterised q for that action against max
    delta = r + g*np.max(np.dot(next_s.T,self._Qtheta)) - np.dot(s.T,self._Qtheta[:,a])
    
    #Update Q_theta parameters online from real experience
    self._Qtheta[:,a] = self._Qtheta[:,a] + (self._step_size*delta*s).reshape(-1)
    
    #Add experience to the replay buffer
    self._replay.append((s,a,r,g,next_s))
    #self._model.update(s, a, r, g, next_s, self._step_size)
    
    #Begin updating qtheta parameters from replay experiences
    for i in range(self._num_offline_updates):
      #Uniform random sample of prior experiences
      ind = np.random.randint(0,len(self._replay))
      #Grab actual previous experiences
      s,a,r,g,next_s = self._replay[ind]
       
      #Qtheta update - from prior experiences
      delta = r + g*np.max(np.dot(next_s.T,self._Qtheta)) - np.dot(s.T,self._Qtheta[:,a])
      self._Qtheta[:,a] = self._Qtheta[:,a] + (self._step_size*delta*s).reshape(-1)

    #take a random next action and update state and action
    explore = bool(np.random.random() < self._eps)
    if explore:
      a_next = np.random.randint(self._number_of_actions)
    else:
      a_next = np.argmax(self.q(next_state))
    
    #a_next = self.behaviour_policy(self.q(next_state))
    
    #next_action = np.random.randint(self._number_of_actions)
    self._s = next_state
    self._last_action = a_next #next_action
    return self._last_action

class FeatureDynaQ(DynaQ):

  def __init__(
      self, number_of_features, number_of_actions, *args, **kwargs):
    super(FeatureDynaQ, self).__init__(
        number_of_actions=number_of_actions, *args, **kwargs)
   
    #Similar to feature experience q but now we will be learning a linear
    #model along the way and using that to generate our imagined experiences
    self._number_of_features = number_of_features
    self._replay = []
    self._Qtheta = np.zeros((self._number_of_features,self._number_of_actions))
    model = kwargs['model']
    self._model = model(self._number_of_features, self._number_of_actions)
    self._eps = 0.1

  def q(self, state):
    state = state.reshape(-1,1)
    return np.dot(state.T,self._Qtheta)

  def step(self, reward, discount, next_state):
    s = self._s.reshape(-1,1)
    a = self._last_action
    r = reward
    g = discount
    next_s = next_state.reshape(-1,1)
    
    #Update qtheta parameters with real online experiences
    delta = r + g*np.max(np.dot(next_s.T,self._Qtheta)) - np.dot(s.T,self._Qtheta[:,a])
    self._Qtheta[:,a] = self._Qtheta[:,a] + (self._step_size*delta*s).reshape(-1)
    
    #update replay buffer
    self._replay.append((s,a,r,g,next_s))
    #Update linear model with real online experience
    self._model.update(s, a, r, g, next_s, self._step_size)
    
    for i in range(self._num_offline_updates):
      #Uniformly choose a prior experience
      ind = np.random.randint(0,len(self._replay))
      #Because we generate from our model unlike experience replay we wont
      #grab r,g,next_s
      s,a,_,_,_ = self._replay[ind]
       
      #given a state action, generate the rest of the experience from the learned model
      r,g,next_s = self._model.transition(s,a)
      r = float(r)
      g = float(g)
        
      #update qtheta parameters with model generated experiences
      delta = r + g*np.max(np.dot(next_s.T,self._Qtheta)) - np.dot(s.T,self._Qtheta[:,a])
      self._Qtheta[:,a] = self._Qtheta[:,a] + (self._step_size*delta*s).reshape(-1)

    #take an eps greedy next action

    explore = bool(np.random.random() < self._eps)
    if explore:
      a_next = np.random.randint(self._number_of_actions)
    else:
      a_next = np.argmax(self.q(next_state))

    #a_next = self.behaviour_policy(self.q(next_state))

    #next_action = np.random.randint(self._number_of_actions)
    self._s = next_state
    self._last_action = a_next #next_action
    return self._last_action

