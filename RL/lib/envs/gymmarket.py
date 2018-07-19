import numpy as np
import math

np.random.seed(0)
class Market(object):
  def __init__(self, kappa, episodes, time_periods, mu, r, sigma):

    self.episodes = episodes
    self.time_periods = time_periods
    self.epi = 0
    self.t = 0
    
    
    self.mu = mu
    self.r = r
    self.sigma = sigma
    
    self.S, self.B, self.dS, self.dB = self.generate_price_series()
    self._number_of_states = int(np.max(self.S)*100)+1
    
    self._start_state = self.S[self.t,self.epi]
    self._state = self._start_state
    self.kappa = kappa

  def reset():
    pass


    
  @property
  def number_of_states(self):
      return self._number_of_states

  def get_obs(self):
    s = self._state
    return int(100*s)

  def step(self, action):
    #an action is an order to the environment
    #to buy a certain proportion
    #the wealth is also provided
    prop, wealth = action
    
    discount = 0.99
    #kappa = 0.008 # 0.009 #check if this makes a difference
        
    if self.t < self.time_periods:
        NB = (1-prop)*wealth/self.B[self.t]
        NS = prop*wealth/self.S[self.t, self.epi]
        
        dX = NB*self.dB[self.t] +NS*self.dS[self.t,self.epi]
       
        mu_adjust = (1.0+self.r)**(1/self.time_periods)-1
        reward = dX - (self.kappa/2)*(dX**2)  #-mu_adjust*wealth)
       
        #reward = np.log(1+dX/wealth)  #try directly for log utility and see if any different
        done = False
        #print(self.t,self.epi,self.S[self.t,self.epi],action, dX, reward)
        
        self.t += 1
        new_state = self.S[self.t, self.epi]
        self._state = new_state 
    else:
        #reached the end of episode...needs tidying
        self.t = 0
        self.epi +=1
        new_state = self.S[self.t, self.epi]
        self._state = new_state
        reward = 0.0   
        dX = 0.0
        done = True
        
    return reward, discount, self.get_obs(), dX, done

  def generate_price_series(self):
    I = self.episodes
    M = self.time_periods
    
    S0 = 1
    B0 = 1
    T = 1.0
    dt = T/M
    
    mu = self.mu
    r = self.r
    sigma = self.sigma
    
    S = np.zeros((M+1,I))
    dS = np.zeros((M,I))
    dB = np.zeros(M)

    B = np.zeros(M+1)
    B[0] = B0
    S[0] = S0
    
    for t in range(1, M+1):
        z = np.random.standard_normal(I)
        df = 10
        z = np.random.standard_t(df,I)
        S[t] = S[t-1]*np.exp((mu-0.5*sigma**2)*dt + sigma*math.sqrt(dt)*z)
        B[t] = B[t-1]*np.exp(r*dt)

    for t in range(1,M):
        dS[t] = S[t+1] - S[t]
        dB[t] = B[t+1] - B[t]

    #utility[i] = np.mean(np.log(X[-1,:]))
    #means[i] = np.mean(X[-1,:])
    #variances[i] = np.var(X[-1,:])
    return S, B, dS, dB

