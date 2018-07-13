import numpy as np
from lib.envs.market import Market
import math

def make_prices(mu, sigma, rf, utes, M, dt, X0, B0,S0, u_star, num_sims=300000):

    I = num_sims
    utility = np.zeros(utes)

    means = np.zeros(utes)
    variances = np.zeros(utes)

    #Will do a lot of runs and discretise
    S = np.zeros((M+1,I))
    X = np.zeros((M+1,I))
    NB = np.zeros((M+1,I))
    NS = np.zeros((M+1,I))
    dS = np.zeros((M,I))
    dB = np.zeros((M,I))
    dX = np.zeros((M,I))

    B = np.zeros(M+1)

    X[0] = X0
    B[0] = B0

    #Simulating lots of paths to the end
    np.random.seed(0)

    S[0] = S0
    for t in range(1, M+1):
        z = np.random.standard_normal(I)
        df = 10 
        z = np.random.standard_t(df,I)
        S[t] = S[t-1]*np.exp((mu-0.5*sigma**2)*dt + sigma*math.sqrt(dt)*z)
        B[t] = B[t-1]*np.exp(rf*dt)

    for t in range(1,M):
        dS[t] = S[t+1] - S[t]
        dB[t] = B[t+1] - B[t]

    for i in range(utes):
        for t in range(0, M):
            NB[t] = (1-u_star[i])*X[t]/B[t]
            NS[t] = u_star[i]*X[t]/S[t]
            dX[t] = NB[t]*dB[t] +NS[t]*dS[t]
            X[t+1] = X[t]+ dX[t]
    
        print(np.min(X[-1,:]))
        utility[i] = np.mean(np.log(X[-1,:]))
        means[i] = np.mean(X[-1,:])
        variances[i] = np.var(X[-1,:])
        
    return S, B, utility, means, variances


#We have now moved from utilities known at the end of an episode,
#to mean variance equivalent form - known at the end of the episode
#to a market environment where the agent receives incremental rewards at each step

#In order to test if this environment gives a good approximation to the original problem
#I will set constant wealth levels and may the utilities - so this is not an agent
#hopefully maximising the  step by step rewards should give a similar picture to the utilities above

#The min and max wealth levels above are helpful to know our range of wealth levels for our future states 
#for our RL agent
def run_stepsim_fixedprop(kappa, mu, sigma, rf,u_star, Market, starting_wealth=100.0, episodes=200000, time_periods=20):
    mean_utes = []
    mean_rewards = []
    all_wealth = []
    
    for prop in u_star:
        wealth = starting_wealth

        episodes = episodes
        time_periods = time_periods

        utilities = []
        rewards = []
        rsum = 0
        Mark = Market(kappa, episodes, time_periods, mu, rf, sigma)

        for i in range((episodes-1)*(time_periods-1)):
            #r, d, s, dx, done = M.step((action, wealth))
            r, d, s, dx, done = Mark.step((prop, wealth))

            if not done:
                rsum += r
                wealth += dx
            else:
                #print(wealth)
                utilities.append(np.log(wealth))
                rewards.append(rsum)
                all_wealth.append(wealth)
                
                rsum = 0
                wealth = starting_wealth

        print("Inv ratio: ", prop, "Mean log utility: ",np.mean(utilities))
        print("Inv ratio: ", prop, "Mean Reward: ", np.mean(rewards))
        print("Min wealth: ",np.min(all_wealth), "Max Wealth: ",np.max(all_wealth))

        mean_rewards.append(np.mean(rewards))
        mean_utes.append(np.mean(utilities))
        
    return mean_rewards, mean_utes, all_wealth

