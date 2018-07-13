import numpy as np
from lib.envs.market import Market
import torch


#Create a test bed...and examine if this set up is even learnable
#Need to create a testbed
#I will compare a random agent, to a merton optimal agent, to a trained agent on a new testset generated
#The agent knows nothing of utilities, so I will compare utilities at the end for the episodes

def execute_strat(kappa, mu, rf, sigma, utes,u_star,best_action, 
                  policy, q_values=None, episodes=300001, time_periods=20, wealth=100.0):
    #Execute learned policy
    # compare to random baseline or best
    #import torch
    #this needs a lot of tidying lots of hard coded dependencies on the above
    
    wealth = wealth

    time_periods = time_periods
    utilities_test = []
    rewards_test = []
    step_rewards = []
    rsum = 0
    wealth_episodes = []
    number_of_actions = utes

    #this needs tidying - specific for problem above
    start_state = int(wealth/10)
    state = start_state 
    
    Mark = Market(kappa, episodes, time_periods, mu, rf, sigma)

    for i_episode in range(episodes-1):
        state = start_state

        while True:
            #this is for q values using torch
            #random_values = Q[state] + torch.rand(1,number_of_actions) / 1000
            #action = torch.max(random_values,1)[1][0]
            #using my code???
            
            if policy=="Agent":
                #action = np.argmax(agent.q_values[state])
                action = np.argmax(q_values[state])

            elif policy=="Random":
                action = int(torch.LongTensor(1).random_(0, number_of_actions))
            elif policy=="Merton":
                action = best_action
            
            #if loading 
            #action = np.argmax(Q1[state])
            #print(action)

            #new_state, reward, done, info = env.step(action)
            prop = u_star[action]
            reward, d, new_state, dx, done = Mark.step((prop, wealth))
            #reward = np.log(dx)                
            #print("reward ",reward)
            #print("dx ",dx)
            #print("wealth", wealth)
            wealth += dx
            new_state = int(wealth/10)

            rsum += reward
            state = new_state
            
            step_rewards.append(reward)

            if done:
                state = start_state
                #print(rsum)
                #print(wealth)
                utilities_test.append(np.log(wealth))
                rewards_test.append(rsum)
                wealth_episodes.append(wealth)
                rsum = 0
                wealth = 100.0
                break
                
    return utilities_test, rewards_test, step_rewards, wealth_episodes
