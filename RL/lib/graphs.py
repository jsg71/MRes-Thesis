import math
import numpy
import numpy as np
import random
import decimal
import scipy.linalg
import numpy.random as nrand
import matplotlib.pyplot as plt
import torch
import seaborn as sns

#Now to implement q learning and variants on the above market environment
#import sys
#if "../" not in sys.path:
#  sys.path.append("../") 


#This perhaps illustrates where some difficulties are coming from
#the differences in summed rewards and utilities are incredibly small 

def make_baseline_graphs(utilities_test_rand,utilities_test_best,
                         rewards_test_rand,rewards_test_best,
                        step_rew_rand, step_rew_best):

    sns.distplot(utilities_test_rand, label='Random')
    #sns.distplot(utilities_test)
    sns.distplot(utilities_test_best, label='Merton')
    plt.title('Dist of end Utilities - Merton v Random')
    plt.xlabel('Utility')
    plt.ylabel('Likelihood')
    plt.legend()
    plt.show()
    
    #This perhaps illustrates where some difficulties are coming from

    sns.distplot(rewards_test_rand, label='Random')
    #sns.distplot(utilities_test)
    sns.distplot(rewards_test_best, label='Merton')
    plt.title('Dist of Summed rewards - Merton v Random')
    plt.xlabel('Episode sum of rewards')
    plt.ylabel('Likelihood')
    plt.legend()
    plt.show()
    
    #The agent has even less to go on...only incremental step by step rewards
    #the difference between rewards from knowing nothing and the best actions is below !

    sns.distplot(step_rew_rand, label='Random')
    #sns.distplot(utilities_test)
    sns.distplot(step_rew_best, label='Merton')
    plt.title('Dist of Step rewards - Merton v Random')
    plt.xlabel('Reward')
    plt.ylabel('Likelihood')
    plt.legend()
    plt.show()
    
    #Is this even learnable ? I will look at it a bit like a poker player...
    #Of the 3million episodes above, chop them into 3000 independent blocks of 1000
    #and examine the difference in distribution of merton v random for this problem
    block_utilities_test_rand = np.mean(np.array(utilities_test_rand).reshape(1000,-1),0)
    block_utilities_test_best = np.mean(np.array(utilities_test_best).reshape(1000,-1),0)
    
    sns.distplot(block_utilities_test_best, label="Merton optimal")
    #sns.distplot(block_utilities_test)
    sns.distplot(block_utilities_test_rand, label="Random agent")
    plt.title('Distribution of final utilities Merton v Random (grouped by 1000 episodes)')
    plt.xlabel('Utility')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()
    
    #Now for rewrds...

    #block_rew_test = np.mean(np.array(utilities_test).reshape(1000,-1),0)
    block_rew_test_rand = np.mean(np.array(rewards_test_rand).reshape(1000,-1),0)
    block_rew_test_best = np.mean(np.array(rewards_test_best).reshape(1000,-1),0)
    
    sns.distplot(block_rew_test_best, label="Merton optimal")
    #sns.distplot(block_utilities_test)
    sns.distplot(block_rew_test_rand, label="Random agent")
    plt.title('Distribution of Final rewards Merton v Random (per 1000 episodes)')
    plt.xlabel('Episode rewards')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

#Show how well the agent did compared to the above

def make_agent_graphs(rewards_test_best, rewards_test_rand, rewards_test, 
                      utilities_test_best, utilities_test_rand, utilities_test,
                     wealth_test_rand, wealth_test_best, wealth_test):
    
    block_utilities_test = np.mean(np.array(utilities_test).reshape(1000,-1),0)
    block_utilities_test_rand = np.mean(np.array(utilities_test_rand).reshape(1000,-1),0)
    block_utilities_test_best = np.mean(np.array(utilities_test_best).reshape(1000,-1),0)


    block_wealth_test_rand = np.mean(np.array(wealth_test_rand).reshape(1000,-1),0)
    block_wealth_test_best = np.mean(np.array(wealth_test_best).reshape(1000,-1),0)
    block_wealth_test = np.mean(np.array(wealth_test).reshape(1000,-1),0)

    
    block_rewards_test = np.mean(np.array(rewards_test).reshape(1000,-1),0)
    block_rewards_test_rand = np.mean(np.array(rewards_test_rand).reshape(1000,-1),0)
    block_rewards_test_best = np.mean(np.array(rewards_test_best).reshape(1000,-1),0)

    #Calculate sharpe ratios instead of longitudinally do it at the end blocks of 1000 again

    #mu - rf/ sigma
    wr = np.array(wealth_test_rand).reshape(300,-1)
    wr_sharpe = (wr.mean(axis=0)/100-1)/(wr.std(axis=0)/100)

    wt = np.array(wealth_test).reshape(300,-1)
    wt_sharpe = (wt.mean(axis=0)/100-1)/(wt.std(axis=0)/100)

    wb = np.array(wealth_test_best).reshape(300,-1)
    wb_sharpe = (wb.mean(axis=0)/100-1)/(wb.std(axis=0)/100)


    sns.distplot(block_rewards_test_best, label="Merton optimal")
    sns.distplot(block_rewards_test, label="Trained Agent")
    sns.distplot(block_rewards_test_rand, label="Random agent")
    plt.title('Distribution of Final rewards Merton v Trained Agent v Random (per 1000 episodes)')
    plt.xlabel('Episode Rewards')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

    plt.violinplot([block_rewards_test_rand, block_rewards_test, block_rewards_test_best])
    plt.xticks([1,2,3], ["Random Agent", "Trained Agent", "Merton Optimal"], rotation=60, size=12)
    plt.ylabel("Test Rewards", size=12)
    plt.xlabel("Agent", size=12)
    ax = plt.gca()
    ax.set_axis_bgcolor('white')
    ax.grid(0)
    plt.title("Violin plot of Rewards - Random v Merton v Agent")
    plt.show()


    sns.distplot(block_utilities_test_best, label="Merton optimal")
    sns.distplot(block_utilities_test, label="Trained Agent")
    sns.distplot(block_utilities_test_rand, label="Random agent")
    plt.title('Distribution of Final Utilities Merton v Trained Agent v Random (per 1000 episodes)')
    plt.xlabel('Episode Utilities')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()
    
    plt.violinplot([block_utilities_test_rand, block_utilities_test, block_utilities_test_best])
    plt.xticks([1,2,3], ["Random Agent", "Trained Agent", "Merton Optimal"], rotation=60, size=12)
    plt.ylabel("Test Utility", size=12)
    plt.xlabel("Agent", size=12)
    ax = plt.gca()
    ax.set_axis_bgcolor('white')
    ax.grid(0)
    plt.title("Violin plot of utilities - Random v Merton v Agent")
    plt.show()
    
    plt.plot(np.convolve(utilities_test_best, np.ones((10000,))/10000, mode='valid'), label='Merton Optimal')
    plt.plot(np.convolve(utilities_test_rand, np.ones((10000,))/10000, mode='valid'), label='Random agent')
    plt.plot(np.convolve(utilities_test, np.ones((10000,))/10000, mode='valid'), label='Trained agent')
    plt.title('Moving average 10,000 episode utilities')
    plt.ylabel('Utility')
    plt.xlabel('Episode')
    plt.legend()
    plt.show()

    plt.violinplot([block_wealth_test_rand, block_wealth_test, block_wealth_test_best])
    plt.xticks([1,2,3], ["Random Agent", "Trained Agent", "Merton Optimal"], rotation=60, size=12)
    plt.ylabel("Test Wealth", size=12)
    plt.xlabel("Agent", size=12)
    ax = plt.gca()
    ax.set_axis_bgcolor('white')
    ax.grid(0)
    plt.title("Violin plot of Wealth - Random v Merton v Agent")
    plt.show()


    sns.distplot(block_wealth_test_best, label="Merton optimal")
    sns.distplot(block_wealth_test, label="Trained Agent")
    sns.distplot(block_wealth_test_rand, label="Random agent")
    plt.title('Distribution of Final Wealth Merton v Trained Agent v Random (per 1000 episodes)')
    plt.xlabel('Episode Wealth')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

    plt.violinplot([wr_sharpe, wt_sharpe, wb_sharpe])
    plt.xticks([1,2,3], ["Random Agent", "Trained Agent", "Merton Optimal"], rotation=60, size=12)
    plt.ylabel("Test Sharpe", size=12)
    plt.xlabel("Agent", size=12)
    ax = plt.gca()
    ax.set_axis_bgcolor('white')
    ax.grid(0)
    plt.title("Violin plot of Sharpe ratios (M period) - Random v Merton v Agent")
    plt.show()

    sns.distplot(wb_sharpe, label="Merton optimal")
    sns.distplot(wt_sharpe, label="Trained Agent")
    sns.distplot(wr_sharpe, label="Random agent")
    plt.title('Distribution of Final Sharpe ratios  Merton v Trained Agent v Random (per 1000 episodes)')
    plt.xlabel('Episode Sharpe')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

    return block_utilities_test_rand, block_utilities_test, block_utilities_test_best, \
           block_rewards_test_rand, block_rewards_test, block_rewards_test_best, \
           block_wealth_test_rand, block_wealth_test, block_wealth_test_best, \
           wr_sharpe, wt_sharpe, wb_sharpe

def plot_sample_paths(S,mu,sigma):
    plt.plot(S[:,0:10])
    plt.title('Sample GBM paths mu '+str(mu)+', sigma '+str(sigma))
    plt.xlabel('Reblance period - time')
    plt.ylabel('Price level')

def plot_disc_utility(u_star,utility,gamma=None):
    plt.plot(u_star, utility)
    if gamma:
        plt.title("Simulated Discretised Power Utility ")
        plt.ylabel("Power Utility " + "Gamma=" +str(gamma))
    else:
        plt.title("Simulated Discretised Log Utility- constant investment")
        plt.ylabel("Log Utility")
    plt.xlabel("Investment Ratio > 1 borrows at risk free")

def plot_mv_equiv(u_star, vals):
    plt.plot(u_star, vals)
    plt.title("Mean Variance equivalent form - using kappa")
    plt.xlabel("Investment Ratio > 1 borrows at risk free")
    plt.ylabel("Mean-kappa/2*variance")
    #np.argmax(vals) #we see can can find the max

def plot_const_step(u_star,mean_rewards):
    plt.plot(u_star,mean_rewards)
    plt.title("Constant investment strategy - step by step rewards")
    plt.xlabel("Investment Ratio")
    plt.ylabel("Mean Reward")



