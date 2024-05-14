import os
import sys
import numpy as np 
import matplotlib.pyplot as plt 
import time 
import dill as pickle

""" 
Functions in this file:

train(env, agents, arglist): Training function for reinforcement learning agents. 
run_episode(agents, env, arglist): Runs an episode of the environment. Used in train.
iterate(obj): returns an iterable over an object that may be a list or a dictionary

env: environment to train 
agents: agents to train
arglist: training arguments (max_episode_len, num_episodes, save_rate, etc.) See run_particle_env
"""

def iterate(obj):
    """ Iterate over obj = dict || list. Return [(idx, key)]. 
    Agents will be indexed by idx, environment objects indexed by key."""
    pairs = []
    for idx, key in enumerate(obj.keys()) if isinstance(obj, dict) \
        else zip(range(len(obj)), range(len(obj))):
            pairs.append((idx,key))
    return pairs

def run_episode(agents, env, arglist):
    """ Runs an episode of the given environments """ 
    obs = env.reset() 
    n = len(agents) 
    total_rewards = 0.0 
    agent_rewards = [0.0 for _ in range(len(agents))] 

    done = False 
    train_steps = 0    

    for _ in range(arglist.max_episode_len):
        if arglist.display:
            env.render()
            time.sleep(0.15)
        
        (actions_e, actions_a) = env.get_action_set(agents, obs, arglist.method) 

        new_obs, rewards, done, info = env.step(actions_e)


        if arglist.method == 'train':
            for idx, key in iterate(actions_e):
                agents[idx].experience_callback(obs[key], actions_a[idx], new_obs[key], rewards[key], done[key])
    
        for idx, key in iterate(actions_e):
            agent_rewards[idx] += rewards[key] 
            total_rewards += rewards[key] 

        obs = new_obs 
        train_steps += 1

        terminal = False
        if type(done) == type(list):
            terminal = all(done)
        elif type(done) == type(dict):
            terminal = done['__all__']

        if terminal:
            break

    return total_rewards, agent_rewards, train_steps


def train(env, agents, arglist):
    """ TODO: what information do we want from training?"""
    episode_rewards = [0.0]
    agent_rewards = [[0.0] for _ in range(len(agents))]
    final_ep_rewards = []
    final_ag_ep_rewards = [[] for _ in range(len(agents))]

    episode_step = 1
    train_steps = 0
    
    print("Starting iterations...")
    t_time = time.time()
    for i in range(arglist.num_episodes):
        ep_results = run_episode(agents, env, arglist) 
        
        t_reward, a_rewards, t_steps = ep_results
        train_steps += t_steps

        episode_rewards[-1] += t_reward
        for (idx, a_reward) in enumerate(a_rewards):
            agent_rewards[idx][-1] += a_reward

        for agent in agents:
            agent.episode_callback()

        if len(episode_rewards) % arglist.save_rate == 0:

            final_ep_rewards.append(np.mean(episode_rewards[-arglist.save_rate:]))   
            for i, rew in enumerate(agent_rewards):
                final_ag_ep_rewards[i].append(np.mean(rew[-arglist.save_rate:]))     

            print("steps: {}, episodes: {}, mean episode reward:{}, time:{}".format(
                train_steps, len(episode_rewards), final_ep_rewards[-1], time.time() - t_time
            ))

            if arglist.save_path:
                with open(arglist.save_path, "wb") as fp:
                    pickle.dump(agents, fp)
            
            if arglist.train_result_path:
                save_obj = dict()
                
                save_obj["final_ep_rewards"] = final_ep_rewards 
                save_obj["final_ag_ep_rewards"] = final_ag_ep_rewards
                save_obj["all_ep_rewards"] = episode_rewards 
                save_obj["all_ag_rewards"] = agent_rewards 
                save_obj["arglist"] = arglist
                
                with open(arglist.train_result_path, "wb") as fp:
                    pickle.dump(save_obj, fp)

            t_time = time.time()
        episode_rewards.append(0)
        for (idx, a_reward) in enumerate(a_rewards):
            agent_rewards[idx].append(0)

        episode_step += 1
    
    print("Finished a total of {} episodes.".format(len(episode_rewards)))
    
    if arglist.save_path:
        print("Agent saved to {}.".format(arglist.save_path))

    if arglist.train_result_path:
        # final_ep_rewards, final_ag_rewards, episode_rewards, agent_rewards 
        print("Train results saved to {}.".format(arglist.train_result_path))
