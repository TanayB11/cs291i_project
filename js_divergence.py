import numpy as np
from scipy.special import rel_entr
#from scipy.spatial.distance import hellinger
from stable_baselines3 import PPO
import gymnasium as gym

def js_divergence(P, Q):

    P = np.clip(P, 1e-12, 1.0)  # avoid log(0)
    Q = np.clip(Q, 1e-12, 1.0)  # avoid log(0)

    M = 0.5 * (P + Q) 
    kl_p_m = np.sum(rel_entr(P, M))  # KL(P || M)
    kl_q_m = np.sum(rel_entr(Q, M))  # KL(Q || M)
    #breakpoint()

    return 0.5 * (kl_p_m + kl_q_m)

def hellinger_distance(P, Q): # not using for now
    return np.sqrt(0.5 * np.sum((np.sqrt(P) - np.sqrt(Q)) ** 2))

def rollout_policies(env, P_agent, Q_agent, num_steps=100, seed=42, task='Humanoid-v5'): # P: unknown, Q: knnown
    env_p = gym.make(task, render_mode="rgb_array")
    env_q = gym.make(task, render_mode="rgb_array")
    
    obs_p, _ = env_p.reset(seed=seed)
    obs_q, _ = env_q.reset(seed=seed)

    jsd, hd = [], []

    for _ in range(num_steps):
        # ppo doesnt expose the action distribution, so we need to sample actions
        
        action_p, _ = P_agent.predict(obs_p, deterministic=False)
        action_q, _ = Q_agent.predict(obs_q, deterministic=False)

        jsd.append(js_divergence(action_p, action_q))
        #hd.append(hellinger_distance(p_actions, q_actions)) # ignore for now
        
        # step to next action
        action_p = P_agent.predict(obs_p, deterministic=False)[0]
        action_q = Q_agent.predict(obs_q, deterministic=False)[0]
        
        obs_p, _, done_p, truncated_p, _ = env_p.step(action_p)
        obs_q, _, done_q, truncated_q, _ = env_q.step(action_q)
        
        if done_p or truncated_p or done_q or truncated_q:
            break
    
    # close the env
    env_p.close()
    env_q.close()
    
    return np.mean(jsd) # aggregate across all steps in rollout # can also return hd here 

def rollout_controller(env, P_agent, Q_agent, num_rollouts=5, num_steps=100, seed=42, task='Humanoid-v5'): 
    # num_rollouts: how many rollouts to do
    # we can set num_rollouts = 5?? for now 
    
    D = []
    for i in range(num_rollouts):
        div = rollout_policies(env, P_agent, Q_agent, num_steps=num_steps, seed=seed, task=task)
        D.append(div)
        print(f"Rollout {i+1}/{num_rollouts}: JS Divergence = {div}")
    
    return np.mean(D) # aggregate across all rollouts


# sample to test with:
# if __name__ == '__main__':
#     env = gym.make('Humanoid-v5', render_mode="rgb_array")
#     test_path = 'humanoid_ppo_evaluated_model.zip'
#     test_path2 = 'humanoid_pretrain_original2/checkpoints/humanoid_1000000_steps.zip'
#     P_agent = PPO.load(test_path)
#     Q_agent = PPO.load(test_path2)
    
#     print("Starting JS divergence calculation between two humanoid policies...")
#     mean_divergence = rollout_controller(env, P_agent, Q_agent)
#     print(f"Final mean JS divergence: {mean_divergence}")











