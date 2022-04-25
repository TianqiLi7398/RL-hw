""" This file provides fucntions of CartPole-v0 environment """

import torch
from utils import to_device

def estimate_advantages(rewards, masks, values, gamma, tau, device):
    """ this function returns the advantage value in batch samples
     """
    rewards, masks, values = to_device(torch.device('cpu'), rewards, masks, values)
    tensor_type = type(rewards)
    deltas = tensor_type(rewards.size(0), 1)
    advantages = tensor_type(rewards.size(0), 1)

    prev_value = 0
    prev_advantage = 0
    for i in reversed(range(rewards.size(0))):
        deltas[i] = rewards[i] + gamma * prev_value * masks[i] - values[i]
        advantages[i] = deltas[i] + gamma * tau * prev_advantage * masks[i]

        prev_value = values[i, 0]
        prev_advantage = advantages[i, 0]

    returns = values + advantages
    advantages = (advantages - advantages.mean()) / advantages.std()

    advantages, returns = to_device(device, advantages, returns)
    return advantages, returns

def pg_step(policy_net, optimizer_policy, states, actions, acc_obj):
    """ this function performs normal Actor-Critic update based on 
    customized critic value. """
    # normalize acc_obj value, which is the customized critic value
    returns = (acc_obj - acc_obj.mean()) / acc_obj.std()
    """update policy"""
    log_probs = policy_net.get_log_prob(states, actions)
    policy_loss = -(log_probs * returns).mean()
    optimizer_policy.zero_grad()
    policy_loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 40)
    optimizer_policy.step()

def a2c_step(policy_net, value_net, optimizer_policy, optimizer_value, states, actions, returns, advantages, l2_reg):
    """ this function performs Advantage Actor-Critic (A2C) updates"""
    """update critic"""
    values_pred = value_net(states)
    value_loss = (values_pred - returns).pow(2).mean()
    # weight decay
    for param in value_net.parameters():
        value_loss += param.pow(2).sum() * l2_reg
    optimizer_value.zero_grad()
    value_loss.backward()
    optimizer_value.step()
    
    """update policy"""
    log_probs = policy_net.get_log_prob(states, actions)
    policy_loss = -(log_probs * advantages).mean()
    optimizer_policy.zero_grad()
    policy_loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 40)
    optimizer_policy.step()