
import torch
from utils import to_device
import numpy as np


def cal_return(rewards, masks, gamma, pg_eq, device):
    # these computations would be performed on CPU
    rewards, masks = to_device(torch.device('cpu'), rewards, masks)
    # tensor_type = type(rewards)
    """ ESTIMATE RETURNS"""
    # tensor_type = type(rewards)
    # print(rewards.size())
    if pg_eq == 0:
        """ eq 1, sum R(tau) for each epsiode """
        rtau = torch.zeros(rewards.size(0), device=device)
        epsoide_reward = 0.0
        same_index = []
        for i in reversed(range(rewards.size(0))):
            if masks[i] == 0 and len(same_index) > 0:
                # find another epsiode, update previous R(tau)
                rtau[same_index] = epsoide_reward
                epsoide_reward = rewards[i]
                same_index = [i]
            else:
                # just accumulate reward

                epsoide_reward = rewards[i] + epsoide_reward * gamma
                same_index.append(i)
        rtau[same_index] = epsoide_reward
        returns = rtau
    elif pg_eq == 1:
        # this is wrong, since gt_gamma is the pointer of gt
        g_t = torch.zeros(rewards.size(0), device=device)
        """ eq 2, G(t) """
        gt = 0.0
        for i in reversed(range(rewards.size(0))):
            gt = rewards[i] + gamma * gt * masks[i]
            if i <=2:   print('i = %s, gt = %s' % (i, gt))
            gt_gamma = gt
            j = 1
            while True:
                # need to find the beginning of the traj
                if masks[i - j] == 0 or i < j:
                    break
                gt_gamma *= gamma
                if i <=2:   print('gt_gamma = %s, gt = %s' % (gt_gamma, gt))
                j += 1
            if i <=2:   print('j-1 = %s ' % (j-1))
            g_t[i] = gt_gamma
        returns = g_t
    else:
        raise RuntimeError("eq %s not defined" % pg_eq)

    # standardize returns for algorithmic stability
    # returns = (returns - returns.mean()) / returns.std()
    # to_device(returns)
    return returns

def estimate_net_grad(rewards, masks, states, actions, gamma, theta, device, pg_eq = 0, acc_obj = None):

    """ ESTIMATE RETURNS"""
    # returns = cal_return(rewards, masks, gamma, pg_eq)
    
    returns = (acc_obj - acc_obj.mean()) / acc_obj.std()
    # returns = torch.tensor(returns, device=device)
    # to_device(returns)

    """ ESTIMATE NET GRADIENT"""
    # https://stackoverflow.com/questions/53496570/matrix-multiplication-in-pytorch
    something = states * returns.view(states.size(0), 1)
    # print(something)
    log_probs = torch.mm(actions.T - torch.mm(theta, states.T), something)
    policy_loss = log_probs / (states.size(0))
    
    grad = policy_loss / (torch.norm(policy_loss) + 1e-8)

    # returns = to_device(device, grad)
    
    return grad

