import argparse
import gym
import os
import pickle
import time

from utils import *
from utils.paint import summarize_plot
from agent import Agent, include_bias

from models.mlp_critic import Value
from models.mlp_policy_disc import DiscretePolicy
from solutions.gradient import estimate_advantages, a2c_step, pg_step
from solutions.point_mass_solutions import estimate_net_grad

parser = argparse.ArgumentParser(description='Pytorch Policy Gradient')
parser.add_argument('--env-name', default="Point-v0", metavar='G',
                    help='name of the environment to run')
parser.add_argument('--render', action='store_true', default=False,
                    help='render the environment')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--learning-rate', type=float, default=0.01, metavar='G',
                    help='gae (default: 3e-4)')
parser.add_argument('--num-threads', type=int, default=1, metavar='N',
                    help='number of threads for agent (default: 4)')
parser.add_argument('--seed', type=int, default=1, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--min-batch-size', type=int, default=2048, metavar='N',
                    help='minimal batch size per PPO update (default: 2048)')
parser.add_argument('--eval-batch-size', type=int, default=2048, metavar='N',
                    help='minimal batch size for evaluation (default: 2048)')
parser.add_argument('--max-iter-num', type=int, default=200, metavar='N',
                    help='maximal number of main iterations (default: 500)')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--save-model-interval', type=int, default=0, metavar='N',
                    help="interval between saving model (default: 0, means don't save)")
parser.add_argument('--gpu-index', type=int, default=0, metavar='N')
parser.add_argument('--pg-eq', type=int, default=0, metavar='N', 
                    help="equation number for policy gradient calculation")
parser.add_argument('--tau', type=float, default=0.95, metavar='G',
                    help='gae (default: 0.95)')
parser.add_argument('--l2-reg', type=float, default=1e-3, metavar='G',
                    help='l2 regularization regression (default: 1e-3)')
args = parser.parse_args()

"""environment"""
env = gym.make(args.env_name)
state_dim = env.observation_space.shape[0]
is_disc_action = len(env.action_space.shape) == 0
action_dim = 1 if is_disc_action else env.action_space.shape[0]
running_state = ZFilter((state_dim,), clip=5)


"""seeding"""
np.random.seed(args.seed)
torch.manual_seed(args.seed)
env.seed(args.seed)

"""cuda setting"""
dtype = torch.float64
torch.set_default_dtype(dtype)
device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')
if torch.cuda.is_available():
    print('using gpu')
    torch.cuda.set_device(args.gpu_index)


"""define actor and critic"""
if args.env_name == 'Point-v0':
    # we use only a linear policy for this environment
    theta = torch.normal(0, 0.01, size=(action_dim, state_dim + 1))
    policy_net = None
    # theta = theta.to(dtype).to(device)
    theta = theta.to(dtype).to(device)
    
    """create agent"""
    agent = Agent(env, args.env_name, device, policy_net, theta, custom_reward=None,
              running_state=None, num_threads=args.num_threads)
else:
    # we use both a policy and a critic network for this environment
    policy_net = DiscretePolicy(state_dim, env.action_space.n, hidden_size=(64, 16))
    theta = None
    value_net = Value(state_dim)
    policy_net.to(device)
    value_net.to(device)

    # Optimizers
    optimizer_policy = torch.optim.Adam(policy_net.parameters(), lr=args.learning_rate)
    optimizer_value = torch.optim.Adam(value_net.parameters(), lr=args.learning_rate)
    """create agent"""
    agent = Agent(env, args.env_name, device, policy_net, theta, custom_reward=None,
                  running_state=running_state, num_threads=args.num_threads, pg_eq = args.pg_eq,
                  gamma = args.gamma)


def update_params(batch, acc_obj, i_iter, theta = None):
    states = torch.from_numpy(np.stack(batch.state)).to(dtype).to(device)
    actions = torch.from_numpy(np.stack(batch.action)).to(dtype).to(device)
    rewards = torch.from_numpy(np.stack(batch.reward)).to(dtype).to(device)
    masks = torch.from_numpy(np.stack(batch.mask)).to(dtype).to(device)
    acc_obj = torch.tensor(acc_obj).to(dtype).to(device)
    if args.env_name == 'CartPole-v0':
        # Context-manager that disabled gradient calculation.
        """
        To implement CartPole on this env, you will need to implement the following functions:
        1. A function to compute returns, or reward to go, or advantages based on the question
            returns = estimate_returns(rewards, masks, args.gamma, device)
            rtg = estimate_rtg(rewards, masks, args.gamma, device)
            advantages, returns = estimate_advantages(rewards, masks, values, args.gamma, device)
        2. Use 1. to update the policy
            pg_step(policy_net, value_net, optimizer_policy, optimizer_value, states, actions, values_from_1)
        """

        if args.pg_eq == 2:
            """perform A2C update"""
            with torch.no_grad():
                values = value_net(states)
            advantages, returns = estimate_advantages(rewards, masks, values, args.gamma, 
                args.tau, device)
            a2c_step(policy_net, value_net, optimizer_policy, optimizer_value, states, actions, 
                returns, advantages, args.l2_reg)
        else:
            
            pg_step(policy_net, optimizer_policy, states, actions, acc_obj)

    if args.env_name == 'Point-v0':
        """get values estimates from the trajectories"""
       
        states_biased = include_bias(states)
        
        net_grad = estimate_net_grad(rewards, masks, states_biased, actions, args.gamma, theta, 
            device, pg_eq=args.pg_eq, acc_obj= acc_obj)

        """update policy parameters"""
        
        theta += net_grad * args.learning_rate



def main_loop():
    record_return = {"avg_reward": [], "avg_discounted_reward": []}
    for i_iter in range(args.max_iter_num):
        """generate multiple trajectories that reach the minimum batch_size"""
        batch, log, acc_obj = agent.collect_samples(args.min_batch_size, render=args.render)

        t0 = time.time()
        update_params(batch, acc_obj, i_iter, theta=theta)
        t1 = time.time()
        """evaluate with determinstic action (remove noise for exploration) For cartpole,set mean action to False"""
        _, log_eval, acc_obj = agent.collect_samples(args.eval_batch_size)
        t2 = time.time()
        record_return["avg_reward"].append(log_eval['avg_reward'])
        record_return["avg_discounted_reward"].append(log_eval['avg_discounted_reward'])

        if i_iter % args.log_interval == 0:
            print('{}\tT_sample {:.4f}\tT_update {:.4f}\ttrain_R_avg {:.2f}\teval_R_avg {:.2f}'.format(
                i_iter, log['sample_time'], t1-t0, log['avg_reward'], log_eval['avg_reward']))

        if args.save_model_interval > 0 and (i_iter+1) % args.save_model_interval == 0:
            if args.env_name == 'Point-v0':
                to_device(torch.device('cpu'), theta)
                
                pickle.dump((theta), open(os.path.join(os.getcwd(), 'data', args.env_name, 'learned_models_eq{}/{}_policy_grads.p'.format(args.pg_eq, args.env_name)), 'wb'))
                to_device(device, theta)
            else:
                to_device(torch.device('cpu'), policy_net, value_net)
                
                pickle.dump((policy_net, value_net), open(os.path.join(os.getcwd(), 'data', args.env_name, 'learned_models_eq{}/{}_policy_grads.p'.format(args.pg_eq, args.env_name)), 'wb'))
                to_device(device, policy_net, value_net)
        """clean up gpu memory"""
        torch.cuda.empty_cache()
    
    _, log_eval, acc_obj = agent.collect_samples(args.eval_batch_size, mean_action=False, render=True)
    # plot the result
    summarize_plot(record_return, args)

main_loop()