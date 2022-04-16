import multiprocessing
from typing_extensions import runtime
import gym
import gym.wrappers
from numpy import float64
from solutions.gradient import pg_step
from utils.replay_memory import Memory, Memory_new
from utils.torch import *
import math
import time
import os
os.environ["OMP_NUM_THREADS"] = "1"


def include_bias(x: torch.tensor) -> torch.tensor:
    # Add a constant term (1.0) to each entry in x
    return torch.cat([x, torch.ones_like(x[..., :1])], axis=-1)
    #return torch.cat(x,torch.ones(1))

def point_get_action(theta, ob):
    ob_1 = include_bias(ob)
    # test material
    # print(theta.squeeze(0).shape)
    # print(ob_1.view(-1,1).dtype)
    mean = torch.mm(theta,ob_1.view(-1,1)).view(-1,2).squeeze(0)
    return torch.normal(mean=mean, std=1.)

def collect_samples(pid, queue, env, env_name, policy, theta, custom_reward,
                    mean_action, render, running_state, min_batch_size, gamma, pg_eq):
    if pid > 0:
        torch.manual_seed(torch.randint(0, 5000, (1,)) * pid)
        if hasattr(env, 'np_random'):
            env.np_random.seed(env.np_random.randint(5000) * pid)
        if hasattr(env, 'env') and hasattr(env.env, 'np_random'):
            env.env.np_random.seed(env.env.np_random.randint(5000) * pid)
    log = dict()

    memory = Memory()
    num_steps = 0
    total_reward = 0
    total_discounted_reward = 0.0
    min_reward = 1e6
    max_reward = -1e6
    total_c_reward = 0
    min_c_reward = 1e6
    max_c_reward = -1e6
    num_episodes = 0
    acc_obj = []

    while num_steps < min_batch_size:
        state = env.reset()
        # print("dafault dtype = %s" % torch.get_default_dtype() )
        if running_state is not None:
            state = running_state(state)
        reward_episode = 0
        
        for t in range(10000):
            # interesting thing, if you'll find torch dafault dtype is float64, but
            # here state_var.dtype is float32, in repo 
            # https://github.com/Khrylx/PyTorch-RL/blob/master/examples/a2c_gym.py
            # it is float64...
            # print(state_var.dtype)
            with torch.no_grad():
                if env_name == 'CartPole-v0':
                    state_var = tensor(state, dtype=torch.float64).unsqueeze(0)
                    action = policy.select_action(state_var)[0].numpy()
                    action = int(action) if policy.is_disc_action else action.astype(np.float64)
                else:
                    state_var = tensor(state, dtype=torch.float32).unsqueeze(0)
                    # print(theta.get_device(), state_var.get_device())
                    action = point_get_action(theta, state_var).numpy()
                    action = action.astype(np.float64)

            next_state, reward, done, _ = env.step(action)
            reward_episode += reward
            if running_state is not None:
                next_state = running_state(next_state)

            if custom_reward is not None:
                reward = custom_reward(state, action)
                total_c_reward += reward
                min_c_reward = min(min_c_reward, reward)
                max_c_reward = max(max_c_reward, reward)

            mask = 0 if done else 1
            # if t == 2:
            #     mask = 0

            memory.push(state, action, mask, next_state, reward)
            if render:
                env.render()
            if done:
                break
            # if mask == 0: break

            state = next_state
        
        # calculate the acc_obj
        discount_reward_episode = 0.0
        for i in range(t+1):
            discount_reward_episode = discount_reward_episode * gamma + memory.memory[-1-i].reward
        if pg_eq == 0:
            # calculate R(tau)
            acc_obj_episode = [discount_reward_episode] * (t + 1)
            acc_obj += acc_obj_episode
        elif pg_eq == 1:
            acc_obj_episode = []
            # calculate G(t)
            gt = 0.0
            for i in range(t+1):
                gt = memory.memory[-1-i].reward + gamma * gt
                acc_obj_episode.insert(0, gt * (gamma ** (t - i)))
            # print(acc_obj_episode)
            acc_obj += acc_obj_episode
        # print(len(memory.memory), len(acc_obj_episode), t+1)
        # log stats
        num_steps += (t + 1)
        num_episodes += 1
        total_reward += reward_episode
        total_discounted_reward += discount_reward_episode
        min_reward = min(min_reward, reward_episode)
        max_reward = max(max_reward, reward_episode)
    # assert len(memory.memory) == len(acc_obj), RuntimeError("acc_obj does not match!")
    # why sum all rewards?????
    log['num_steps'] = num_steps
    log['num_episodes'] = num_episodes
    log['total_reward'] = total_reward
    log['avg_reward'] = total_reward / num_episodes
    log['total_discounted_reward'] = total_discounted_reward
    log['avg_discounted_reward'] = total_discounted_reward / num_episodes
    log['max_reward'] = max_reward
    log['min_reward'] = min_reward
    if custom_reward is not None:
        log['total_c_reward'] = total_c_reward
        log['avg_c_reward'] = total_c_reward / num_steps
        log['max_c_reward'] = max_c_reward
        log['min_c_reward'] = min_c_reward

    if queue is not None:
        queue.put([pid, memory, log, acc_obj])
    else:
        return memory, log, acc_obj

def merge_log(log_list):
    log = dict()
    log['total_reward'] = sum([x['total_reward'] for x in log_list])
    log['total_discounted_reward'] = sum([x['total_discounted_reward'] for x in log_list])
    log['num_episodes'] = sum([x['num_episodes'] for x in log_list])
    log['num_steps'] = sum([x['num_steps'] for x in log_list])
    log['avg_reward'] = log['total_reward'] / log['num_episodes']
    log['avg_discounted_reward'] = log['total_discounted_reward'] / log['num_episodes']
    log['max_reward'] = max([x['max_reward'] for x in log_list])
    log['min_reward'] = min([x['min_reward'] for x in log_list])
    if 'total_c_reward' in log_list[0]:
        log['total_c_reward'] = sum([x['total_c_reward'] for x in log_list])
        log['avg_c_reward'] = log['total_c_reward'] / log['num_steps']
        log['max_c_reward'] = max([x['max_c_reward'] for x in log_list])
        log['min_c_reward'] = min([x['min_c_reward'] for x in log_list])

    return log


class Agent:

    def __init__(self, env, env_name, device, policy, theta, custom_reward=None,
            running_state=None, num_threads=1, pg_eq = 0, gamma = 0.99):
        self.env = env
        self.policy = policy
        self.device = device
        self.custom_reward = custom_reward
        self.running_state = running_state
        self.num_threads = num_threads
        self.theta = theta
        self.env_name = env_name
        self.pg_eq = pg_eq                  # 0: R(tau), 1: G_t, 2: a2c
        self.gamma = gamma

    def collect_samples(self, min_batch_size, mean_action=False, render=False):
        t_start = time.time()
        # ?? move policy, theta from cuda to cpu, to run the cases
        local_theta = None
        if self.env_name == 'CartPole-v0':
            to_device(torch.device('cpu'), self.policy)
        elif self.env_name == 'Point-v0':
            ######## we should set theta to device as well?
            local_theta = self.theta.to('cpu').numpy()
            local_theta = torch.from_numpy(local_theta).float().unsqueeze(0).to('cpu').squeeze(0)
            # print(local_theta)
            # print("after change, %s" % self.theta.get_device())

        if render:
            env = gym.wrappers.Monitor(env=self.env, directory="./assets", force=True)
            # env = gym.make(self.env_name, render_mode='human')
            memory, log, acc_obj = collect_samples(0, None, env, self.env_name, self.policy, local_theta, self.custom_reward, mean_action,
                                      render, self.running_state, min_batch_size, self.gamma, self.pg_eq)
            batch = memory.sample()
            return batch, log, acc_obj
        else:

            thread_batch_size = int(math.floor(min_batch_size / self.num_threads))
            queue = multiprocessing.Queue()
            workers = []
            for i in range(self.num_threads-1):
                worker_args = (i+1, queue, self.env, self.env_name, self.policy, local_theta, self.custom_reward, mean_action,
                            False, self.running_state, thread_batch_size, self.gamma, self.pg_eq)
                workers.append(multiprocessing.Process(target=collect_samples, args=worker_args))
            for worker in workers:
                worker.start()

            memory, log, acc_obj = collect_samples(0, None, self.env, self.env_name, self.policy, local_theta, self.custom_reward, mean_action,
                                        render, self.running_state, thread_batch_size, self.gamma, self.pg_eq)
            
            worker_logs = [None] * len(workers)
            worker_memories = [None] * len(workers)
            worker_acc_objs = [None] * len(workers)
            for _ in workers:
                # print(queue.get())
                pid, worker_memory, worker_log, worker_acc_obj = queue.get()
                worker_memories[pid - 1] = worker_memory
                worker_logs[pid - 1] = worker_log
                worker_acc_objs[pid - 1] = worker_acc_obj
            # print(len(worker_memories))
            for worker_memory in worker_memories:
                memory.append(worker_memory)
            for worker_acc_obj in worker_acc_objs:
                acc_obj += worker_acc_obj
            # there is no random sample, but just zip all memory sections together
            # print(len(memory.memory), len(acc_obj))
            batch = memory.sample()

            if self.num_threads > 1:
                log_list = [log] + worker_logs
                log = merge_log(log_list)
            if self.env_name == 'CartPole-v0':
                to_device(self.device, self.policy)
            t_end = time.time()
            log['sample_time'] = t_end - t_start
            log['action_mean'] = np.mean(np.vstack(batch.action), axis=0)
            log['action_min'] = np.min(np.vstack(batch.action), axis=0)
            log['action_max'] = np.max(np.vstack(batch.action), axis=0)
            return batch, log, acc_obj