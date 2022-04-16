import matplotlib.pyplot as plt 
import os

def summarize_plot(data, args):
    # 1. plot undiscounted reward
    plt.plot(data["avg_reward"])
    plt.title("Undiscounted cumulative reward of training with eq %s in %s environment" %(args.pg_eq, args.env_name))
    plt.xlabel("Episode")
    plt.ylabel("Average Un-discounted Cumulative Reward")
    filename = os.path.join(os.getcwd(), 'assets', '%s_plot_%s_undiscounted.png' % (args.env_name, args.pg_eq)) 
    plt.savefig(filename)
    plt.close()

    # 2. plot undiscounted reward
    plt.plot(data["avg_discounted_reward"])
    plt.title("Discounted cumulative reward of training with eq %s in %s environment" %(args.pg_eq, args.env_name))
    plt.xlabel("Episode")
    plt.ylabel("Average Discounted Cumulative Reward")
    filename = os.path.join(os.getcwd(), 'assets', '%s_plot_%s_discounted.png' % (args.env_name, args.pg_eq)) 
    plt.savefig(filename)
    plt.close()