""" 
This document is the DQN practice hw for ECEN 689 RL
Author: Tianqi Li
Date: Mar 13, 2022
1 class is util

util.name_file: generats the prefix of files 
util.plot_sliding: plot the sliding window reward
 """

import matplotlib.pyplot as plt
import os, json

class util:

    @staticmethod
    def name_file(dqn_type):
        return 'LunarLander-v2_%s_nn_' % dqn_type

    @staticmethod
    def plot_sliding(episode, dqn_type):
        story = util.name_file(dqn_type)
        filename = os.path.join(os.getcwd(), 'model', '%s_scores_%s.json' % (story, episode)) 
        with open(filename) as json_file:
            data = json.load(json_file)
        
        # plot rewards
        plt.plot(data["scores"])
        plt.title("Reward plot")
        plt.xlabel("Iteration")
        plt.ylabel("Reward")
        filename = os.path.join(os.getcwd(), 'pics', '%sscore_plot_%s.png' % (story, episode)) 
        plt.savefig(filename)
        plt.close()

        # plot rewards
        plt.plot(data["scores_sliding"])
        plt.title("Reward window average plot with sliding window = %s" % data["sliding_size"])
        plt.xlabel("Iteration")
        plt.ylabel("Reward")
        filename = os.path.join(os.getcwd(), 'pics', '%sscore_sliding_plot_%s.png' % (story, episode)) 
        plt.savefig(filename)
        plt.close()

        return