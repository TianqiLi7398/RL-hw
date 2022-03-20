## Usage of the main file

To use the `main.py`, please run the following command

`$ python3 main.py task dqn_type episode`

here 

1. `task` can be `train` - to train the DQN, `plot` - to plot the reward record in the traing and `video` - generate a video based on the DQN setup of a certain episode checkpoint

2. `dqn_type` can be 

| `dqn_type`  | type of DQN |
| :---        |  :---: |
| `dqn_with_memoryreplay`  | normal DQN with memory replay|
| `dqn_without_memoryreplay`  | DQN without memory replay |
| `dqn_uniform_behavior`      | DQN with memory replay, and uses uniform behavior to explore |

3. `episode`

the episode to stop training/check video& plot, `episode` = integer x 500 as you want.

## Installation of LunarLander

To install LanrLander in Ubuntu 18/20, please use

`$ pip install gym[box2d]`
