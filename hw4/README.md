# Some instructions to run the code

## 1. Point-v0

Training command:

`$ python3 main.py --learning-rate 1e-1 --pg-eq 0 --save-model-interval 100 --num-threads 4`

## 2. CartPole-v0

Training command:

`$ python3 main.py --pg-eq 0 --save-model-interval 50 --env-name CartPole-v0 --learning-rate 1e-1`

`$ python3 main.py --pg-eq 1 --save-model-interval 50 --env-name CartPole-v0 --learning-rate 1e-1`

`$ python3 main.py --learning-rate 1e-1 --pg-eq 3 --save-model-interval 100 --env-name CartPole-v0`

# env

fun fact about gym: if want to use render, need to have specific gym env. 0.18.3