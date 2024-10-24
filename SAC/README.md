# Soft Actor-Critic

- [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/abs/1801.01290) propose **soft actor-critic**, an **off-policy** **actor-critic** deep RL algorithm based on the **maximum entropy** reinforcement learning framework. 

- [Soft Actor-Critic Algorithms and Applications](https://arxiv.org/abs/1812.05905)의 알고리즘을 따라 구현.



## Install

✨Tensorflow

*tensorflow와 tensorflow-probability 버전 호환 주의*
```
pip install tensorflow==2.12.0
pip install tensorflow-probability==0.20.0
```

✨
```
pip install matplotlib
pip install gym
pip install pygame
pip install mujoco==2.3.7
pip install imageio
```


## Test
- [Pendulum](https://www.gymlibrary.dev/environments/classic_control/pendulum/)
- [Half Cheetah](https://www.gymlibrary.dev/environments/mujoco/half_cheetah/)
- [Bipedal Walker](https://www.gymlibrary.dev/environments/box2d/bipedal_walker/)
