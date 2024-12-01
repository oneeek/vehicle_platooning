# Soft Actor-Critic

- [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/abs/1801.01290) propose **soft actor-critic**, an **off-policy** **actor-critic** deep RL algorithm based on the **maximum entropy** reinforcement learning framework. 

- [Soft Actor-Critic Algorithms and Applications](https://arxiv.org/abs/1812.05905)의 알고리즘을 따라 구현.



## Install

*tensorflow와 tensorflow-probability 버전 호환 주의*
```
pip install tensorflow==2.12.0
pip install tensorflow-probability==0.20.0
```

```python
if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))

CONFIG_FILE = "D:/result/testbed/simpleConfig.sumocfg"
DETECTOR_FILE = "D:/result/testbed/simpleDet.det.xml"
SIMUL_STEP = 1
TRAFFIC_SCALE = 1

SUMO_CMD = [sumolib.checkBinary('sumo'), '-c', CONFIG_FILE, '--start',
            '--step-length', str(SIMUL_STEP), '--scale', str(TRAFFIC_SCALE), 
            '--additional-files', DETECTOR_FILE, '--quit-on-end']

ROUTE_FILE = "D:/result/testbed/simpleRou.rou.xml"
tree = ET.parse(ROUTE_FILE)
root = tree.getroot()

max_episode_num = 300
agent = SACagent()
reward_a, throughput_a = train(agent, max_episode_num)
```

```Python
import pandas as pd
import matplotlib.pyplot as plt

plt.plot(reward_a)
plt.show()
plt.plot(throughput_a)
plt.show()
```




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
