[TOC]

# The difficulty of Dota 2

- **long time horizons** : 每秒30帧，每局游戏大概45分钟，每4帧决策一次，一个episode大概20000个step。
- **Cooperative and Competitive POMDP** : 5 vs 5, 既有合作又有竞争，并且是部分观测的MDP。
- **High-dimensional action and observation spaces** ：观测维度16000维，离散化动作空间，共8000到80000个actions。

# Model Architecture

<center>
    <img src="./model.png">
    <br>
    <div>
        simplified model architecture
    </div>
</center>

## Observation Space

## Action Space

## Reward Weights

## Neural Network Architecture

# System Architecture

<center>
    <img src = "system.png">
    <br>
    <div>
        system architecture
    </div>
</center>

## reaction time

<center>
    <img src="reaction time.png">
    <div>
        reaction time
    </div>
</center>

- 每4帧决策一次，设决策时刻为$T$, 做出的动作将在$T+1$时刻直行，即$T$时刻执行$T-1$时刻的动作
- 每个动作中有一个维度输出将在接下来的4帧中哪一帧执行此动作。这样延迟在5帧到8帧之间（167ms到267ms)

## opponent sampling 

- 80% 的游戏对手为当前最新版本的参数，20%选择过去的老版本
- 每个过去的对手有一个评估其水平的得分$q_i, i=1,2,\dots,n$, 根据$p_i \sim \exp{(q_i)}$ 采样对手，每10个iteration将当前的智能体加入对手池，得分初始化为最大值。如果当前rollout的game中，当前智能体获胜，则$q_i \leftarrow q_i - \frac{\eta}{Np_i}$ ,若智能体失败，则不更新。

## training algorithm

- **rollout data generation** : Actor 每30秒收集120个time step(每秒4个)的数据送到learner的memory buffer, 然后每一分钟向controller中索要最新的模型参数。
- learner每2秒更新一次参数，这样的话，actor的参数实际上落后learner的参数32个PPO gradient step。

### Analysis of Indepent PPO

设$O_{0:t}$表示$0,1,2,\dots,t$时刻的观测，$O_{0:t}^{i}$表示第$i$个智能体的0到$t$时刻的观测，$a_t^{i}$表示第$i$个智能体$t$时刻的动作，$r_t^i$表示第$i$个智能体$t$时刻的奖励，$\pi(a_t^1,a_t^2,\dots,a_t^5|O_{0:t}) \stackrel{def}{=} \prod_{i=1}^5\pi(a_t^i|O_{0:t}^i)$ ,则对第i个agent, PPO 优化目标为：
$$
\begin{aligned}
&E_{a_t,O_{0:t} \sim \pi_{\theta_0}}\left[\frac{\prod_{i=1}^5\pi_\theta(a_t^i|O_{0:t}^i)}{\prod_{i=1}^5\pi_{\theta_0}(a_t^i|O_{0:t}^i)}\cdot(Q^i_{\theta_0}(O_{0:t},a_t)-V^i_{\theta_0}(O_{0:t}))\right]\\
&\approx E_{a_t^i,O_{0:t} \sim \pi_{\theta_0}}\left[\frac{\pi_\theta(a_t^i|O_{0:t}^i)}{\pi_{\theta_0}(a_t^i|O_{0:t}^i)}\cdot(Q^i_{\theta_0}(O_{0:t},a_t^i)-V^i_{\theta_0}(O_{0:t}))\right]
\end{aligned}
$$












  








