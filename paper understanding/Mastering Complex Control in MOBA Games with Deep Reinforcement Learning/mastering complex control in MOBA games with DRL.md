[TOC] 

# Mastering Complex Control in MOBA Games with Deep Reinforcement Learning

## Dual-clip PPO

- **motivation** : 对于asynchronous的actor-learner的架构来说，actor和learner存在不同步的问题，从而导致actor的behavior policy落后与learner的policy，导致训练数据off-policy的问题，对于PPO算法来说，PPO算法基于的TRPO的理论基础是：
  $$
  \begin{aligned}
  V_\pi(s_0)-V_\tilde\pi(s_0) &= E_{a_t,s_t \sim \pi}[Q_\tilde{\pi}(s_t,a_t)-V_\tilde{\pi}(s_t)]\\
  &\ge E_{a_t,s_t \sim \tilde{\pi}}[\frac{\pi(a_t|s_t)}{\tilde{\pi}(a_t|s_t)}Q_\tilde{\pi}(s_t,a_t)-V_\tilde{\pi}(s_t)]-C_1 \qquad if~max_s KL(\pi(\cdot|s)||\tilde{\pi}(\cdot|s)) < C_2
  \end{aligned}
  $$
  即要求$max_s KL(\pi(\cdot|s)||\tilde{\pi}(\cdot|s))$距离足够小，才成立。

  而在off-policy情况下，需要设计一个算法，若$KL(\pi||\tilde{\pi})$比较小，才能用$\tilde{\pi}$的数据更新$\pi$,不然不做任何更新。而PPO的目标函数为：
  $$
  \mathcal{L}^{\mathrm{CLIP}}(\theta)=\hat{\mathbb{E}}_{t}\left[\min \left(r_{t}(\theta) \hat{A}_{t}, \operatorname{clip}\left(r_{t}(\theta), 1-\epsilon, 1+\epsilon\right) \hat{A}_{t}\right)\right]
  $$
  不满足若$KL(\pi||\tilde{\pi})$比较小，才能用$\tilde{\pi}$的数据更新$\pi$,不然不做任何更新。

- **goal** : 若$KL(\pi||\tilde{\pi})$比较小，才能用$\tilde{\pi}$的数据更新$\pi$,不然不做任何更新

- **solution** : Dual-clip PPO
  $$
  \hat{\mathbb{E}}_{t}\left[\max \left(\min \left(r_{t}(\theta) \hat{A}_{t}, \operatorname{clip}\left(r_{t}(\theta), 1-\epsilon, 1+\epsilon\right) \hat{A}_{t}\right), c \hat{A}_{t}\right)\right]
  $$
  

