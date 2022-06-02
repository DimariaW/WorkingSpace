**Grandmaster level in StarCraft II using multi-agent reinforcement learning**

# StarCraft II

- two player zero-sum non-transmissive partial observed game

  [game introduction](https://arxiv.org/abs/1708.04782)

- camera view

  Humans play StarCraft through a screen that displays  only part of the map along with a high-level view of the entire map . The agent interacts with  the game through a similar camera-like interface, which naturally imposes an economy of attention, so that the agent chooses which area  it fully sees and interacts with. The agent can move the camera as an  action.

- APM limits

  Humans are physically limited in the number of actions  per minute (APM) they can execute. Our agent has a monitoring layer  that enforces APM limitations. This introduces an action economy that  requires actions to be prioritized. Agents are limited to executing at  most 22 non-duplicate actions per 5-s window.

- delays

  AlphaStar has two sources of delays. First, in real-time evaluation  (not training), AlphaStar has a delay of about 110 ms between when a  frame is observed and when an action is executed, owing to latency,  observation processing, and inference. Second, because agents  decide ahead of time when to observe next (on average 370 ms,  but possibly multiple seconds), they may react late to unexpected  situations. 

# Model Architecture

## Observation

## Action

# Algorithm

## Supervised learning

## Reinforcement learning

- First, we  initialize the policy parameters to the supervised policy and continually minimize the KL divergence between the supervised and current  policy.
- Second, we train the main agents with pseudo-rewards to  follow a strategy statistic z, which we randomly sample from human  data. These pseudo-rewards measure the edit distance between sampled and executed build orders, and the Hamming distance between  sampled and executed cumulative statistics.
- Each type of pseudo-reward is active (that is,  non-zero) with probability 25%, and separate value functions and losses  are computed for each pseudo-reward

## Multi-agent learning







