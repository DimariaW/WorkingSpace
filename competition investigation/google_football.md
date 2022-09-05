# WeKick

- **feature engineering**
  - 22 - (x,y) coordinates of left team players
  - 22 - (x,y) direction of left team players
  - 22 - (x,y) coordinates of right team players
  - 22 - (x, y) direction of right team players
  - 3 - (x, y and z) - ball position
  - 3 - ball direction
  - 3 - one hot encoding of ball ownership (noone, left, right)
  - 11 - one hot encoding of which player is active
  - 7 - one hot encoding of `game_mode`
  - **sticky actions**
  - **relative information**

- **reward shaping**

  make it **zero sum**

  - intercept & outside & offside: zero-sum, +0.2 for getting possession, -0.2 for losing possession.
  - slide: zero-sum, +0.2 for our successful slide, -0.2 for opponent successful slide. We found slide is useful for defense, so we added this reward during early training.
  - hold ball: zero-sum, +0.001 if team holds the ball('ball_owned_team' = 0), -0.001 if opponent holds the ball('ball_owned_team' = 1).
  - pass: +0.1 for each pass before a goal. This reward is added to scoring reward, because we just want to encourage passing which is helpful for the goal.

- **self-play** 

  1. initial opponent pool
     - reward shaping to train agent with specialization
     - GAIL to learn other team

  2. league train

# TamakEri (5th)

**the component of moments**

1. 
