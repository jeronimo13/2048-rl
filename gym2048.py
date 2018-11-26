import random

import gym
from policy_gradient import PolicyGradient
import matplotlib.pyplot as plt
import numpy as np
import gym_2048

env = gym.make('game-2048-v0')
env = env.unwrapped

# Policy gradient has high variance, seed for reproducability
# env.seed(1)

print("env.action_space", env.action_space)
print("env.observation_space", env.observation_space)
# print("env.observation_space.high", env.observation_space.high)
# print("env.observation_space.low", env.observation_space.low)


RENDER_ENV = False
EPISODES = 500
rewards = []
RENDER_REWARD_MIN = 50


def create_traversal(self, vector):
    v_x = list(range(0, self.N))
    v_y = list(range(0, self.N))

    if vector['x'] == 1:
        v_x.reverse()
    elif vector['y'] == 1:
        v_y.reverse()

    return (v_y, v_x)

def create_vector(self, direction):
    if direction == 0:
        return {'x': 0, 'y': -1}
    elif direction == 1:
        return {'x': 1, 'y': 0}
    elif direction == 2:
        return {'x': 0, 'y': 1}
    else:
        return {'x': -1, 'y': 0}

def find_furthest(board, row, col, vector, N, merged):
        """ finds furthest cell interactable (empty or same value) """
        found = False
        val = board[row][col]
        i = row + vector['y']
        j = col + vector['x']
        while i >= 0 and i < N and j >= 0 and j < N:
            val_tmp = board[i][j]
            if merged[i][j] or (val_tmp != 0 and val_tmp != val):
                return (i - vector['y'], j - vector['x'])
            if val_tmp:
                return (i, j)

            i += vector['y']
            j += vector['x']

        return (i - vector['y'], j - vector['x'])

def moves_available(observation, merged):
    moves = [False]*4
    for direction in range(4):
        dir_vector = create_vector(direction)
        traversal_y, traversal_x = create_traversal(dir_vector)

        for row in traversal_y:
            for col in traversal_x:
                val = observation[row][col]

                if val:
                    n_row, n_col = find_furthest(observation, row, col, dir_vector)

                    if not ((n_row,n_col) == (row,col)):
                        n_val = observation[n_row][n_col]
                        if (val == n_val and not merged[n_row][n_col]) or (n_val == 0):
                            moves[direction] = True
    return moves


if __name__ == "__main__":


    # Load checkpoint
    load_path = "output/weights/game2048-v0-temp.ckpt"
    save_path = "output/weights/game2048-v0-temp.ckpt"

    PG = PolicyGradient(
        n_x = 16,
        n_y = env.action_space.n,
        learning_rate=0.01,
        reward_decay=0.95,
        load_path=load_path,
        save_path=save_path
    )


    for episode in range(EPISODES):

        observation = env.reset()
        episode_reward = 0

        prev_obs = None

        cnt = 0
        rnd_moves =0

        while True:
            cnt+=1
            # if RENDER_ENV: env.render('human')

            # 1. Choose an action based on observation
            obs = np.array([i for sublist in observation for i in sublist])

            action = PG.choose_action(obs, env.moves_available())
            #
            if prev_obs is not None and np.array_equal(obs,prev_obs):
              # action = random.choice(range(3))
              rnd_moves +=1
            prev_obs = obs

            # 2. Take action in the environment
            observation_, reward, done, info = env.step(action)

            # 3. Store transition for training
            PG.store_transition(obs, action, reward)


            if done:
                episode_rewards_sum = sum(PG.episode_rewards)
                rewards.append(episode_rewards_sum)
                max_reward_so_far = np.amax(rewards)

                print("==========================================")
                print("Episode: ", episode)
                print("Reward: ", episode_rewards_sum)
                print("Random moves: ", rnd_moves/cnt)
                print("Max reward so far: ", max_reward_so_far)
                cnt=0

                # 4. Train neural network
                discounted_episode_rewards_norm = PG.learn(episode==EPISODES-1)

                # Render env if we get to rewards minimum
                if max_reward_so_far > RENDER_REWARD_MIN: RENDER_ENV = True


                break

            # Save new observation
            observation = observation_