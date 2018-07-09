## 0.78 is solving the FrozenLake Game
import gym
import random
import numpy as np

learning_rate = 0.9
discount_factor = 0.8
random.seed(0)


def main():
    env = gym.make('FrozenLake-v0')
    env.seed(0)

    Q = np.zeros((env.observation_space.n, env.action_space.n))

    total = 0
    for i in range(1,5000):
        state = env.reset()
        while True:
            noise = np.random.random((1, env.action_space.n)) / float(i)**2
            action = np.argmax(Q[state] + noise)
            state2, reward, done, _ = env.step(action)
            Qtarget = reward + discount_factor * np.max(Q[state2,:])
            Q[state, action] = learning_rate * Q[state, action] + (1- learning_rate) * Qtarget
            total += reward
            state = state2
            if done:
                break
    print(total / 5000.)

if __name__ == '__main__':
    main()
