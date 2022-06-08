import gym
env = gym.make('CartPole-v1')
env.reset()

done = False
for _ in range(200):
    env.render()
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
env.close()