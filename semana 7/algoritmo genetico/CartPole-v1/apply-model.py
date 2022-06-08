import gym
import gzip
import neat
import os
import numpy as np
try:
    import cPickle as pickle # pylint: disable=import-error
except ImportError:
    import pickle # pylint: disable=import-error


def load_object(filename):
    with gzip.open(filename) as f:
        obj = pickle.load(f)
        return obj


def test_model_trained(env, saved_net, render=False):
    observation = env.reset()
    score = 0
    done = False
    for _ in range(500):
        if render:
            env.render()

        infer = saved_net.activate(observation)
        action = int(np.rint(infer[0]))
        # action = np.argmax(infer)

        observation, reward, done, info = env.step(action)
        score += reward

    print("Episode finished with score: {}".format(score))
    env.close()


def test_model_random(env, render=False):
    observation = env.reset()
    score = 0
    done = False
    for _ in range(500):
        if render:
            env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        score += reward
    print("Episode finished with score: {}".format(score))
    env.close()


if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    saved_model_path = os.path.join(local_dir, 'saved_model')
    winner_net = load_object(saved_model_path)

    env = gym.make('CartPole-v0')

    # test_model_random(env, True)

    test_model_trained(env, winner_net, True)
