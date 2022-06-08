# ╔═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
# ║                                               		IMPORT LIBRARIES	                                                    ║
# ╚═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
# import tensorflow.keras.backend.tensorflow_backend as tfback # Does NOT work on M1 Macs
import tensorflow as tf
# from tensorflow.python.framework.ops import disable_eager_execution
# disable_eager_execution()
import os
import imageio
from collections import deque
import matplotlib.pyplot as plt
import numpy as np
import gym
from skimage.transform import resize
from tensorflow.keras.layers import Dense, Activation, Conv2D, Flatten
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam
import math
import time
from datetime import timedelta

start_time = time.time()

# ╔═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
# ║                                               		GLOBAL VARIABLES	                                                    ║
# ╚═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                                                  CONSTANTS (tweak if needed)	                                                   

ENV_NAME	 			= 'PongNoFrameskip-v4'				# Valid ATARI Game. For example: PongNoFrameskip-v4, BreakoutDeterministic-v4, MsPacmanNoFrameskip-v4, KungFuMasterNoFrameskip-v4

PATH 					= "output/"  							# Where to save the model and gif files
os.makedirs(PATH, exist_ok=True)

LOAD_CHECKPOINT 		= False 			# In case you want to save a model file and then train it more later. 
LOAD_FILE 				= 'atari_model.h5' 	# Filename 

EVALUATION_FREQUENCY 	= 50				# Evaluate model once every ... games
SAVE_FREQUENCY 			= 50				# Save model once every ... games
MODEL_COPY_FREQUENCY 	= 3					# Copy model once every ... games

MAX_MEMORY_LENGTH		= 15000 			# Size of the memory in frames. Reduce this if you get out of RAM errors
BATCH_SIZE			 	= 32 				# How many memory samples to train on

NUM_GAMES 				= 2000 				# Reduce this if it takes too long to train, increase it if you need more time to learn. 

FRAME_HEIGHT 			= 80 				# Smaller frames reduce training time but could lose info
FRAME_WIDTH 			= 80 				# Especially for smaller items like knives or dots 

INITIAL_EXP 			= 15000 			# Number of random actions to start out with
EPS_MIN 				= 0.05 				# Mimimum chance to take a random action
EPS_DEC 				= 0.00001 			# Amount to decrease random chance every frame

ALPHA 					= 0.0001 			# Learning rate -  how fast network parameters are updated. Increasing this will let you learn faster, but you may 'overshoot'
GAMMA 					= 0.99 				# Future reward discount



# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                                                  VARIABLES (DO NOT TOUCH)	                                                   

eps 			= 1.0 		# Chance to take a random action
n_frames 		= 0 		# The cumulative number of frames the agent has seen
evaluation 		= False 	# Don't change here, it will
frames_for_gif 	= []		# Array to store gif frames



# ╔═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
# ║                                               			SET GPUs	                                                    	║
# ╚═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
#  NO need with M1 Mac
# def _get_available_gpus(): 	# We define this function so recent versions of keras and tf will work together
#   # global _LOCAL_DEVICES
#   if tfback._LOCAL_DEVICES is None:
#     devices = tf.config.list_logical_devices()
#     tfback._LOCAL_DEVICES = [x.name for x in devices]
#   return [x for x in tfback._LOCAL_DEVICES if 'device:gpu' in x.lower()]


# tfback._get_available_gpus = _get_available_gpus 	# Assign GPUs to TF





# ╔═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
# ║                                               EXPLORATION vs EXPLOITATION                                                   ║
# ║ Say there are 100 restaurants in your town. You've been to the 10 nearby, and really like 2 or 3. When you finally get that ║
# ║ date night, do you hit up one that you know will be good, or do you venture out to try something new?  If you never try     ║
# ║ something new, there is a 90% chance that you will miss the best restaurant out there.                                      ║
# ║ Our agent has a similar issue. Should it try the move it thinks is best, or try a random move with the chance to come up    ║
# ║ with a better strategy?                                                                                                     ║
# ║         																													║
# ║ The following code will tell it which to do - When we start out, we are going to do only random actions. After some amount  ║
# ║ of time, we are going to slowly decrease the amount of random actions we take. We will reach a plateau, though, because we  ║
# ║ always want to be taking at least a few random chances.     																║
# ║                                                                                   											║
# ║ Of course, during evaulation mode, we want the agent to only do what it thinks is best, so we will never take random        ║
# ║ actions when we are testing.        																						║
# ╚═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
def ExplorerExploiter():
  # Never take a random action during evaluation games
  if evaluation: 
    return 'Exploit'

  # Take only random actions during initial exploration phase
  if n_frames < INITIAL_EXP:
    return 'Explore'

  # Explore with chance eps
  if np.random.random() < eps:
    return 'Explore'

  # Otherwise exploit
  return 'Exploit'


# ╔═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
# ║ The main idea behind deep Q learning is to create a neural network that takes the pixels of the game board in as input and  ║
# ║ outputs the best action. The final output layer has one node for each possible action - the value of the node for the input ║
# ║ pixels is what value we expect to get from taking that action (ie, fire, jump, turn left) in the given state (input of all 	║
# ║ pixels on the screen). When training, we continually adjust the network so that the weights are close to the actual value 	║
# ║ received for taking a move in a given state. Then, for new states, we pick the move that network says will give us the 		║
# ║ highest reward + future reward.																								║
# ║ 																															║
# ║ We could train on each frame or game, but it's much more effecient to create a Memory that has a large number of previous 	║
# ║ frames and their results. When training, we simply sample from that memory.													║
# ║ 																															║
# ║ Each 'memory' consists of the state (pixels), what action we took, what reward we got, what the next state was, and whether ║
# ║ or not the game ended. If you are running the notebook and crash due to a memory error, you should still be able to get 	║
# ║ good results with a smaller memory.																							║
# ╚═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
class ExperienceMemory(object):
  def __init__(self, max_memory_length):
      self.state_memory = deque(maxlen=max_memory_length)
      self.action_memory = deque(maxlen=max_memory_length)
      self.reward_memory = deque(maxlen=max_memory_length)
      self.next_state_memory = deque(maxlen=max_memory_length)
      self.done_memory = deque(maxlen=max_memory_length)
      
  def get_length(self):
      return len(self.state_memory)

  # Store a new memory    
  def store_transition(self, state, action, reward, next_state, done):
      self.state_memory.append(state)
      self.action_memory.append(action)
      self.reward_memory.append(reward)
      self.next_state_memory.append(next_state)
      self.done_memory.append(done)

  # Get out batch_size samples from the memory
  def sample_buffer(self, batch_size):
    states = []
    actions = []
    rewards = []
    next_states = []
    dones = []
  
    for i in range(batch_size):
      sample_id  = np.random.randint(len(self.state_memory))
      states.append(self.state_memory[sample_id])
      actions.append(self.action_memory[sample_id])
      rewards.append(self.reward_memory[sample_id])
      next_states.append(self.next_state_memory[sample_id])
      dones.append(self.done_memory[sample_id])

    return np.asarray(states), np.asarray(actions), np.asarray(rewards), np.asarray(next_states), np.asarray(dones)




# ╔═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
# ║                                                     ATARI WRAPPERS                                                          ║
# ║ The Atari environment outputs 'raw' frames, so we need to apply these wrappers to format the frames to be what our agent    ║
# ║ should see.                                                                                                                 ║
# ║                                                                                                                             ║
# ║ The Skip wrapper helps us pool the result from 4 frames, since sometimes the atari environments can skil                    ║
# ║                                                                                                                             ║
# ║ The PreProcessFrame wrapper converts the frame to grayscale, while the move image channels wrapper gets the channels        ║
# ║ into a format our DQN can use.                                                                                              ║
# ║                                                                                                                             ║
# ║ The Normalize Frame wrapper puts all pixel values to between 0 and 1, which helps with the math. We could also clip the     ║
# ║ reward - this helps on some games as well.                                                                                  ║
# ║                                                                                                                             ║
# ║ The FrameStacker stacks the last 4 frames together - this is important because our agent needs to understand motion.        ║
# ║ With only one image, how could it tell which direction a ball was going? With 2 it could tell direction and speed,          ║ 
# ║ but not acceleration. Most implementations use 4 frames, because it is an even number and it gives some better historical   ║
# ║ information about the movement of objects.                                                                                  ║
# ╚═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

class SkipEnv(gym.Wrapper):
    def __init__(self, env=None, skip=4):
        super(SkipEnv, self).__init__(env)
        self._skip = skip

    def step(self, action):
        t_reward = 0.0
        done = False
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            t_reward += reward
            if done:
                break
         # When we are in evaluation mode, we need to send the deepest wrapped 
        # (original) observation back out so that we can see how the agent plays
        if evaluation == True:
          frames_for_gif.append(np.asarray(obs))
        return obs, t_reward, done, info

    def reset(self):
        self._obs_buffer = []
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs


class PreProcessFrame(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(PreProcessFrame, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(FRAME_HEIGHT,FRAME_WIDTH,1), dtype=np.uint8)

    def observation(self, obs):
        new_frame = np.reshape(obs, obs.shape).astype(np.float32)
        # convert to grayscale
        new_frame = tf.image.rgb_to_grayscale(new_frame)
        # scale to frame height and width
        new_frame = tf.image.resize(new_frame,[FRAME_HEIGHT, FRAME_WIDTH],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        # convert to numpy array
        new_frame = np.asarray(new_frame)
        return new_frame.astype(np.float32)


class MoveImgChannel(gym.ObservationWrapper):
    def __init__(self, env):
        super(MoveImgChannel, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0,
                            shape=(self.observation_space.shape[-1],
                                   self.observation_space.shape[0],
                                   self.observation_space.shape[1]),
                            dtype=np.float32)
  
    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)


class NormalizeFrame(gym.ObservationWrapper):
    # The match is easier if everything is normalized to be betwee 0 and 1.
    def observation(self, obs):
        return np.array(obs).astype(np.float32) / 255.0


class FrameStacker(gym.ObservationWrapper):
    def __init__(self, env, n_steps):
        super(FrameStacker, self).__init__(env)
        self.observation_space = gym.spaces.Box(
                             env.observation_space.low.repeat(n_steps, axis=0),
                             env.observation_space.high.repeat(n_steps, axis=0),
                             dtype=np.float32)

    def reset(self):
        self.buffer = np.zeros_like(self.observation_space.low, dtype=np.float32)
        return self.observation(self.env.reset())

    def observation(self, observation):
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation
        return self.buffer


# Apply all the wrappers on top of each other.
def make_env(env_name):
    env = gym.make(env_name)
    env = SkipEnv(env)
    env = PreProcessFrame(env)
    env = MoveImgChannel(env)
    env = FrameStacker(env, 4)
    return NormalizeFrame(env)




# Here is the code to generate and save the animated gifs.
def generate_gif(frame_number, frames_for_gif, reward):
    for idx, frame_idx in enumerate(frames_for_gif):
        frames_for_gif[idx] = resize(frame_idx, (420, 320, 3), preserve_range=True, order=0).astype(np.uint8)

    imageio.mimsave(f'{PATH}{"{0}_game_{1}_reward_{2}.gif".format(ENV_NAME, frame_number, reward)}', frames_for_gif, duration=1/30)



# ╔═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
# ║                                                     	AGENT		                                                        ║
# ║ Our Agent class is the key player. The agent has 2 neural networks: 														║
# ║ - the Q network used for predicting the best action to take in each state													║
# ║ - the target network, used for predicting the value of that action															║
# ║ 																															║
# ║ When there is only one network training is difficult because it's chasing after itself. The q network's values are 			║
# ║ periodically copied over to the target network. 																			║
# ╚═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
class Agent(object):
    def __init__(self, alpha, n_actions, input_dims):
        self.action_space = [i for i in range(n_actions)]
        self.gamma = GAMMA
        self.batch_size = BATCH_SIZE
        self.memory= ExperienceMemory(MAX_MEMORY_LENGTH)
        self.q_network = self.build_dqn(alpha, n_actions, input_dims)
        self.target_network = self.build_dqn(alpha, n_actions, input_dims)

    # This is the same architecture used by deep mind
    def build_dqn(self, lr, n_actions, input_dims):
        model = Sequential()
        model.add(Conv2D(filters=32, kernel_size=8, strides=4, activation='relu', input_shape=(*input_dims,), data_format='channels_first'))
        model.add(Conv2D(filters=64, kernel_size=4, strides=2, activation='relu', data_format='channels_first'))
        model.add(Conv2D(filters=64, kernel_size=3, strides=1, activation='relu', data_format='channels_first'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(n_actions))

        model.compile(optimizer=Adam(lr=lr), loss='mean_squared_error')
        model.summary() # Print network details
        return model

    def store_transition(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    # The agent will make use of our ExplorerExploiter to choose either
    # Random actions or the action that gives the highest Q value
    def choose_action(self, observation):
        
        if ExplorerExploiter() == 'Explore': 
            action = np.random.choice(self.action_space)
        else:
            state = np.array([observation], copy=False, dtype=np.float32)
            actions = self.q_network.predict(state)
            action = np.argmax(actions)

        return action
    
    def learn(self):
      # First of all, make sure we have enough memories to train on.
        if self.memory.get_length() > self.batch_size:
            # Get a batch of memories. Each is an array of 32 memories
            state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)

            #Predict both the values we thought we could get. Use the q
            # network for the state and the target network for the next state
            q_eval = self.q_network.predict(state)
            q_next = self.target_network.predict(new_state)

            q_target = q_eval[:]

            indices = np.arange(self.batch_size)
            # Dones is 0 or 1, so this acts as a mask so that when the episode
            # is done, we will only take the reward
            # When it is not done, we will take the best value reward of the
            # next state times the future discount

            q_target[indices, action] = reward + self.gamma*np.max(q_next, axis=1)*(1 - done)
            # finally, train the network to backpropogate the loss
            self.q_network.train_on_batch(state, q_target)

    def save_models(self, custom_name):
        print('... saving model ...')
        self.q_network.save('atari_model_{}_{}.h5'.format(ENV_NAME, custom_name))
    
    # Restore the model and copy parameters to target network
    def load_models(self, file):
        print('... loading model ...')
        self.q_network = load_model(file)
        self.target_network.set_weights(self.q_network.get_weights())


# ╔═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
# ║                                                        MAIN LOOP                                                            ║
# ║ There are two nested loops here: 																							║
# ║ - one for each frame of each game																							║
# ║ - and then the outer loop runs for the specified number of games															║
# ║ 																															║
# ║ Finally, plot out the scores over time. Most agents follow a roughly logrithmic pattern, where they don't learn any more 	║
# ║ after a certain time.																										║
# ╚═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

def main():

    global n_frames
    global evaluation
    global frames_for_gif
    global eps

    # Make our environment
    env = make_env(ENV_NAME)

    # Make our agent
    agent = Agent(alpha=ALPHA, input_dims=(4,FRAME_HEIGHT,FRAME_WIDTH), n_actions=env.action_space.n)   

    # Set the worst possible best score
    best_score = - math.inf 

    if LOAD_CHECKPOINT:
        agent.load_models(LOAD_FILE)

    scores  = []

    try:

        # Main training loop 
        for i in range(NUM_GAMES):

            # Check if it's time to play an evaluation game, save models, or copy network parameters   
            if i != 0 and i % EVALUATION_FREQUENCY == 0:
                evaluation = True
            else: 
                evaluation = False

            if i % MODEL_COPY_FREQUENCY == 0:
                agent.target_network.set_weights(agent.q_network.get_weights())

            # Reset parameters for each game   
            done = False
            observation = env.reset()
            score = 0

            # Loop that runs for each frame of a game
            while not done:
                # Pick which action to take
                action = agent.choose_action(observation)

                # Receive the results - next observation (frame), reward, done, and 
                # info (not used here)
                observation_, reward, done, info = env.step(action)
                n_frames += 1
                score += reward

                # See if it's time to update epsilon
                if eps > EPS_MIN and n_frames > INITIAL_EXP:
                    eps = eps - EPS_DEC

                # Store the memory
                agent.store_transition(observation, action, reward, observation_, int(done))
                # Train on one batch
                agent.learn() 
                # Increment observation
                observation = observation_


            # When game is over, add to scores
            scores.append(score)

            # If it was an evaluation game, record gif
            if (i!= 0 and i % EVALUATION_FREQUENCY == 0):
                generate_gif(i, frames_for_gif, score)
                frames_for_gif = []

            # Update average scores
            avg_score = np.mean(scores[-50:])

            print('Game: ', i, '/', NUM_GAMES, ' - Score: ', score, ' - Average score: %.3f' % avg_score, ' - Epsilon: %.2f' % eps, ' - Total frames:', n_frames, ' - Time:', str(timedelta(seconds=time.time() - start_time)))

            # Print a message. Many of these over time is an indication of learning.
            if avg_score > best_score:
                print('Go you! - last 50 games avg score %.2f better than best 50 games avg %.2f. ' % (avg_score, best_score))
                best_score = avg_score


            # Save the weights ?
            if (i!= 0 and i % SAVE_FREQUENCY == 0):
                agent.save_models('game_' + str(i) + '_avg_score_' + str(avg_score) )


        # Plot the final results    
        env.close()
        plt.ylabel('Score')
        plt.xlabel('Game Number')
        plt.plot(scores)


    except KeyboardInterrupt: # If we manually stop the script, save last checkpoint first
        agent.save_models('game_interrupt_save')

        # Plot the final results    
        env.close()
        plt.ylabel('Score')
        plt.xlabel('Game Number')
        plt.plot(scores)


main()
