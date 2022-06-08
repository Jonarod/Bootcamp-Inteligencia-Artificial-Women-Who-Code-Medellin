import gym
import gzip
import numpy as np
import os
import neat
import pickle
import multiprocessing

env = gym.make('CartPole-v1')

def eval_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config) #Create net for genome with configs

    observation = env.reset()  #Inital observation
    fitness = 0  #Starting fitness of 0
    done = False
    while not done:
        #env.render()   #Render game

        infer = net.activate(observation)

        # If only 1 sigmoid output (between 0 and 1): round number with rint, then convert it to integer 0 or 1.
        action = int(np.rint(infer[0]))

        # If 2 outputs: 1 per possible action.
        # Argmax returns the index of the maximum value in an array. ex: [0, 27], argmax returns 1. ex2: [13, 0], returns 0. 
        # action = np.argmax(infer)

        observation, reward, done, info = env.step(action)  #Performs action
        fitness += reward
    return fitness


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config)


def run(config_file):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.StdOutReporter(True))
    #p.add_reporter(neat.Checkpointer(5))

    # Run for up to x generations.
    winner = p.run(eval_genomes, 100)
    # pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome)
    # winner = p.run(pe.evaluate)

    # show final stats
    print('\nBest genome:\n{!s}'.format(winner))

    #test winner
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    save_object(winner_net, "saved_model")
    # test_model(winner_net)  #Tests model 100 times and prints result


def test_model(winner):    
    scores = []
    for i in range(100):
        score = 0
        done = False
        observation = env.reset()
        while not done:
            #env.render()   #Render game      
            action = np.argmax(winner.activate(observation))

            observation, reward, done, info = env.step(action)

            score += reward
        scores.append(score)
        env.reset()

    print("Max score over 100 tries:")
    print(np.max(scores))


def save_object(obj, filename):
    with gzip.open(filename, 'w', compresslevel=5) as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward.toml')
    run(config_path)