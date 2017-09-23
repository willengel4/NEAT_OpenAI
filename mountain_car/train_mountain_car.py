import neat
import gym
import operator
import random

env = gym.make('MountainCar-v0')

def test_genome(network, render):
    observation = env.reset()
    maxFitness = -4
    for t in range(200):
        if render:
            env.render()
        inputs = (observation[0], observation[1])
        output = network.activate(inputs)
        action, maxValue = max(enumerate(output), key=operator.itemgetter(1))
        observation, reward, done, info = env.step(action)
        if observation[1] > maxFitness:
            maxFitness = observation[1]
        if done:
            break

    return maxFitness

def eval_genomes(genomes, config):
    maxFitness = -4
    bestNet = None
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        genome.fitness = test_genome(net, False)
        if genome.fitness > maxFitness:
            maxFitness = genome.fitness
            bestNet = net
    test_genome(bestNet, True)
    print('Max fitness: ', maxFitness)

config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, 
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        'config-feedforward')

p = neat.Population(config)
generationNum = 0

winner = p.run(eval_genomes, 300)
winnerNet = neat.nn.FeedForwardNetwork.create(winner, config)

test_genome(winnerNet, True)
