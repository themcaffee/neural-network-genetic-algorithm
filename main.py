"""Entry point to evolving the neural network. Start here."""
import logging
from pprint import pprint

import numpy as np
from celery import Celery, group

from es import CMAES, OpenES, PEPG, SimpleGA
from network import Network
import os

# Setup logging.
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.DEBUG,
    filename='log.txt'
)

QUEUE_USERNAME = os.environ['QUEUE_USERNAME']
QUEUE_PASSWORD = os.environ['QUEUE_PASSWORD']
QUEUE_HOST = os.environ['QUEUE_HOST']
QUEUE_VHOST = os.environ['QUEUE_VHOST']

app = Celery('main', backend='rpc://', broker="amqp://{}:{}@{}:5672/{}".format(QUEUE_USERNAME, QUEUE_PASSWORD, QUEUE_HOST, QUEUE_VHOST))


def train_networks(nn_param_choices, nn_params, dataset):
    """Train each network.

    Args:
        networks (list): Current population of networks
        dataset (str): Dataset to use for training/evaluating
    """
    # Create a celery group of jobs to train these networks
    job = group(train_single_network.s(nn_param_choices, params, dataset) for params in nn_params)

    print("Starting distributed training jobs")
    # Wait for all results to return
    result = job.apply_async()
    results = result.join()
    print("Distributed training jobs finished for this generation")
    pprint(results)
    return results


@app.task
def train_single_network(nn_param_choices, parameters, dataset):
    network = Network(nn_param_choices)
    network.create_set(parameters)
    network.train(dataset)
    return -(1-network.accuracy)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def translate(value, max, min=0):
    """
    Translate a real number into a number between min and max
    :param value:
    :param max:
    :param min:
    :return:
    """
    # Make all numbers positive
    value = abs(value)
    # Constrain the value to the 0-1 range
    value = sigmoid(value)
    # Figure out how 'wide' each range is
    span = max - min
    # Convert the 0-1 range into a value in the right range.
    result = int(min + (value * span))
    return result


def get_nn_params(solutions, nn_param_choices):
    """
    Convert continuous numpy array of GENERATIONS x Num of params
    into an array of network parameters given the choices
    :param solutions: A set of real numbers
    :param nn_param_choices: The set of possible options
    :return:
    """
    nn_params = []
    for i in range(len(solutions)):
        current_param = {}
        current_parameter_index = 0
        for param_name in nn_param_choices:
            # Choose one possible parameter based on the values given
            choice_index = translate(solutions[i][current_parameter_index],
                                     max=len(nn_param_choices[param_name]) - 1)
            current_param[param_name] = nn_param_choices[param_name][choice_index]
            current_parameter_index += 1
        nn_params.append(current_param)
    return nn_params


def get_average_accuracy(fitness_list):
    """Get the average accuracy for a group of networks.

    Args:
        networks (list): List of networks

    Returns:
        float: The average accuracy of a population of networks.

    """
    total_accuracy = 0
    for network_accuracy in fitness_list:
        total_accuracy += network_accuracy

    return total_accuracy / len(fitness_list)


def generate(generations, nn_param_choices, dataset, solver):
    """Generate a network with the genetic algorithm.

    Args:
        generations (int): Number of times to evole the population
        population (int): Number of networks in each generation
        nn_param_choices (dict): Parameter choices for networks
        dataset (str): Dataset to use for training/evaluating

    """
    history = []
    nn_params = []

    # Evolve the generation.
    for i in range(generations):
        logging.info("***Doing generation %d of %d***" %
                     (i + 1, generations))

        # Get the set of parameters for this generation
        nn_params = get_nn_params(solver.ask(), nn_param_choices)

        # Train the networks of this generation
        fitness_list = train_networks(nn_param_choices, nn_params, dataset)

        # Save the results
        solver.tell(fitness_list)
        result = solver.result()
        history.append(result[1])

        logging.info("fitness at iteration {} {}".format(str((i + 1)), str(result[1])))

        # Get the average accuracy for this generation.
        average_accuracy = get_average_accuracy(fitness_list)

        # Print out the average accuracy each generation.
        logging.info("Generation average: %.2f%%" % (average_accuracy * 100))
        logging.info('-' * 80)

    # Print out the best 5 networks
    # networks = sorted(networks, key=lambda x: x.accuracy, reverse=True)
    # print_networks(networks[:5])
    return history


def print_networks(networks):
    """Print a list of networks.

    Args:
        networks (list): The population of networks

    """
    logging.info('-' * 80)
    for network in networks:
        network.print_network()


def main():
    """Evolve a network."""
    generations = 10  # Number of times to evole the population.
    population = 20  # Number of networks in each generation.
    dataset = 'cifar10'
    # Possible options: simplega, cmaes, pepg, or oes
    genetic_algorithm_name = 'cmaes'

    nn_param_choices = {
        'nb_neurons': [64, 128, 256, 512, 768, 1024],
        'nb_layers': [1, 2, 3, 4],
        'activation': ['relu', 'elu', 'tanh', 'sigmoid'],
        'optimizer': ['rmsprop', 'adam', 'sgd', 'adagrad',
                      'adadelta', 'adamax', 'nadam'],
    }

    logging.info("***Evolving %d generations with population %d***" %
                 (generations, population))

    num_params = len(nn_param_choices)
    if genetic_algorithm_name == 'cmaes':
        # Choose the genetic algorithm to use
        genetic_algorithm = CMAES(num_params,
                                  popsize=population,
                                  weight_decay=0.0,
                                  sigma_init=0.5)
    elif genetic_algorithm_name == 'pepg':
        genetic_algorithm = PEPG(num_params,
                                 sigma_init=0.5,
                                 learning_rate=0.1,
                                 learning_rate_decay=1.0,
                                 popsize=population,
                                 average_baseline=False,
                                 weight_decay=0.00,
                                 rank_fitness=False,
                                 forget_best=False)
    elif genetic_algorithm_name == 'oes':
        genetic_algorithm = OpenES(num_params,
                                   sigma_init=0.5,
                                   sigma_decay=0.999,
                                   learning_rate=0.1,
                                   learning_rate_decay=1.0,
                                   popsize=population,
                                   antithetic=False,
                                   weight_decay=0.00,
                                   rank_fitness=False,
                                   forget_best=False)
    elif genetic_algorithm_name == 'simplega':
        genetic_algorithm = SimpleGA(num_params,
                                     sigma_init=0.5,
                                     popsize=population,
                                     elite_ratio=0.1,
                                     forget_best=False,
                                     weight_decay=0.00)
    else:
        raise Exception("Invalid genetic algorithm selected")

    history = generate(generations, nn_param_choices, dataset, genetic_algorithm)


if __name__ == '__main__':
    main()
