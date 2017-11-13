"""Entry point to evolving the neural network. Start here."""
import logging

import numpy as np
from tqdm import tqdm

from es import CMAES
from network import Network

# Setup logging.
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.DEBUG,
    filename='log.txt'
)


def train_networks(networks, dataset):
    """Train each network.

    Args:
        networks (list): Current population of networks
        dataset (str): Dataset to use for training/evaluating
    """
    fitness_list = np.zeros(len(networks))
    pbar = tqdm(total=len(networks))
    for i in range(len(networks)):
        network = networks[i]
        network.train(dataset)
        fitness_list[i] = network.accuracy
        pbar.update(1)
    pbar.close()
    return fitness_list


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


def create_networks(nn_param_choices, nn_params):
    """
    Create all of the networks for a generation
    :param nn_param_choices:
    :param nn_params:
    :return:
    """
    networks = []
    for parameters in nn_params:
        new_network = Network(nn_param_choices)
        new_network.create_set(parameters)
        networks.append(new_network)
    return networks


def get_average_accuracy(networks):
    """Get the average accuracy for a group of networks.

    Args:
        networks (list): List of networks

    Returns:
        float: The average accuracy of a population of networks.

    """
    total_accuracy = 0
    for network in networks:
        total_accuracy += network.accuracy

    return total_accuracy / len(networks)


def generate(generations, nn_param_choices, dataset, solver):
    """Generate a network with the genetic algorithm.

    Args:
        generations (int): Number of times to evole the population
        population (int): Number of networks in each generation
        nn_param_choices (dict): Parameter choices for networks
        dataset (str): Dataset to use for training/evaluating

    """
    history = []
    result = []
    networks = []

    # Evolve the generation.
    for i in range(generations):
        logging.info("***Doing generation %d of %d***" %
                     (i + 1, generations))

        # Get the set of parameters for this generation
        nn_params = get_nn_params(solver.ask(), nn_param_choices)

        # Create the set of networks for this generation
        networks = create_networks(nn_param_choices, nn_params)

        # Train the networks of this generation
        fitness_list = train_networks(networks, dataset)

        # Save the results
        solver.tell(fitness_list)
        result = solver.result()
        history.append(result[1])

        logging.info("fitness at iteration", (i + 1), result[1])

        # Get the average accuracy for this generation.
        average_accuracy = get_average_accuracy(networks)

        # Print out the average accuracy each generation.
        logging.info("Generation average: %.2f%%" % (average_accuracy * 100))
        logging.info('-'*80)

    # Print out the best 5 networks
    networks = sorted(networks, key=lambda x: x.accuracy, reverse=True)
    print_networks(networks[:5])

    logging.info("local optimum discovered by solver: \n", result[0])
    logging.info("fitness score at this local optimum", result[1])
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
    generations = 2  # Number of times to evole the population.
    population = 10  # Number of networks in each generation.
    dataset = 'mnist'

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
    cmaes = CMAES(num_params,
                  popsize=population,
                  weight_decay=0.0,
                  sigma_init=0.5)

    history = generate(generations, nn_param_choices, dataset, cmaes)


if __name__ == '__main__':
    main()
