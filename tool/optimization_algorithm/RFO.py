
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.spatial import distance
from sklearn.metrics.pairwise import haversine_distances
import math
import random
from tool.optimization_algorithm._utilities import Solution, Data, initialize, sort_agents, display, compute_fitness, Conv_plot
from tool.optimization_algorithm._transfer_functions import get_trans_function

# LAHC Local Search #####################################################

tolerance = 0.02  # to set an upper limit for including a slightly worse particle in LAHC


def mutate(agent, num_features,
           muprob
           ):
    # muprob = 0.2
    numChange = int(num_features * muprob)
    pos = np.random.randint(0, num_features - 1, numChange)  # choose random positions to be mutated
    agent[pos] = 1 - agent[pos]  # mutation
    return agent


def LAHC(particle, train_X, val_X, train_Y, val_Y, weight_acc, num_features, classifier,
         # tolerance,
         muprob
         ):
    _lambda = 15  # upper limit on number of iterations in LAHC
    target_fitness = compute_fitness(particle, train_X, val_X, train_Y, val_Y, classifier,
                                     weight_acc=weight_acc)  # original fitness
    for i in range(_lambda):
        new_particle = mutate(particle, num_features,
                              muprob
                              )  # first mutation
        temp = compute_fitness(new_particle, train_X, val_X, train_Y, val_Y, classifier, weight_acc=weight_acc)
        if temp > target_fitness:
            particle = new_particle.copy()  # updation
            target_fitness = temp
        elif (temp >= (1 - tolerance) * target_fitness):
            temp_particle = new_particle.copy()
            for j in range(_lambda):
                temp_particle1 = mutate(temp_particle, num_features,
                                        muprob
                                        )  # second mutation
                temp_fitness = compute_fitness(temp_particle1, train_X, val_X, train_Y, val_Y, classifier, weight_acc=weight_acc)
                if temp_fitness > target_fitness:
                    target_fitness = temp_fitness
                    particle = temp_particle1.copy()  # updation
                    break
    return particle


def sign(x):
    if x > 0:
        return 1.0
    elif x < 0:
        return -1.0
    else:
        return 0.0


############################# RFO #############################
def RFO(num_agents, max_iter, train_data, train_label, local_search, classifier, obj_function=compute_fitness, trans_function_shape='s',
       save_conv_graph=False, seed=0):

    short_name = 'RFO'
    agent_name = "fox"
    train_data, train_label = np.array(train_data), np.array(train_label)
    num_features = train_data.shape[1]
    np.random.seed(seed)
    trans_function = get_trans_function(trans_function_shape)

    # setting up the objectives
    weight_acc = None
    if obj_function == compute_fitness:
        weight_acc = 0.9
    obj = (obj_function, weight_acc)
    compute_accuracy = (compute_fitness, 1)  # compute_accuracy is just compute_fitness with accuracy weight as 1

    # initialize foxes and Leader (the agent with the max fitness)
    foxes = initialize(num_agents, num_features)
    Leader_agent = np.zeros((1, num_features))
    Leader_fitness = float("-inf")

    # initialize convergence curves
    convergence_curve = {}
    convergence_curve['fitness'] = np.zeros(max_iter)

    binary_solution_rfo = np.empty((max_iter, num_agents, num_features))
    binary_solution_dmlahc = np.empty((max_iter, num_agents, num_features))

    # initialize data class
    data = Data()
    val_size = 0.2
    data.train_X, data.val_X, data.train_Y, data.val_Y = train_test_split(train_data, train_label, stratify=train_label,
                                                                          test_size=val_size, random_state=42)

    # create a solution object
    solution = Solution()
    solution.num_agents = num_agents
    solution.max_iter = max_iter
    solution.num_features = num_features
    solution.obj_function = obj_function
    # rank initial population
    foxes, fitness = sort_agents(foxes, obj, data, classifier)

    # start timer
    start_time = time.time()

    L= -3  # size of search space solution
    R= 3  # size of search space solution
    phi0 = np.random.uniform(0, 2 * 3.14)
    seta = np.random.uniform(0, 1)  # weather conditions

    maxmu = 0.205
    minmu = 0.195
    delta = (maxmu - minmu) / max_iter

    # main loop
    for iter_no in range(max_iter):
        print('\n================================================================================')
        print('                          Iteration - {}'.format(iter_no + 1))
        print('================================================================================\n')

        # reproduction and leaving the herd
        FromIndex = num_agents-0.05*num_agents
        for index in range(int(FromIndex), num_agents):
            habitatCenter = []
            for i in range(num_features):
                habitatCenter.append((foxes[0][i]+foxes[1][i])/2)
            kappa = np.random.uniform(0, 1)
            if kappa >= 0.45:
                for i in range(num_features):
                    foxes[index][i] = np.random.uniform(L, R)
            else:
                for i in range(num_features):
                    # foxes[index][i] = kappa*habitatCenter[i]
                    foxes[index][i] = kappa * foxes[0][i] + (1 - kappa) * foxes[1][i]

        # global phase - food searching
        for i in range(len(foxes)):
            alpha = np.random.uniform(0, math.sqrt(distance.euclidean(foxes[i], foxes[0])))
            # alpha = np.random.uniform(0, math.sqrt(distance.cityblock(foxes[i], foxes[0])))
            # alpha = np.random.uniform(0, math.sqrt(distance.cosine(foxes[i], foxes[0])))
            # alpha = np.random.uniform(0, math.sqrt(distance.chebyshev(foxes[i], foxes[0])))
            for j in range(num_features):
                f = foxes[i][j] + alpha * sign(foxes[0][j]-foxes[i][j])
                if f > R:
                    f = 3
                elif f < L:
                    f = -3
                foxes[i][j] = f

        # local phase - traversing through the local habitat
        a = []
        for xx in range(len(foxes)):
            a.append(np.random.uniform(0, 0.2))

        for i in range(len(foxes)):
            if np.random.uniform(0, 1) > 0.75:
                phi = []
                for xx in range(num_features):
                    phi.append(np.random.uniform(0, 2 * 3.14))
                phi[0] = phi0
                if phi[0] != 0:
                    r = a[i]*math.sin(phi[0])/phi[0]
                else:
                    r = seta
                for j in range(num_features):
                    if j == 0:
                        f = foxes[i][j]+a[i]*r*math.cos(phi[1])
                        if f > R:
                            f = 3
                        elif f < L:
                            f = -3
                        foxes[i][j] = f
                    elif j == num_features-1:
                        for k in range(1, j+1):
                            f = foxes[i][j] + a[i] * r * math.sin(phi[k])
                        if f > R:
                            f = 3
                        elif f < L:
                            f = -3
                        foxes[i][j] = f
                    else:
                        for k in range(1, j+1):
                            f = foxes[i][j] + a[i] * r * math.sin(phi[k])
                        f = f + a[i]*r*math.cos(phi[j+1])
                        if f > R:
                            f = 3
                        elif f < L:
                            f = -3
                        foxes[i][j] = f

            # Apply transformation function on the updated whale
            for j in range(num_features):
                trans_value = trans_function(foxes[i, j])
                # if (np.random.random() < trans_value):
                if 0.5 <= trans_value:
                    foxes[i, j] = 1
                else:
                    foxes[i, j] = 0

            binary_solution_rfo[iter_no][i] = foxes[i]

            if local_search:
                # Add local search for all the chromosomes
                train_X, val_X, train_Y, val_Y = data.train_X, data.val_X, data.train_Y, data.val_Y
                muprob = maxmu - iter_no * delta
                # tolerance = maxmu - iter_no * delta
                foxes[i] = LAHC(foxes[i], train_X, val_X, train_Y, val_Y, weight_acc,
                                  num_features, classifier,
                                # tolerance,
                                muprob
                                )  # To activte LAHC local search
                # foxes[fox] = adaptiveBeta(foxes[fox], train_X, val_X, train_Y, val_Y, weight_acc)

            binary_solution_dmlahc[iter_no][i] = foxes[i]

        # update final information
        foxes, fitness = sort_agents(foxes, obj, data, classifier)
        display(foxes, fitness, agent_name)

        if fitness[0] > Leader_fitness:
            Leader_agent = foxes[0].copy()
            Leader_fitness = fitness[0].copy()

        # convergence_curve['fitness'][iter_no] = np.mean(fitness)
        convergence_curve['fitness'][iter_no] = Leader_fitness

    # compute final accuracy
    Leader_agent, Leader_accuracy = sort_agents(Leader_agent, compute_accuracy, data, classifier)
    # chromosomes, accuracy = sort_agents(foxes, compute_accuracy, data)

    print('\n================================================================================')
    print('                                    Final Result                                  ')
    print('================================================================================\n')
    print('Leader ' + agent_name + ' Dimension : {}'.format(int(np.sum(Leader_agent))))
    print('Leader ' + agent_name + ' Fitness : {}'.format(Leader_fitness))
    print('Leader ' + agent_name + ' Classification Accuracy : {}'.format(Leader_accuracy))
    print('\n================================================================================\n')

    # stop timer
    end_time = time.time()
    exec_time = end_time - start_time

    # plot convergence graph
    fig, axes = Conv_plot(convergence_curve)
    if (save_conv_graph):
        plt.savefig('convergence_graph_' + short_name + '.jpg')
    plt.show()

    # update attributes of solution
    solution.best_agent = Leader_agent
    solution.best_fitness = Leader_fitness
    solution.best_accuracy = Leader_accuracy
    solution.convergence_curve = convergence_curve
    # solution.final_population = chromosomes
    # solution.final_fitness = fitness
    # solution.final_accuracy = accuracy
    solution.execution_time = exec_time
    solution.binary_solution_rfo = binary_solution_rfo
    solution.binary_solution_dmlahc = binary_solution_dmlahc

    return solution
