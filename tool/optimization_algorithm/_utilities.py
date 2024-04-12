# @File :_utilities.py
# @Time :2023/3/8   10:33
# @Author : zhaozl
# @Describe :

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier as KNN
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_decomposition import PLSRegression
from sklearn.naive_bayes import GaussianNB
from sklearn_lvq import GlvqModel
import matplotlib.pyplot as plt
import joblib


class Solution():
    # structure of the solution
    def __init__(self):
        self.num_features = None
        self.num_agents = None
        self.max_iter = None
        self.obj_function = None
        self.execution_time = None
        self.convergence_curve = {}
        self.best_agent = None
        self.best_fitness = None
        self.best_accuracy = None
        self.final_population = None
        self.final_fitness = None
        self.final_accuracy = None


class Data():
    # structure of the training data
    def __init__(self):
        self.train_X = None
        self.train_Y = None
        self.val_X = None
        self.val_Y = None


def initialize(num_agents, num_features):
    # define min and max number of features
    min_features = int(0.3 * num_features)
    max_features = int(0.6 * num_features)

    # initialize the agents with zeros
    agents = np.zeros((num_agents, num_features))

    # select random features for each agent
    for agent_no in range(num_agents):
        # find random indices
        cur_count = np.random.randint(min_features, max_features)
        temp_vec = np.random.rand(1, num_features)
        temp_idx = np.argsort(temp_vec)[0][0:cur_count]

        # select the features with the ranom indices
        agents[agent_no][temp_idx] = 1

    return agents


def sort_agents(agents, obj, data, classifier, fitness=None):
    # sort the agents according to fitness
    train_X, val_X, train_Y, val_Y = data.train_X, data.val_X, data.train_Y, data.val_Y
    (obj_function, weight_acc) = obj

    if fitness is None:
        # if there is only one agent
        if len(agents.shape) == 1:

            # num_agents = 1
            # fitness = obj_function(agents, train_X, val_X, train_Y, val_Y, weight_acc)
            # model, fitness = compute_accuracy(agents, train_X, val_X, train_Y, val_Y)
            cols = np.flatnonzero(agents)
            if (cols.shape[0] == 0):
                return 0

            if classifier=='lr':
                model = LogisticRegression()
            if classifier=='lda':
                model = LDA()
            if classifier=='knn':
                model = KNN()
            if classifier == 'svm':
                model = SVC(probability=True)
            if classifier == 'rf':
                model = RF()
            if classifier == 'xgboost':
                model = XGBClassifier()
            if classifier == 'dt':
                model = DecisionTreeClassifier()
            if classifier == 'plsda':
                model = PLSRegression(n_components=5)
            if classifier == 'gnb':
                model = GaussianNB()
            if classifier == 'glvq':
                model = GlvqModel()

            train_data = train_X[:, cols]
            train_label = train_Y
            test_data = val_X[:, cols]
            test_label = val_Y

            model.fit(train_data, train_label)
            acc = model.score(test_data, test_label)

            y_prob = model.predict_proba(test_data)

            writer = pd.ExcelWriter('./result/' + classifier + '_output.xlsx', engine='xlsxwriter')
            test_label = pd.DataFrame(test_label)
            test_label.to_excel(writer, sheet_name='val_label', index=False)
            y_prob = pd.DataFrame(y_prob)
            y_prob.to_excel(writer, sheet_name='val_y_prob', index=False)
            writer.save()
            # writer.close()

            # 保存训练好的模型到文件
            model_filename = './result/' + classifier + '_model.joblib'
            joblib.dump(model, model_filename)
            print(f"Model saved as {model_filename}")
            return agents, acc

        # for multiple agents
        else:
            num_agents = agents.shape[0]
            fitness = np.zeros(num_agents)
            for id, agent in enumerate(agents):
                fitness[id] = obj_function(agent, train_X, val_X, train_Y, val_Y, classifier, weight_acc)

    idx = np.argsort(-fitness)
    sorted_agents = agents[idx].copy()
    sorted_fitness = fitness[idx].copy()

    return sorted_agents, sorted_fitness


def display(agents, fitness, agent_name='Agent'):
    # display the population
    print('\nNumber of agents: {}'.format(agents.shape[0]))
    print('\n------------- Best Agent ---------------')
    print('Fitness: {}'.format(fitness[0]))
    print('Number of Features: {}'.format(int(np.sum(agents[0]))))
    print('----------------------------------------\n')

    for id, agent in enumerate(agents):
        print('{} {} - Fitness: {}, Number of Features: {}'.format(agent_name, id + 1, fitness[id], int(np.sum(agent))))

    print('================================================================================\n')


def compute_accuracy(agent, train_X, test_X, train_Y, test_Y, classifier):
    # compute classification accuracy of the given agents
    cols = np.flatnonzero(agent)
    if (cols.shape[0] == 0):
        return 0

    if classifier == 'lr':
        model = LogisticRegression()
    if classifier == 'lda':
        model = LDA()
    if classifier == 'knn':
        model = KNN()
    if classifier == 'svm':
        model = SVC(probability=True)
    if classifier == 'rf':
        model = RF()
    if classifier == 'xgboost':
        model = XGBClassifier()
    if classifier == 'dt':
        model = DecisionTreeClassifier()
    if classifier == 'plsda':
        model = PLSRegression(n_components=5)
    if classifier == 'gnb':
        model = GaussianNB()
    if classifier == 'glvq':
        model = GlvqModel()

    train_data = train_X[:, cols]
    train_label = train_Y
    test_data = test_X[:, cols]
    test_label = test_Y

    model.fit(train_data, train_label)
    acc = model.score(test_data, test_label)
    return acc

def compute_fitness(agent, train_X, test_X, train_Y, test_Y, classifier, weight_acc=0.9):
    # compute a basic fitness measure
    if (weight_acc == None):
        weight_acc = 0.9

    weight_feat = 1 - weight_acc
    num_features = agent.shape[0]

    acc = compute_accuracy(agent, train_X, test_X, train_Y, test_Y, classifier)
    feat = (num_features - np.sum(agent)) / num_features

    fitness = weight_acc * acc + weight_feat * feat
    # return model, fitness
    return fitness

def Conv_plot(convergence_curve):
    # plot convergence curves
    num_iter = len(convergence_curve['fitness'])
    iters = np.arange(num_iter) + 1
    fig, axes = plt.subplots(1)
    fig.tight_layout(pad=5)
    fig.suptitle('Convergence Curves')

    axes.set_title('Convergence of Fitness over Iterations')
    axes.set_xlabel('Iteration')
    axes.set_ylabel('Best Fitness')
    axes.plot(iters, convergence_curve['fitness'])

    return fig, axes




