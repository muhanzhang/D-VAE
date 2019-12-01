import pdb
import pickle
import gzip
import sys
import os
import os.path
import collections
from shutil import copy
import torch
from tqdm import tqdm
import itertools
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sparse_gp import SparseGP
import scipy.stats as sps
import numpy as np
from scipy.io import loadmat
from scipy.stats.stats import pearsonr
sys.path.append('%s/..' % os.path.dirname(os.path.realpath(__file__))) 
sys.path.insert(0, '../')
from util import *

'''
This script is for summarizing the Bayesian optimization and latent space
predictivity results.
'''

# Change experiment settings here

data_type = 'BN'
data_type = 'ENAS'
if data_type == 'ENAS':
    max_n = 8  # number of nodes
    #save_appendix = ['DVAE', 'SVAE', 'GraphRNN', 'GCN']
    #model_name = ['D-VAE', 'S-VAE', 'GraphRNN', 'GCN']
    save_appendix = ['DVAE']
    model_name = ['D-VAE']
    save_appendix = ['DVAE_GCN56_lv3']
    model_name = ['GCN']
    save_appendix = ['DVAE_fast']
    model_name = ['D-VAE']
    
    res_dir = 'ENAS_results/'
    n_iter = 10
    num_random_seeds = 10
    
    random_baseline = False  # whether to compare BO with random 
    include_train_scores = False  # whether to also include train scores when showing best selected arcs's scores
    random_as_test = False  # whether to use results on random test

elif data_type == 'BN':
    max_n = 10  # number of nodes
    #save_appendix = ['DVAE', 'SVAE', 'GraphRNN', 'GCN']
    #model_name = ['D-VAE', 'S-VAE', 'GraphRNN', 'GCN']
    save_appendix = ['DVAE']
    model_name = ['D-VAE']
    save_appendix = ['DVAE_GCN56_lv2']
    model_name = ['GCN']
    

    res_dir = 'BN_results/'
    n_iter = 10
    num_random_seeds = 10

    random_baseline = False  # whether to compare BO with random 
    include_train_scores = False
    random_as_test = False


aggregate_dir = '{}_aggregate_results/'.format(data_type)

aggregate_dir = 'tmp_aggregate_results/'.format(data_type)

if not os.path.exists(aggregate_dir):
    os.makedirs(aggregate_dir) 
copy(os.path.realpath(__file__), aggregate_dir)

if random_as_test:
    test_res_file = 'Random_Test_RMSE_ll.txt'
else:
    test_res_file = 'Test_RMSE_ll.txt'


All_Scores = [[] for _ in model_name]
All_Arcs = [[] for _ in model_name]
All_Random_scores = [[] for _ in model_name]
All_Train_scores = [[] for _ in model_name]
All_Train_arcs = [[] for _ in model_name]
all_test_rmse, all_test_r = [[] for _ in model_name], [[] for _ in model_name]
for random_seed in range(1, num_random_seeds+1):
    save_dir = ['{}results_{}_{}/'.format(res_dir, x, random_seed) for x in save_appendix]  # where to load the BO results of first model
    if random_baseline:
        random_dir = ['{}results_{}_{}/'.format(res_dir, x, random_seed) for x in save_appendix]

    mean_y_train, std_y_train = [0] * len(model_name), [0] * len(model_name)
    for i, x in enumerate(save_appendix):
        mean_y_train[i], std_y_train[i] = load_object('{}results_{}_{}/mean_std_y_train.dat'.format(res_dir, x, random_seed))

    Train_scores = [[] for _ in model_name]
    Train_arcs = [[] for _ in model_name]
    Scores = [[] for _ in model_name]
    Arcs = [[] for _ in model_name]
    Random_scores = [[] for _ in model_name]
    '''
    pbar = tqdm(range(n_iter))
    for iteration in pbar:
        for i, x in enumerate(save_dir):
            scores = load_object("{}scores{}.dat".format(x, iteration))
            arcs = load_object("{}valid_arcs_final{}.dat".format(x, iteration))
            Scores[i].append(scores)
            Arcs[i].append(arcs)
            if random_baseline:
                random_scores = load_object("{}random_scores{}.dat".format(random_dir[i], iteration))
                Random_scores[i].append(random_scores)

        if include_train_scores:
            train_scores = load_object("{}scores-1.dat".format(x))
            train_arcs = load_object("{}valid_arcs_final-1.dat".format(x))
            Train_scores[i].append(train_scores)
            Train_arcs[i].append(train_arcs)
    '''

    test_rmse, test_r = [[] for _ in model_name], [[] for _ in model_name]
    for i in range(len(model_name)):
        Scores[i] = np.array(Scores[i]) * std_y_train[i] + mean_y_train[i]
        Random_scores[i] = np.array(Random_scores[i]) * std_y_train[i] + mean_y_train[i]
        Train_scores[i] = np.array(Train_scores[i])

        All_Scores[i].append(Scores[i])
        All_Arcs[i].append(Arcs[i])
        All_Random_scores[i].append(Random_scores[i])
        All_Train_scores[i].append(Train_scores[i])
        All_Train_arcs[i].append(Train_arcs[i])

        with open(save_dir[i] + test_res_file, 'r') as f:
            lines = f.readlines()
            for j, line in enumerate(lines):
                if j >= n_iter:
                    break
                blocks = line.split(',')
                rmse, ll, r = blocks[0][-6:], blocks[1][-6:], blocks[2][-6:]
                test_rmse[i].append(float(rmse))
                test_r[i].append(float(r))
        all_test_rmse[i].append(test_rmse[i])
        all_test_r[i].append(test_r[i])


for i in range(len(model_name)):
    All_Scores[i] = np.array(All_Scores[i])  # n_random_seeds * n_bo_iters * n_bo_batch_selections
    All_Random_scores[i] = np.array(All_Random_scores[i])
    All_Train_scores[i] = np.array(All_Train_scores[i])
    all_test_rmse[i] = np.array(all_test_rmse[i])
    all_test_r[i] = np.array(all_test_r[i])

print(np.mean(all_test_rmse))
print(np.std(all_test_rmse))
print(np.mean(all_test_r))
print(np.std(all_test_r))
pdb.set_trace()

# plot average scores
fig = plt.figure()
for i in range(len(model_name)):
    plt.errorbar(range(1, n_iter+1), -All_Scores[i].mean(2).mean(0), All_Scores[i].mean(2).std(0), label=model_name[i] + '+BO')
    if random_baseline:
        plt.errorbar(range(1, n_iter+1), -All_Random_scores[i].mean(2).mean(0), All_Random_scores[i].mean(2).std(0), label=model_name[i] + '+Random')
plt.xlabel('Iteration')
if data_type == 'ENAS':
    plt.ylabel('Average weight-sharing accuracy of the selected batch')
elif data_type == 'BN':
    plt.ylabel('Average BIC of the selected batch')
plt.subplots_adjust(left=0.15)
plt.legend()
ax = fig.gca()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.savefig(aggregate_dir + 'Average_scores_plot.pdf')
    

# plot best scores
fig = plt.figure()
for i in range(len(model_name)):
    plt.errorbar(range(1, n_iter+1), -All_Scores[i].min(2).mean(0), All_Scores[i].min(2).std(0), label=model_name[i] + '+BO')
    if random_baseline:
        plt.errorbar(range(1, n_iter+1), -All_Random_scores[i].min(2).mean(0), All_Random_scores[i].min(2).std(0), label=model_name[i] + '+Random')
plt.xlabel('Iteration')
if data_type == 'ENAS':
    plt.ylabel('Highest weight-sharing accuracy of the selected batch')
elif data_type == 'BN':
    plt.ylabel('Highest BIC of the selected batch')
plt.subplots_adjust(left=0.15)
plt.legend()
ax = fig.gca()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.savefig(aggregate_dir + 'Highest_scores_plot.pdf')


# plot best scores over time
def get_highest_over_time(scores):
    highest_mean = scores.max(2).mean(0)
    highest_std = scores.max(2).std(0)
    highest_so_far = [highest_mean[0]]
    std_so_far = [highest_std[0]]
    for i, x in enumerate(highest_mean):
        if i == 0:
            continue
        if x > highest_so_far[-1]:
            cm, cs = x, highest_std[i]
        else:
            cm, cs = highest_so_far[-1], std_so_far[-1]
        highest_so_far.append(cm)
        std_so_far.append(cs)
fig = plt.figure()
for i in range(len(model_name)):
    plt.errorbar(range(1, n_iter+1), *get_highest_over_time(-All_Scores[i]), label=model_name[i] + '+BO')
    if random_baseline:
        plt.errorbar(range(1, n_iter+1), *get_highest_over_time(-All_Random_scores[i]), label=model_name[i] + '+Random')
plt.xlabel('Iteration')
if data_type == 'ENAS':
    plt.ylabel('Highest weight-sharing accuracy over time')
elif data_type == 'BN':
    plt.ylabel('Highest BIC over time')
plt.subplots_adjust(left=0.15)
plt.legend()
ax = fig.gca()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.savefig(aggregate_dir + 'Over_time_highest_scores_plot.pdf')


# plot test rmse, r
for name, label in zip(['rmse', 'r'], ['RMSE', 'Pearson\'s r']):
    fig = plt.figure()
    for i in range(len(model_name)):
        plt.errorbar(range(1, n_iter+1), eval('all_test_{}[{}]'.format(name, i)).mean(0), eval('all_test_{}[{}]'.format(name, i)).std(0), label=model_name[i])
    plt.xlabel('Iteration')
    plt.ylabel('Test {}'.format(label))
    plt.legend()
    ax = fig.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.savefig(aggregate_dir + 'Test_{}_plot.pdf'.format(name))

f = open(aggregate_dir + 'output.txt', 'a')
for name, label in zip(['rmse', 'r'], ['RMSE', 'Pearson\'s r']):
    for i in range(len(model_name)):
        string = 'Model {0}, first iter, '.format(i), name, eval('all_test_{1}[{0}]'.format(i, name)).mean(0)[0], eval('all_test_{}[{}]'.format(name, i)).std(0)[0]
        print(*string)
        print(*string, file=f)
     

# print best arcs
flatten_scores = [[] for _ in model_name]
flatten_arcs = [[] for _ in model_name]
flatten_random_scores = [[] for _ in model_name]
flatten_random_arcs = [[] for _ in model_name]
for i in range(len(model_name)):
    flatten_scores[i] = [xxx for x in All_Scores[i] for xx in x for xxx in xx]
    flatten_arcs[i] = [xxx for x in All_Arcs[i] for xx in x for xxx in xx]
    if include_train_scores:
        flatten_scores[i] += [xxx for x in All_Train_scores[i] for xx in x for xxx in xx]
        flatten_arcs[i] += [xxx for x in All_Train_arcs[i] for xx in x for xxx in xx]
    flatten_scores[i] = {x: y for x, y in zip(flatten_arcs[i], flatten_scores[i])}
    flatten_arcs[i], flatten_scores[i] = list(flatten_scores[i].keys()), list(flatten_scores[i].values())

    k = 15
    top_k_idxs = np.argsort(flatten_scores[i])[:k]
    print("Top {} arcs selected by {} are".format(k, model_name[i]))
    print("Top {} arcs selected by {} are".format(k, model_name[i]), file=f)
    for rank, j in enumerate(top_k_idxs):
        print(flatten_arcs[i][j], -flatten_scores[i][j])
        print(flatten_arcs[i][j], -flatten_scores[i][j], file=f)
        if data_type == 'ENAS':
            row, _ = decode_ENAS_to_igraph(flat_ENAS_to_nested(flatten_arcs[i][j], max_n-2))
        elif data_type == 'BN':
            row, _ = decode_BN_to_igraph(adjstr_to_BN(flatten_arcs[i][j]))
        plot_DAG(row, aggregate_dir, 'Model{}_top{}'.format(i, rank), data_type=data_type, pdf=True)

    if random_baseline:
        # note that there duplicate arcs are not filtered for random
        flatten_random_scores[i] = [xxx for x in All_Random_scores[i] for xx in x for xxx in xx]
        print("Best random scores selected in each space")
        print("Best random scores selected in each space", file=f)
        top_k_idxs = np.argsort(flatten_random_scores[i])[:k]
        for j in top_k_idxs:
            print(-flatten_random_scores[i][j])
            print(-flatten_random_scores[i][j], file=f)

f.close()

