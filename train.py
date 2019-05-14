from __future__ import print_function
import os
import sys
import math
import pickle
import pdb
import argparse
import random
from tqdm import tqdm
from shutil import copy
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import scipy.io
from scipy.linalg import qr 
import igraph
from random import shuffle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from util import *
from models import *
from bayesian_optimization.evaluate_BN import Eval_BN


parser = argparse.ArgumentParser(description='Train Variational Autoencoders for DAGs')
# general settings
parser.add_argument('--data-type', default='ENAS',
                    help='ENAS: ENAS-format CNN structures; BN: Bayesian networks')
parser.add_argument('--data-name', default='final_structures6', help='graph dataset name')
parser.add_argument('--nvt', type=int, default=6, help='number of different node types, \
                    6 for final_structures6, 8 for asia_200k')
parser.add_argument('--save-appendix', default='', 
                    help='what to append to data-name as save-name for results')
parser.add_argument('--save-interval', type=int, default=100, metavar='N',
                    help='how many epochs to wait each time to save model states')
parser.add_argument('--sample-number', type=int, default=20, metavar='N',
                    help='how many samples to generate each time')
parser.add_argument('--no-test', action='store_true', default=False,
                    help='if True, merge test with train, i.e., no held-out set')
parser.add_argument('--reprocess', action='store_true', default=False,
                    help='if True, reprocess data instead of using prestored .pkl data')
parser.add_argument('--keep-old', action='store_true', default=False,
                    help='if True, do not remove any old data in the result folder')
parser.add_argument('--only-test', action='store_true', default=False,
                    help='if True, perform some experiments without training the model')
parser.add_argument('--small-train', action='store_true', default=False,
                    help='if True, use a smaller version of train set')
# model settings
parser.add_argument('--model', default='DVAE', help='model to use: DVAE, SVAE, \
                    DVAE_fast, DVAE_BN, SVAE_oneshot, DVAE_GCN')
parser.add_argument('--load-latest-model', action='store_true', default=False,
                    help='whether to load latest_model.pth')
parser.add_argument('--continue-from', type=int, default=None, 
                    help="from which epoch's checkpoint to continue training")
parser.add_argument('--hs', type=int, default=501, metavar='N',
                    help='hidden size of GRUs')
parser.add_argument('--nz', type=int, default=56, metavar='N',
                    help='number of dimensions of latent vectors z')
parser.add_argument('--bidirectional', action='store_true', default=False,
                    help='whether to use bidirectional encoding')
parser.add_argument('--predictor', action='store_true', default=False,
                    help='whether to train a performance predictor from latent\
                    encodings and a VAE at the same time')
# optimization settings
parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                    help='learning rate (default: 1e-4)')
parser.add_argument('--epochs', type=int, default=100000, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='batch size during training')
parser.add_argument('--infer-batch-size', type=int, default=128, metavar='N',
                    help='batch size during inference')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--all-gpus', action='store_true', default=False,
                    help='use all available GPUs')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
np.random.seed(args.seed)
random.seed(args.seed)
print(args)


'''Prepare data'''
args.file_dir = os.path.dirname(os.path.realpath('__file__'))
args.res_dir = os.path.join(args.file_dir, 'results/{}{}'.format(args.data_name, 
                                                                 args.save_appendix))
if not os.path.exists(args.res_dir):
    os.makedirs(args.res_dir) 

pkl_name = os.path.join(args.res_dir, args.data_name + '.pkl')

# check whether to load pre-stored pickle data
if os.path.isfile(pkl_name) and not args.reprocess:
    with open(pkl_name, 'rb') as f:
        train_data, test_data, graph_args = pickle.load(f)
# otherwise process the raw data and save to .pkl
else:
    # determine data formats according to models, DVAE: igraph, SVAE: string (as tensors)
    if args.model.startswith('DVAE'):
        input_fmt = 'igraph'
    elif args.model.startswith('SVAE'):
        input_fmt = 'string'
    if args.data_type == 'ENAS':
        train_data, test_data, graph_args = load_ENAS_graphs(args.data_name, n_types=args.nvt,
                                                             fmt=input_fmt)
    elif args.data_type == 'BN':
        train_data, test_data, graph_args = load_BN_graphs(args.data_name, n_types=args.nvt,
                                                           fmt=input_fmt)
    with open(pkl_name, 'wb') as f:
        pickle.dump((train_data, test_data, graph_args), f)

# delete old files in the result directory
remove_list = [f for f in os.listdir(args.res_dir) if not f.endswith(".pkl") and 
        not f.startswith('train_graph') and not f.startswith('test_graph') and
        not f.endswith('.pth')]
for f in remove_list:
    tmp = os.path.join(args.res_dir, f)
    if not os.path.isdir(tmp) and not args.keep_old:
        os.remove(tmp)

if not args.keep_old:
    # backup current .py files
    copy('train.py', args.res_dir)
    copy('models.py', args.res_dir)
    copy('util.py', args.res_dir)
    # save command line input
    cmd_input = 'python ' + ' '.join(sys.argv)
    with open(os.path.join(args.res_dir, 'cmd_input.txt'), 'w') as f:
        f.write(cmd_input)
    print('Command line input: ' + cmd_input + ' is saved.')

# construct train data
if args.no_test:
    train_data = train_data + test_data

if args.small_train:
    train_data = train_data[:40]
#train_data = [train_data[6]]
#test_data = train_data
#train_data = train_data[:40]
#train_data = train_data[2:4]
#test_data = train_data



'''Prepare the model'''
# model
model = eval(args.model)(
        graph_args.max_n, 
        graph_args.num_vertex_type, 
        graph_args.START_TYPE, 
        graph_args.END_TYPE, 
        hs=args.hs, 
        nz=args.nz, 
        bidirectional=args.bidirectional
        )
if args.predictor:
    predictor = nn.Sequential(
            nn.Linear(args.nz, args.hs), 
            nn.Tanh(), 
            nn.Linear(args.hs, 1)
            )
    model.predictor = predictor
    model.mseloss = nn.MSELoss(reduction='sum')
# optimizer and scheduler
optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10, verbose=True)

model.to(device)

if args.all_gpus:
    net = custom_DataParallel(model, device_ids=range(torch.cuda.device_count()))

if args.load_latest_model:
    load_module_state(model, os.path.join(args.res_dir, 'latest_model.pth'))
else:
    if args.continue_from is not None:
        epoch = args.continue_from
        load_module_state(model, os.path.join(args.res_dir, 
                                              'model_checkpoint{}.pth'.format(epoch)))
        load_module_state(optimizer, os.path.join(args.res_dir, 
                                                  'optimizer_checkpoint{}.pth'.format(epoch)))
        load_module_state(scheduler, os.path.join(args.res_dir, 
                                                  'scheduler_checkpoint{}.pth'.format(epoch)))

# plot sample train/test graphs
if not os.path.exists(os.path.join(args.res_dir, 'train_graph_id0.pdf')) or args.reprocess:
    if not args.keep_old:
        for data in ['train_data', 'test_data']:
            G = [g for g, y in eval(data)[:10]]
            if args.model.startswith('SVAE'):
                G = [g.to(device) for g in G]
                G = model._collate_fn(G)
                G = model.construct_igraph(G[:, :, :model.nvt], G[:, :, model.nvt:], False)
            for i, g in enumerate(G):
                name = '{}_graph_id{}'.format(data[:-5], i)
                plot_DAG(g, args.res_dir, name, data_type=args.data_type)

            '''
            for i, (g, y) in enumerate(eval(data)):
                if args.model.startswith('SVAE'):
                    g = model.construct_igraph(g[:, :, :model.nvt], g[:, :, model.nvt:], False)[0]
                if i < 10:
                    name = '{}_graph_id{}'.format(data[:-5], i)
                    plot_DAG(g, args.res_dir, name, data_type=args.data_type)
            '''


'''Define some train/test functions'''
def train(epoch):
    model.train()
    train_loss = 0
    recon_loss = 0
    kld_loss = 0
    pred_loss = 0
    shuffle(train_data)
    pbar = tqdm(train_data)
    g_batch = []
    y_batch = []
    for i, (g, y) in enumerate(pbar):
        if args.model.startswith('SVAE'):  # for SVAE, g is tensor
            g = g.to(device)
        g_batch.append(g)
        y_batch.append(y)
        if len(g_batch) == args.batch_size or i == len(train_data) - 1:
            optimizer.zero_grad()
            g_batch = model._collate_fn(g_batch)
            if args.all_gpus:  # does not support predictor yet
                loss = net(g_batch).sum()
                pbar.set_description('Epoch: %d, loss: %0.4f' % (epoch, loss.item()/len(g_batch)))
                recon, kld = 0, 0
            else:
                mu, logvar = model.encode(g_batch)
                loss, recon, kld = model.loss(mu, logvar, g_batch)
                if args.predictor:
                    y_batch = torch.FloatTensor(y_batch).unsqueeze(1).to(device)
                    y_pred = model.predictor(mu)
                    pred = model.mseloss(y_pred, y_batch)
                    loss += pred
                    pbar.set_description('Epoch: %d, loss: %0.4f, recon: %0.4f, kld: %0.4f, pred: %0.4f'\
                            % (epoch, loss.item()/len(g_batch), recon.item()/len(g_batch), 
                            kld.item()/len(g_batch), pred/len(g_batch)))
                else:
                    pbar.set_description('Epoch: %d, loss: %0.4f, recon: %0.4f, kld: %0.4f' % (
                                     epoch, loss.item()/len(g_batch), recon.item()/len(g_batch), 
                                     kld.item()/len(g_batch)))
            loss.backward()
            
            train_loss += float(loss)
            recon_loss += float(recon)
            kld_loss += float(kld)
            if args.predictor:
                pred_loss += float(pred)
            optimizer.step()
            g_batch = []
            y_batch = []

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_data)))

    if args.predictor:
        return train_loss, recon_loss, kld_loss, pred_loss
    return train_loss, recon_loss, kld_loss


def visualize_recon(epoch):
    model.eval()
    # draw some reconstructed train/test graphs to visualize recon quality
    '''
    G = [g for (g, y) in test_data[:10]+train_data[:10]]
    if args.model.startswith('SVAE'):
        G = [g.to(device) for g in G]
        G = model._collate_fn(G)
        G_recon = model.encode_decode(G)
        G = model.construct_igraph(G[:, :, :model.nvt], G[:, :, model.nvt:], False)
    elif args.model.startswith('DVAE'):
        G = model._collate_fn(G)
        G_recon = model.encode_decode(G)
    for i, g in enumerate(G):
        g_recon = G_recon[i]
        name0 = 'graph_epoch{}_id{}_original'.format(epoch, i)
        plot_DAG(g, args.res_dir, name0, data_type=args.data_type)
        name1 = 'graph_epoch{}_id{}_recon'.format(epoch, i)
        plot_DAG(g_recon, args.res_dir, name1, data_type=args.data_type)

    '''
    for i, (g, y) in enumerate(test_data[:10]+train_data[:10]):
        if args.model.startswith('SVAE'):
            g = g.to(device)
            g_recon = model.encode_decode(g)[0]
            g = model.construct_igraph(g[:, :, :model.nvt], g[:, :, model.nvt:], False)[0]
        elif args.model.startswith('DVAE'):
            g_recon = model.encode_decode(g)[0]
        name0 = 'graph_epoch{}_id{}_original'.format(epoch, i)
        plot_DAG(g, args.res_dir, name0, data_type=args.data_type)
        name1 = 'graph_epoch{}_id{}_recon'.format(epoch, i)
        plot_DAG(g_recon, args.res_dir, name1, data_type=args.data_type)


def test():
    # test recon accuracy
    model.eval()
    encode_times = 10
    decode_times = 10
    Nll = 0
    pred_loss = 0
    n_perfect = 0
    print('Testing begins...')
    pbar = tqdm(test_data)
    g_batch = []
    y_batch = []
    for i, (g, y) in enumerate(pbar):
        if args.model.startswith('SVAE'):
            g = g.to(device)
        g_batch.append(g)
        y_batch.append(y)
        if len(g_batch) == args.infer_batch_size or i == len(test_data) - 1:
            g = model._collate_fn(g_batch)
            mu, logvar = model.encode(g)
            _, nll, _ = model.loss(mu, logvar, g)
            pbar.set_description('nll: {:.4f}'.format(nll.item()/len(g_batch)))
            Nll += nll.item()
            if args.predictor:
                y_batch = torch.FloatTensor(y_batch).unsqueeze(1).to(device)
                y_pred = model.predictor(mu)
                pred = model.mseloss(y_pred, y_batch)
                pred_loss += pred.item()
            # construct igraph g from tensor g to check recon quality
            if args.model.startswith('SVAE'):  
                g = model.construct_igraph(g[:, :, :model.nvt], g[:, :, model.nvt:], False)
            for _ in range(encode_times):
                z = model.reparameterize(mu, logvar)
                for _ in range(decode_times):
                    g_recon = model.decode(mu)
                    n_perfect += sum(is_same_DAG(g0, g1) for g0, g1 in zip(g, g_recon))
            g_batch = []
            y_batch = []
    Nll /= len(test_data)
    pred_loss /= len(test_data)
    pred_rmse = math.sqrt(pred_loss)
    acc = n_perfect / (len(test_data) * encode_times * decode_times)
    if args.predictor:
        print('Test average recon loss: {0}, recon accuracy: {1:.4f}, pred rmse: {2:.4f}'.format(
            Nll, acc, pred_rmse))
        return Nll, acc, pred_rmse
    else:
        print('Test average recon loss: {0}, recon accuracy: {1:.4f}'.format(Nll, acc))
        return Nll, acc


def prior_validity(scale_to_train_range=False):
    if scale_to_train_range:
        Z_train, Y_train = extract_latent(train_data)
        z_mean, z_std = Z_train.mean(0), Z_train.std(0)
        z_mean, z_std = torch.FloatTensor(z_mean).to(device), torch.FloatTensor(z_std).to(device)
    n_latent_points = 1000
    decode_times = 10
    n_valid = 0
    print('Prior validity experiment begins...')
    G = []
    G_valid = []
    G_train = [g for g, y in train_data]
    if args.model.startswith('SVAE'):
        G_train = [g.to(device) for g in G_train]
        G_train = model._collate_fn(G_train)
        G_train = model.construct_igraph(G_train[:, :, :model.nvt], G_train[:, :, model.nvt:], False)
    pbar = tqdm(range(n_latent_points))
    cnt = 0
    for i in pbar:
        cnt += 1
        if cnt == args.infer_batch_size or i == n_latent_points - 1:
            z = torch.randn(cnt, model.nz).to(model.get_device())
            if scale_to_train_range:
                z = z * z_std + z_mean  # move to train's latent range
            for j in range(decode_times):
                g_batch = model.decode(z)
                G.extend(g_batch)
                if args.data_type == 'ENAS':
                    for g in g_batch:
                        if is_valid_ENAS(g, graph_args.START_TYPE, graph_args.END_TYPE):
                            n_valid += 1
                            G_valid.append(g)
                elif args.data_type == 'BN':
                    for g in g_batch:
                        if is_valid_BN(g, graph_args.START_TYPE, graph_args.END_TYPE):
                            n_valid += 1
                            G_valid.append(g)
            cnt = 0
    r_valid = n_valid / (n_latent_points * decode_times)
    print('Ratio of valid decodings from the prior: {:.4f}'.format(r_valid))

    G_valid_str = [decode_igraph_to_ENAS(g) for g in G_valid]
    r_unique = len(set(G_valid_str)) / len(G_valid_str) if len(G_valid_str)!=0 else 0.0
    print('Ratio of unique decodings from the prior: {:.4f}'.format(r_unique))

    r_novel = 1 - ratio_same_DAG(G_train, G_valid)
    print('Ratio of novel graphs out of training data: {:.4f}'.format(r_novel))
    return r_valid, r_unique, r_novel


def extract_latent(data):
    model.eval()
    Z = []
    Y = []
    g_batch = []
    for i, (g, y) in enumerate(tqdm(data)):
        if args.model.startswith('SVAE'):
            g_ = g.to(device)
        elif args.model.startswith('DVAE'):
            # copy igraph
            # otherwise original igraphs will save the H states and consume more GPU memory
            g_ = g.copy()  
        g_batch.append(g_)
        if len(g_batch) == args.infer_batch_size or i == len(data) - 1:
            g_batch = model._collate_fn(g_batch)
            mu, _ = model.encode(g_batch)
            mu = mu.cpu().detach().numpy()
            Z.append(mu)
            g_batch = []
        Y.append(y)
    return np.concatenate(Z, 0), np.array(Y)


'''Extract latent representations Z'''
def save_latent_representations(epoch):
    Z_train, Y_train = extract_latent(train_data)
    Z_test, Y_test = extract_latent(test_data)
    latent_pkl_name = os.path.join(args.res_dir, args.data_name +
                                   '_latent_epoch{}.pkl'.format(epoch))
    latent_mat_name = os.path.join(args.res_dir, args.data_name + 
                                   '_latent_epoch{}.mat'.format(epoch))
    with open(latent_pkl_name, 'wb') as f:
        pickle.dump((Z_train, Y_train, Z_test, Y_test), f)
    print('Saved latent representations to ' + latent_pkl_name)
    scipy.io.savemat(latent_mat_name, 
                     mdict={
                         'Z_train': Z_train, 
                         'Z_test': Z_test, 
                         'Y_train': Y_train, 
                         'Y_test': Y_test
                         }
                     )


def interpolation_exp(epoch, num=5):
    print('Interpolation experiments between two random testing graphs')
    interpolation_res_dir = os.path.join(args.res_dir, 'interpolation')
    if not os.path.exists(interpolation_res_dir):
        os.makedirs(interpolation_res_dir) 
    if args.data_type == 'BN':
        eva = Eval_BN(interpolation_res_dir)
    interpolate_number = 10
    model.eval()
    cnt = 0
    for i in range(0, len(test_data), 2):
        cnt += 1
        (g0, _), (g1, _) = test_data[i], test_data[i+1]
        if args.model.startswith('SVAE'):
            g0 = g0.to(device)
            g1 = g1.to(device)
            g0 = model._collate_fn([g0])
            g1 = model._collate_fn([g1])
        z0, _ = model.encode(g0)
        z1, _ = model.encode(g1)
        print('norm of z0: {}, norm of z1: {}'.format(torch.norm(z0), torch.norm(z1)))
        print('distance between z0 and z1: {}'.format(torch.norm(z0-z1)))
        Z = []  # to store all the interpolation points
        for j in range(0, interpolate_number + 1):
            zj = z0 + (z1 - z0) / interpolate_number * j
            Z.append(zj)
        Z = torch.cat(Z, 0)
        # decode many times and select the most common one
        G, G_str = decode_from_latent_space(Z, model, return_igraph=True,
                                            data_type=args.data_type) 
        names = []
        scores = []
        for j in range(0, interpolate_number + 1):
            namej = 'graph_interpolate_{}_{}_of_{}'.format(i, j, interpolate_number)
            namej = plot_DAG(G[j], interpolation_res_dir, namej, backbone=True, 
                             data_type=args.data_type)
            names.append(namej)
            if args.data_type == 'BN':
                scorej = eva.eval(G_str[j])
                scores.append(scorej)
        fig = plt.figure(figsize=(120, 20))
        for j, namej in enumerate(names):
            imgj = mpimg.imread(namej)
            fig.add_subplot(1, interpolate_number + 1, j + 1)
            plt.imshow(imgj)
            if args.data_type == 'BN':
                plt.title('{}'.format(scores[j]), fontsize=40)
            plt.axis('off')
        plt.savefig(os.path.join(args.res_dir, 
                    args.data_name + '_{}_interpolate_exp_ensemble_epoch{}_{}.pdf'.format(
                    args.model, epoch, i)), bbox_inches='tight')
        '''
        # draw figures with the same height
        new_name = os.path.join(args.res_dir, args.data_name + 
                                '_{}_interpolate_exp_ensemble_{}.pdf'.format(args.model, i))
        combine_figs_horizontally(names, new_name)
        '''
        if cnt == num:
            break


def interpolation_exp2(epoch):
    if args.data_type != 'ENAS':
        return
    print('Interpolation experiments between flat-net and dense-net')
    interpolation_res_dir = os.path.join(args.res_dir, 'interpolation2')
    if not os.path.exists(interpolation_res_dir):
        os.makedirs(interpolation_res_dir) 
    interpolate_number = 10
    model.eval()
    n = graph_args.max_n
    g0 = [[1]+[0]*(i-1) for i in range(1, n-1)]  # this is flat-net
    g1 = [[1]+[1]*(i-1) for i in range(1, n-1)]  # this is dense-net

    if args.model.startswith('SVAE'):
        g0, _ = decode_ENAS_to_tensor(str(g0), n_types=6)
        g1, _ = decode_ENAS_to_tensor(str(g1), n_types=6)
        g0 = g0.to(device)
        g1 = g1.to(device)
        g0 = model._collate_fn([g0])
        g1 = model._collate_fn([g1])
    elif args.model.startswith('DVAE'):
        g0, _ = decode_ENAS_to_igraph(str(g0))
        g1, _ = decode_ENAS_to_igraph(str(g1))
    z0, _ = model.encode(g0)
    z1, _ = model.encode(g1)
    print('norm of z0: {}, norm of z1: {}'.format(torch.norm(z0), torch.norm(z1)))
    print('distance between z0 and z1: {}'.format(torch.norm(z0-z1)))
    Z = []  # to store all the interpolation points
    for j in range(0, interpolate_number + 1):
        zj = z0 + (z1 - z0) / interpolate_number * j
        Z.append(zj)
    Z = torch.cat(Z, 0)
    # decode many times and select the most common one
    G, _ = decode_from_latent_space(Z, model, return_igraph=True, data_type=args.data_type)  
    names = []
    for j in range(0, interpolate_number + 1):
        namej = 'graph_interpolate_{}_of_{}'.format(j, interpolate_number)
        namej = plot_DAG(G[j], interpolation_res_dir, namej, backbone=True, 
                         data_type=args.data_type)
        names.append(namej)
    fig = plt.figure(figsize=(120, 20))
    for j, namej in enumerate(names):
        imgj = mpimg.imread(namej)
        fig.add_subplot(1, interpolate_number + 1, j + 1)
        plt.imshow(imgj)
        plt.axis('off')
    plt.savefig(os.path.join(args.res_dir, 
                args.data_name + '_{}_interpolate_exp2_ensemble_epoch{}.pdf'.format(
                args.model, epoch)), bbox_inches='tight')


def interpolation_exp3(epoch):
    if args.data_type != 'ENAS':
        return
    print('Interpolation experiments around a great circle')
    interpolation_res_dir = os.path.join(args.res_dir, 'interpolation3')
    if not os.path.exists(interpolation_res_dir):
        os.makedirs(interpolation_res_dir) 
    interpolate_number = 36
    model.eval()
    n = graph_args.max_n
    g0 = [[1]+[0]*(i-1) for i in range(1, n-1)]  # this is flat-net
    if args.model.startswith('SVAE'):
        g0, _ = decode_ENAS_to_tensor(str(g0), n_types=6)
        g0 = g0.to(device)
        g0 = model._collate_fn([g0])
    elif args.model.startswith('DVAE'):
        g0, _ = decode_ENAS_to_igraph(str(g0))
    z0, _ = model.encode(g0)
    norm0 = torch.norm(z0)
    z1 = torch.ones_like(z0)
    # there are infinite possible directions that are orthogonal to z0,
    # we just randomly pick one from a finite set
    dim_to_change = random.randint(0, z0.shape[1]-1)  # this to get different great circles
    print(dim_to_change)
    z1[0, dim_to_change] = -(z0[0, :].sum() - z0[0, dim_to_change]) / z0[0, dim_to_change]
    z1 = z1 / torch.norm(z1) * norm0
    print('z0: ', z0, 'z1: ', z1, 'dot product: ', (z0 * z1).sum().item())
    print('norm of z0: {}, norm of z1: {}'.format(norm0, torch.norm(z1)))
    print('distance between z0 and z1: {}'.format(torch.norm(z0-z1)))
    omega = torch.acos(torch.dot(z0.flatten(), z1.flatten()))
    print('angle between z0 and z1: {}'.format(omega))
    Z = []  # to store all the interpolation points
    for j in range(0, interpolate_number + 1):
        theta = 2*math.pi / interpolate_number * j
        zj = z0 * np.cos(theta) + z1 * np.sin(theta)
        Z.append(zj)
    Z = torch.cat(Z, 0)
    # decode many times and select the most common one
    G, _ = decode_from_latent_space(Z, model, return_igraph=True, data_type=args.data_type) 
    names = []
    for j in range(0, interpolate_number + 1):
        namej = 'graph_interpolate_{}_of_{}'.format(j, interpolate_number)
        namej = plot_DAG(G[j], interpolation_res_dir, namej, backbone=True, 
                         data_type=args.data_type)
        names.append(namej)
    # draw figures with the same height
    new_name = os.path.join(args.res_dir, args.data_name + 
                            '_{}_interpolate_exp3_ensemble_epoch{}.pdf'.format(args.model, epoch))
    combine_figs_horizontally(names, new_name)


def smoothness_exp(epoch, gap=0.05):
    print('Smoothness experiments around zero latent vector')
    smoothness_res_dir = os.path.join(args.res_dir, 'smoothness')
    if not os.path.exists(smoothness_res_dir):
        os.makedirs(smoothness_res_dir) 
    
    #z0 = torch.zeros(1, model.nz).to(device)  # use all-zero vector as center
    
    if args.data_type == 'ENAS': 
        g_str = '4 4 0 3 0 0 5 0 0 1 2 0 0 0 0 5 0 0 0 1 0'  # the best found by SVAE BO
        row = [int(x) for x in g_str.split()]
        row = flat_ENAS_to_nested(row, model.max_n-2)
        if args.model.startswith('SVAE'):
            g0, _ = decode_ENAS_to_tensor(row, n_types=model.max_n-2)
            g0 = g0.to(device)
            g0 = model._collate_fn([g0])
        elif args.model.startswith('DVAE'):
            g0, _ = decode_ENAS_to_igraph(row)
    elif args.data_type == 'BN':
        g0 = train_data[20][0]
        if args.model.startswith('SVAE'):
            g0 = g0.to(device)
            g0 = model._collate_fn([g0])
    z0, _ = model.encode(g0)

    # select two orthogonal directions in latent space
    tmp = np.random.randn(z0.shape[1], z0.shape[1])
    Q, R = qr(tmp)
    dir1 = torch.FloatTensor(tmp[0:1, :]).to(device)
    dir2 = torch.FloatTensor(tmp[1:2, :]).to(device)

    # generate architectures along two orthogonal directions
    grid_size = 13
    mid = grid_size // 2
    Z = []
    pbar = tqdm(range(grid_size ** 2))
    for idx in pbar:
        i, j = divmod(idx, grid_size)
        zij = z0 + dir1 * (i - mid) * gap + dir2 * (j - mid) * gap
        Z.append(zij)
    Z = torch.cat(Z, 0)
    if True:
        G, _ = decode_from_latent_space(Z, model, return_igraph=True, data_type=args.data_type)
    else:  # decode by 3 batches in case of GPU out of memory 
        Z0, Z1, Z2 = Z[:len(Z)//3, :], Z[len(Z)//3:len(Z)//3*2, :], Z[len(Z)//3*2:, :]
        G = []
        G += decode_from_latent_space(Z0, model, return_igraph=True, data_type=args.data_type)[0]
        G += decode_from_latent_space(Z1, model, return_igraph=True, data_type=args.data_type)[0]
        G += decode_from_latent_space(Z2, model, return_igraph=True, data_type=args.data_type)[0]
    names = []
    for idx in pbar:
        i, j = divmod(idx, grid_size)
        pbar.set_description('Drawing row {}/{}, col {}/{}...'.format(i+1, 
                             grid_size, j+1, grid_size))
        nameij = 'graph_smoothness{}_{}'.format(i, j)
        nameij = plot_DAG(G[idx], smoothness_res_dir, nameij, data_type=args.data_type)
        names.append(nameij)
    fig = plt.figure(figsize=(200, 200))
    nrow, ncol = grid_size, grid_size
    for ij, nameij in enumerate(names):
        imgij = mpimg.imread(nameij)
        fig.add_subplot(nrow, ncol, ij + 1)
        plt.imshow(imgij)
        plt.axis('off')
    plt.savefig(os.path.join(args.res_dir, 
                args.data_name + '_{}_smoothness_ensemble_epoch{}_gap={}.pdf'.format(
                args.model, epoch, gap)), bbox_inches='tight')


'''Training begins here'''
min_loss = math.inf  # >= python 3.5
min_loss_epoch = None
loss_name = os.path.join(args.res_dir, 'train_loss.txt')
loss_plot_name = os.path.join(args.res_dir, 'train_loss_plot.pdf')
test_results_name = os.path.join(args.res_dir, 'test_results.txt')
if os.path.exists(loss_name) and not args.keep_old:
    os.remove(loss_name)

if args.only_test:
    epoch = args.continue_from
    #save_latent_representations(epoch)
    #visualize_recon(300)
    #interpolation_exp2(epoch)
    interpolation_exp3(epoch)
    #prior_validity(True)
    #test()
    #smoothness_exp(epoch, 0.1)
    smoothness_exp(epoch, 0.05)
    #interpolation_exp(epoch)
    pdb.set_trace()

start_epoch = args.continue_from if args.continue_from is not None else 0
for epoch in range(start_epoch + 1, args.epochs + 1):
    if args.predictor:
        train_loss, recon_loss, kld_loss, pred_loss = train(epoch)
    else:
        train_loss, recon_loss, kld_loss = train(epoch)
        pred_loss = 0.0
    with open(loss_name, 'a') as loss_file:
        loss_file.write("{:.2f} {:.2f} {:.2f} {:.2f}\n".format(
            train_loss/len(train_data), 
            recon_loss/len(train_data), 
            kld_loss/len(train_data), 
            pred_loss/len(train_data), 
            ))
    scheduler.step(train_loss)
    if epoch % args.save_interval == 0:
        print("save current model...")
        model_name = os.path.join(args.res_dir, 'model_checkpoint{}.pth'.format(epoch))
        optimizer_name = os.path.join(args.res_dir, 'optimizer_checkpoint{}.pth'.format(epoch))
        scheduler_name = os.path.join(args.res_dir, 'scheduler_checkpoint{}.pth'.format(epoch))
        torch.save(model.state_dict(), model_name)
        torch.save(optimizer.state_dict(), optimizer_name)
        torch.save(scheduler.state_dict(), scheduler_name)
        print("visualize reconstruction examples...")
        visualize_recon(epoch)
        print("extract latent representations...")
        save_latent_representations(epoch)
        print("sample from prior...")
        sampled = model.generate_sample(args.sample_number)
        for i, g in enumerate(sampled):
            namei = 'graph_{}_sample{}.pdf'.format(epoch, i)
            plot_DAG(g, args.res_dir, namei, data_type=args.data_type)
        print("plot train loss...")
        losses = np.loadtxt(loss_name)
        if losses.ndim == 1:
            continue
        fig = plt.figure()
        num_points = losses.shape[0]
        plt.plot(range(1, num_points+1), losses[:, 0], label='Total')
        plt.plot(range(1, num_points+1), losses[:, 1], label='Recon')
        plt.plot(range(1, num_points+1), losses[:, 2], label='KLD')
        plt.plot(range(1, num_points+1), losses[:, 3], label='Pred')
        plt.xlabel('Epoch')
        plt.ylabel('Train loss')
        plt.legend()
        plt.savefig(loss_plot_name)

'''Testing begins here'''
if args.predictor:
    Nll, acc, pred_rmse = test()
else:
    Nll, acc = test()
    pred_rmse = 0
r_valid, r_unique, r_novel = prior_validity(True)
with open(test_results_name, 'a') as result_file:
    result_file.write("Epoch {} Test recon loss: {} recon acc: {:.4f} r_valid: {:.4f}".format(
            epoch, Nll, acc, r_valid) + 
            " r_unique: {:.4f} r_novel: {:.4f} pred_rmse: {:.4f}\n".format(
            r_unique, r_novel, pred_rmse))
interpolation_exp2(epoch)
smoothness_exp(epoch)
interpolation_exp3(epoch)
interpolation_exp(epoch)

pdb.set_trace()
