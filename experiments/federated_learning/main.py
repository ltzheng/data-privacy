import numpy as np
import time
import torch
from torchvision import datasets, transforms, utils
from models.Nets import CNNMnist
from options import args_parser
from client import *
from server import *
import copy
from termcolor import colored
import matplotlib.pyplot as plt

def load_dataset():
    trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True, transform=trans_mnist)
    dataset_test = datasets.MNIST('./data/mnist/', train=False, download=True, transform=trans_mnist)
    return dataset_train, dataset_test

def create_client_server():
    num_items = int(len(dataset_train) / args.num_users)
    clients, all_idxs = [], [i for i in range(len(dataset_train))]
    net_glob = CNNMnist(args=args).to(args.device)

    # divide training data, i.i.d.
    # init models with same parameters
    for i in range(args.num_users):
        new_idxs = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - new_idxs)
        new_client = Client(args=args, dataset=dataset_train, idxs=new_idxs, w=copy.deepcopy(net_glob.state_dict()))
        clients.append(new_client)

    server = Server(args=args, w=copy.deepcopy(net_glob.state_dict()))

    return clients, server


if __name__ == '__main__':

    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    print(args.device)

    print("load dataset...")
    dataset_train, dataset_test = load_dataset()

    print("clients and server initialization...")
    clients, server = create_client_server()

    # statistics for plot
    all_acc_train = []
    all_acc_test = []
    all_loss_glob = []

    # training
    print("start training...")
    print('Algorithm:', colored(args.mode, 'green'))
    # Paillier is too slow, train only 1 epoch as demo
    num_epochs = 1 if args.mode == 'Paillier' else args.epochs

    for iter in range(num_epochs):
        epoch_start = time.time()

        server.clients_update_w, server.clients_loss = [], []
        for idx in range(args.num_users):
            update_w, loss = clients[idx].train()
            server.clients_update_w.append(update_w)
            server.clients_loss.append(loss)

        # calculate global weights
        w_glob, loss_glob = server.FedAvg()

        # update local weights
        for idx in range(args.num_users):
            clients[idx].update(w_glob)
        
        epoch_end = time.time()
        print(colored('=====Epoch {:3d}====='.format(iter), 'yellow'))
        print('Training time:', epoch_end - epoch_start)

        # testing
        acc_train, loss_train = server.test(dataset_train)
        acc_test, loss_test = server.test(dataset_test)
        print("Training accuracy: {:.2f}".format(acc_train))
        print("Testing accuracy: {:.2f}".format(acc_test))
        print('Training average loss {:.3f}'.format(loss_glob))
        all_acc_train.append(acc_train)
        all_acc_test.append(acc_test)
        all_loss_glob.append(loss_glob)

    # plot learning curve
    x = np.linspace(0, num_epochs - 1, num_epochs)
    f, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.plot(x, all_acc_train)
    ax1.set_title('Train accuracy')
    ax2.plot(x, all_acc_test)
    ax2.set_title('Train accuracy')
    ax3.plot(x, all_loss_glob)
    ax3.set_title('Training average loss')
    plt.savefig(args.mode + 'training_curve.png')
    plt.show()
