import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
from models.Nets import CNNMnist
import copy
from phe import paillier

global_pub_key, global_priv_key = paillier.generate_paillier_keypair()

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

class Client():
    
    def __init__(self, args, dataset=None, idxs=None, w = None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        self.model = CNNMnist(args=args).to(args.device)
        self.model.load_state_dict(w)
        # DP hyperparameters
        self.C = self.args.C
        # Paillier initialization
        if self.args.mode == 'Paillier':
            self.pub_key = global_pub_key
            self.priv_key = global_priv_key
        
    def train(self):
        w_old = copy.deepcopy(self.model.state_dict())
        net = copy.deepcopy(self.model)

        # train and update
        net.train()   
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
        
        w_new = net.state_dict()

        update_w = {}
        if self.args.mode == 'plain':
            for k in w_new.keys():
                update_w[k] = w_new[k] - w_old[k]
                
        elif self.args.mode == 'DP':  # DP mechanism
            for k in w_new.keys():
                update_w[k] = w_new[k] - w_old[k]
                # L2-norm
                sensitivity = torch.norm(update_w[k], p=2)
                # clip
                update_w[k] = update_w[k] / max(1, sensitivity / self.C)

        elif self.args.mode == 'Paillier':  # Paillier encryption
            for k in w_new.keys():
                update_w[k] = w_new[k] - w_old[k]
                origin_shape = list(update_w[k].size())
                # flatten update weight
                update_w[k] = update_w[k].view(1, -1)
                # encryption
                for i, elem in enumerate(update_w[k]):
                    update_w[k][i] = self.pub_key.encrypt(elem)
                # reshape to original one
                update_w[k] = update_w[k].view(*origin_shape)
        else:
            raise NotImplementedError

        return update_w, sum(batch_loss) / len(batch_loss)

    def update(self, w_glob):
        if self.args.mode == 'plain' or self.args.mode == 'DP':
            self.model.load_state_dict(w_glob)
        elif self.args.mode == 'Paillier':  # Paillier decryption
            for k in w_glob.keys():
                origin_shape = list(w_glob[k].size())
                # flatten global weight
                w_glob[k] = w_glob[k].view(1, -1)
                # decryption
                for i, elem in enumerate(w_glob[k]):
                    w_glob[k][i] = self.priv_key.decrypt(elem)
                # reshape to original one
                w_glob[k] = w_glob[k].view(*origin_shape)
            self.model.load_state_dict(w_glob)
        else:
            raise NotImplementedError