import torch
import copy
from torch.utils.data import DataLoader
import torch.nn.functional as F
from models.Nets import CNNMnist

class Server():
    def __init__(self, args, w):
        self.args = args
        self.clients_update_w = []
        self.clients_loss = []
        self.model = CNNMnist(args=args).to(args.device)
        self.model.load_state_dict(w)
        # DP hyperparameters
        self.C = self.args.C
        self.sigma = self.args.sigma

    def FedAvg(self):
        if self.args.mode == 'plain':
            update_w_avg = copy.deepcopy(self.clients_update_w[0])
            for k in update_w_avg.keys():
                for i in range(1, len(self.clients_update_w)):
                    update_w_avg[k] += self.clients_update_w[i][k]
                update_w_avg[k] = torch.div(update_w_avg[k], len(self.clients_update_w))
                self.model.state_dict()[k] += update_w_avg[k]   
            return copy.deepcopy(self.model.state_dict()), sum(self.clients_loss) / len(self.clients_loss)

        elif self.args.mode == 'DP':  # DP mechanism
            update_w_avg = copy.deepcopy(self.clients_update_w[0])
            for k in update_w_avg.keys():
                for i in range(1, len(self.clients_update_w)):
                    update_w_avg[k] += self.clients_update_w[i][k]
                # add gauss noise
                update_w_avg[k] += torch.normal(0, self.sigma**2 * self.C**2, update_w_avg[k].shape).to(self.args.device)
                update_w_avg[k] = torch.div(update_w_avg[k], len(self.clients_update_w))
                self.model.state_dict()[k] += update_w_avg[k]
            return copy.deepcopy(self.model.state_dict()), sum(self.clients_loss) / len(self.clients_loss)

        elif self.args.mode == 'Paillier':
            update_w_avg = copy.deepcopy(self.clients_update_w[0])
            for k in update_w_avg.keys():
                for n, update_w in enumerate(self.clients_update_w):
                    for i in range(len(update_w_avg[k])):
                        # incremental averaging
                        update_w_avg[k][i] += (update_w[k][i] - update_w_avg[k][i]) / (n + 1)
            return update_w_avg, sum(self.clients_loss) / len(self.clients_loss)

        else:
            raise NotImplementedError

    def test(self, datatest):
        self.model.eval()

        # testing
        test_loss = 0
        correct = 0
        data_loader = DataLoader(datatest, batch_size=self.args.bs)
        for idx, (data, target) in enumerate(data_loader):
            if self.args.gpu != -1:
                data, target = data.cuda(), target.cuda()
            log_probs = self.model(data)

            # sum up batch loss
            test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()

            # get the index of the max log-probability
            y_pred = log_probs.data.max(1, keepdim=True)[1]
            correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

        test_loss /= len(data_loader.dataset)
        accuracy = 100.00 * correct / len(data_loader.dataset)
        return accuracy, test_loss
