# Privacy Preserving Federated Learning

**郑龙韬 PB18061352**
**6/25/2021**

## User Guide

### Prerequisite

```
sudo apt install libgmp-dev
sudo apt install libmpc-dev 
pip install gmpy2
pip install phe
```

In addition, we use `torch` for training.

### Run

#### Plain

Run with CPU:

```bash
python main.py --gpu -1
```

Run with GPU(ID=0):

```bash
python main.py --gpu 0
```

Results:

> Note: Through this report, the training time is measured by seconds.

![](figs/plain.png)

#### DP

run with GPU(ID=0):

```bash
python main.py --gpu 0 --mode DP
```

Results:

![](figs/dp_default.png)

#### Paillier

run with GPU(ID=0):

```bash
python main.py --gpu 0 --mode Paillier
```

Results:

> We only train 1 epoch as demo since Paillier is quite slow.

![](figs/paillier_default.png)

See more argument options in [options.py](options.py).

## Differential Privacy

### Key Implementation

#### Server code

In `FedAvg` method:

```py
elif self.args.mode == 'DP':  # DP mechanism
    update_w_avg = copy.deepcopy(self.clients_update_w[0])
    for k in update_w_avg.keys():
        for i in range(1, len(self.clients_update_w)):
            update_w_avg[k] += self.clients_update_w[i][k]
        # add gauss noise
        update_w_avg[k] += torch.normal(0, self.sigma**2 * self.C**2, update_w_avg[k].shape).to(self.args.device)
        update_w_avg[k] = torch.div(update_w_avg[k], len(self.clients_update_w))
        self.model.state_dict()[k] += update_w_avg[k]
```

#### Client code

In `train` method:

```py
elif self.args.mode == 'DP':  # DP mechanism
    for k in w_new.keys():
        update_w[k] = w_new[k] - w_old[k]
        # L2-norm
        sensitivity = torch.norm(update_w[k], p=2)
        # clip
        update_w[k] = update_w[k] / max(1, sensitivity / self.C)
```

The `update` method for DP is the same as plain federated learning, we edit the condition as follows.

```py
if self.args.mode == 'plain' or self.args.mode == 'DP':
    self.model.load_state_dict(w_glob)
```

#### Add DP default arguments

Both in `options.py` and the `__init__` methods of clients and server

In `options.py`:

```py
# DP arguments
parser.add_argument('--C', type=int, default=0.5, help="DP model clip parameter")
parser.add_argument('--sigma', type=int, default=0.05, help="DP Gauss noise parameter")
```

In `__init__` method of clients:

```py
# DP hyperparameters
self.C = self.args.C
```

In `__init__` method of server:

```py
# DP hyperparameters
self.C = self.args.C
self.sigma = self.args.sigma
```

### Experiments

The influence of $\sigma, C$ for model's accuracy

Accordingly, the $\epsilon$ is ($\delta = 10^{-3}$)

### Bonus

For $\epsilon \geq 1$, prove this mechanism satisfies DP.

Other mechanisms that solve this problem:

---

## Paillier Cryptosystem

Paillier Cryptosystem is a probabilistic asymmetric algorithm for public key cryptography. The scheme is an additive homomorphic cryptosystem.

### Principle of Paillier

```py

```

```py

```

```py

```

#### Results

correct

running time

### Paillier & Federated Learning

#### Server code

In `FedAvg` method:

```py
elif self.args.mode == 'Paillier':
    update_w_avg = copy.deepcopy(self.clients_update_w[0])
    for k in update_w_avg.keys():
        for n, update_w in enumerate(self.clients_update_w):
            for i in range(len(update_w_avg[k])):
                # incremental averaging
                update_w_avg[k][i] += (update_w[k][i] - update_w_avg[k][i]) / (n + 1)
    return update_w_avg, sum(self.clients_loss) / len(self.clients_loss)
```

#### Client code

In `train` method:

```py
elif self.args.mode == 'Paillier':  # Paillier encryption
    for k in w_new.keys():
        update_w[k] = w_new[k] - w_old[k]
        # flatten weight
        list_w = update_w[k].view(-1).cpu().tolist()
        # encryption
        for i, elem in enumerate(list_w):
            list_w[i] = self.pub_key.encrypt(elem)
        update_w[k] = list_w
```

In `update` method:

```py
elif self.args.mode == 'Paillier':  # Paillier decryption
    # w_glob is update_w_avg here
    for k in w_glob.keys():
        # decryption
        for i, elem in enumerate(w_glob[k]):
            w_glob[k][i] = self.priv_key.decrypt(elem)
        # reshape to original and update
        origin_shape = list(self.model.state_dict()[k].size())
        torch.FloatTensor(w_glob[k]).to(self.args.device).view(*origin_shape)
        self.model.state_dict()[k] += w_glob[k]
```

#### Add Paillier default arguments

In the `__init__` method of clients:

```py
# Paillier initialization
if self.args.mode == 'Paillier':
    self.pub_key = global_pub_key
    self.priv_key = global_priv_key
```

#### Results

correct

running time in MNIST