# Privacy Preserving Federated Learning

**郑龙韬 PB18061352**
**6/20/2021**

## User Guide

### Prerequisite

```
sudo apt install libgmp-dev
sudo apt install libmpc-dev 
pip install gmpy2
pip install phe
```

In addition, `torch` is needed.

### Run

#### Plain

Run with GPU(ID=0):

```
python main.py --gpu 0
```

Run with CPU:

```
python main.py --gpu -1
```

Results:

![](figs/plain.png)

#### DP

run with GPU(ID=0):

```
python main.py --gpu 0 --mode DP
```

Results:

![](figs/dp_default.png)

#### Paillier

run with GPU(ID=0):

```
python main.py --gpu 0 --mode Paillier
```

Results:

![](figs/paillier_default.png)

See more options in [options.py](options.py).

## Differential Privacy

### Key Implementation

#### Server code

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

```py
elif self.args.mode == 'DP':  # DP mechanism
    for k in w_new.keys():
        update_w[k] = w_new[k] - w_old[k]
        # L2-norm
        sensitivity = torch.norm(update_w[k], p=2)
        # clip
        update_w[k] = update_w[k] / max(1, sensitivity / self.C)
```

#### Add default arguments

Both in `options.py` and the `__init__` methods of clients and server

In `options.py`:

```py
# DP arguments
parser.add_argument('--C', type=int, default=0.5, help="DP model clip parameter")
parser.add_argument('--sigma', type=int, default=0.05, help="DP Gauss noise parameter")
```

In `__init__` methods of clients 

```py
# DP hyperparameters
self.C = self.args.C
```

In `__init__` methods of server

```py
# DP hyperparameters
self.C = self.args.C
self.sigma = self.args.sigma
```

### Experiments

The influence of $\sigma, C$ for model's accuracy

Accordingly, the $\epsilon$ is ($\delta = 10^{-3}$)

### Addition

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

#### server

```py

```


#### client

```py

```

#### Results

correct

running time in MNIST