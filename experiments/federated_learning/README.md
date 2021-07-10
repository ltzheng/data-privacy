# Privacy Preserving Federated Learning

**郑龙韬 PB18061352**
**7/10/2021**

## User Guide

### Prerequisite

```bash
sudo apt install libgmp-dev
sudo apt install libmpc-dev 
pip install gmpy2
pip install phe
pip install termcolor
```

In addition, `torch` is the framework for training.

### Run

#### Plain

Run in CPU:

```bash
python main.py --gpu -1
```

Run in GPU(ID=0):

```bash
python main.py --gpu 0
```

Example results (more graphs are illustrated in the following sections):

> Note: Through this report, the elapsed time is measured by seconds.

![](figs/plain.png)

#### Differential Privacy

Run in GPU(ID=0) with default $C=0.5$ and $\sigma=0.05$:

```bash
python main.py --gpu 0 --mode DP
```

Example results:

![](figs/dp_default.png)

#### Paillier

run in GPU(ID=0):

```bash
python main.py --gpu 0 --mode Paillier
```

Example results:

Besides training time, accuracy and loss, the time of paillier encryption and decryption is also measured.

Epoch 1:

![](figs/paillier.png)

After training 6 epochs:

![](figs/paillier2.png)

See more argument options in [options.py](options.py).

## Differential Privacy in Federated Learning

### Key Implementation

$$
w_{t+1}=w_t+\frac{1}{k}(\sum_{i=0}^{k}\Delta w^i/max(1,\frac{\|\Delta w^i\|_2}{C})+N(0,\sigma^2C^2I))
$$

#### Server code

In `FedAvg` method, add gauss noise $N(0,\sigma^2C^2I)$ for the update weight sum and then apply the average operation.

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

In `train` method, replace the $\sum_{i=0}^{k} \Delta w^i$ in plain federated learning with $\sum_{i=0}^{k}\Delta w^i/max(1,\frac{\|\Delta w^i\|_2}{C})$ as the following code.

```py
elif self.args.mode == 'DP':  # DP mechanism
    for k in w_new.keys():
        update_w[k] = w_new[k] - w_old[k]
        # L2-norm
        sensitivity = torch.norm(update_w[k], p=2)
        # clip
        update_w[k] = update_w[k] / max(1, sensitivity / self.C)
```

The `update` method for DP is the same as plain federated learning, the code is as follows.

```py
if self.args.mode == 'plain' or self.args.mode == 'DP':
    self.model.load_state_dict(w_glob)
```

#### Add DP default arguments

Both in `options.py` and the `__init__` methods of clients and server.

In `options.py`:

```py
# DP arguments
parser.add_argument('--C', type=float, default=0.5, help="DP model clip parameter")
parser.add_argument('--sigma', type=float, default=0.05, help="DP Gauss noise parameter")
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

### Influence of $\sigma, C$ for model's accuracy

Experiment condition: $C \in [0.1, 0.9]$ and $\sigma \in [0.1, 0.4]$ (due to limited computation resource). The experiment code is implemented in [dp_plot.py](dp_plot.py). The results are both in the graph below. Each training/testing accuracy and loss are measured after training **2 epochs** (again, due to limited computation resource).

Results in terminal:

![](figs/dp_exp.png)

3D visualization:

![](figs/)

With $\delta = 10^{-3}$, the according $\epsilon$ is shown in the table below.

### Bonus

When $\sigma$ is small, it may result in $\epsilon \geq 1$, which contradicts the condition of theorem 3.22 (i.e. Classical Gauss Mechanism).

To address this issue, I investigate several papers:

1) Borja Balle, Yu-Xiang Wang. *Improving the Gaussian Mechanism for Differential Privacy: Analytical Calibration and Optimal Denoising*. [[ICML 2018](http://proceedings.mlr.press/v80/balle18a/balle18a.pdf)].

2) Jun Zhao et al. *Reviewing and Improving the Gaussian Mechanism for Differential Privacy*. [[arXiv 2019](https://arxiv.org/abs/1911.12060)].

3) Fang Liu. *Generalized Gaussian Mechanism for Differential Privacy* [[TKDE 2019](https://arxiv.org/pdf/1602.06028.pdf)].

and conclude the following points.

#### Usage of $\epsilon > 1$

Although $\epsilon \leq 1$ is preferred in practical applications, there are still cases where $\epsilon > 1$ is used, so it is necessary to have Gaussian mechanisms which apply to not only $\epsilon \leq 1$ but also $\epsilon > 1$. For example, the Differential Privacy Synthetic Data Challenge organized by the National Institute of Standards and Technology (NIST) included experiments of $\epsilon$ as 10. Also, for a variant of differential privacy called **local differential privacy** which is implemented in several industrial applications (e.g. Apple and Google) have adopted $\epsilon > 1$.

#### Classical Gaussian Mechanism cannot be extended to $\epsilon > 1$

In section 2.3 of the 1st paper, the authors answer the **question**: whether the order of magnitude $\sigma = \Theta(1/\epsilon)$ for $\epsilon \leq 1$ can be extended to privacy parameters of the form $\epsilon > 1$?

They show this is not the case by providing the **lower bound** $\sigma \geq \Delta/\sqrt{2\epsilon}$ (theorem 4 in the paper). As $\epsilon \rightarrow \infty$, the upper bound on $\delta$ converges to $1/2$. Thus, as $\epsilon$ increases the range of $\delta$'s requiring noise of the order $\Omega(1/\sqrt{\epsilon})$ increases to include all parameters of practical interest. This shows that the rate $\sigma = \Theta(1/\epsilon)$ provided by the classical Gaussian mechanism **cannot be extended** beyond the interval $\epsilon \in (0, 1)$.

In the 2nd paper, the fact that classical Gaussian mechanism do not achieve $(\epsilon, \delta)$-DP for large $\epsilon$ given $\delta$ is also shown.

#### Mechanisms that address this issue

##### The Analytic Gaussian Mechanism (1st paper)

In the 1st paper, the authors address these limitations by developing an **optimal Gaussian mechanism** whose **variance is calibrated directly using the Gaussian cumulative density function instead of a tail bound approximation**.

![](figs/analytic_gauss_mechanism.png)

This algorithm is proven to satisfy $(\epsilon, \delta)$-DP theorem 8 of that paper, 


In the 1st paper also propose to **equip the Gaussian mechanism with a post-processing step based on adaptive estimation techniques** by leveraging that the **distribution of the perturbation is known**.

##### New Gaussian Mechanisms proposed by Jun Zhao et al. (2nd paper)

In the 2nd paper, the authors propose Gaussian mechanisms by deriving closed-form upper bounds for $\sigma_{DP-OPT}$. Their mechanisms achieve
$(\epsilon, \delta)$-DP for **any** $\epsilon$, while the classical Gaussian mechanisms do not.

---

## Paillier Cryptosystem

Paillier Cryptosystem is a probabilistic asymmetric algorithm for public key cryptography. The scheme is an additive homomorphic cryptosystem.

### Task 1: Principle of Paillier

In the following, there are 5 methods (`enc`, `dec`, `enc_add`, `enc_add_const`, `enc_mul_const`) filled with `gmpy2`, respectively.

```py

```

The `test` method (for correctness validation):
```py

```

The `main` of `paillier_test.py`:
```py

```

#### Results

The correctness of this implementations is shown in the following 2 figures. The `PASS` in green indicates the decrypted number after operations is equal to the ground truth.

In the case that the number is small:

![](figs/paillier_test_1.png)

In the case that the number is large:

![](figs/paillier_test_2.png)

The `enryption time`, `decryption time` and `total time` of:

- `enc_add`
- `enc_add_const`
- `enc_mul_const`

with 1024-bit keys and 10-1000 bits integers, shown in the following figure.

![](figs/paillier_time.png)

Therefore, it is safe to draw the conclusion that the elapsed time is related to the length of bits. As the bits increase, the time increases.

### Task 2: Paillier Federated Learning

#### Server code

In `FedAvg` method:

```py

```

#### Client code

In `train` method:

```py

```

In `update` method:

```py

```

#### Add Paillier default arguments

In the `__init__` method of clients:

```py
# Paillier initialization
if self.args.mode == 'Paillier':
    self.pub_key = global_pub_key
    self.priv_key = global_priv_key
```

#### Edit `main.py`

In this implementation of paillier federated learning, the server only conduct the summation (no model updating in the `FedAvg` process). Since the test is done by the model of server, it is needed to copy the model of clients to the server in `main.py` in the end of each epoch (before testing).

```py

```

#### Results

correct

running time in MNIST

![](figs/paillier_training_curve.png)

### References

1) Borja Balle, Yu-Xiang Wang. *Improving the Gaussian Mechanism for Differential Privacy: Analytical Calibration and Optimal Denoising*. [ICML 2018](http://proceedings.mlr.press/v80/balle18a/balle18a.pdf).

2) Jun Zhao et al. *Reviewing and Improving the Gaussian Mechanism for Differential Privacy*. [arXiv 2019](https://arxiv.org/abs/1911.12060).

3) Fang Liu. *Generalized Gaussian Mechanism for Differential Privacy* [TKDE 2019](https://arxiv.org/pdf/1602.06028.pdf).

4) Martín Abadi, Andy Chu, Ian J. Goodfellow, H. Brendan McMahan, Ilya Mironov, Kunal Talwar, Li Zhang: Deep Learning with Differential Privacy. ACM Conference on Computer and Communications Security 2016: 308-318

5) Abadi, Martin, et al. Deep learning with differential privacy. Proceedings of the 2016 ACM SIGSAC conference on computer and communications security. 2016.

6) **Methodology, Ethics and Practice of Data Privacy** Course at University of Science and Technology of China. 2021 Spring. (**Lecturer: Prof. Lan Zhang**).
