import subprocess
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import copy
import math
import re


def compute_epsilon(C, sigma):
    return math.sqrt(2 * math.log(math.e, 1.25 / 1e-3)) * C / sigma

C_list = np.array([0.1, 0.3, 0.5, 0.9])
sigma_list = np.array([0.1, 0.2, 0.3, 0.4])

train_acc_map = {C: dict.fromkeys(sigma_list) for C in C_list}
test_acc_map = copy.deepcopy(train_acc_map)
loss_map = copy.deepcopy(train_acc_map)
epsilon_map = copy.deepcopy(train_acc_map)

for C in C_list:
    for sigma in sigma_list:
        print('Running DP with C:', C, '\tsigma:', sigma)
        epsilon_map[C][sigma] = compute_epsilon(C, sigma)

        cmd = "python main.py --mode DP --gpu 0 --no-plot --epoch 2\
            --C " + str(C) + " --sigma " + str(sigma)
        proc = subprocess.Popen(cmd, -1, stdout=subprocess.PIPE, shell=True)
        out, _ = proc.communicate()
        out = out.decode('utf-8')
        print('out:', out)

        train_acc, test_acc, loss = [float(x) \
            for x in re.findall(r"[-+]?\d*\.\d+|\d+", out)[-3:]]
        print(train_acc, test_acc, loss)
        train_acc_map[C][sigma] = train_acc
        test_acc_map[C][sigma] = test_acc
        loss_map[C][sigma] = loss

print('epsilon:', epsilon_map)
print('train_acc:', train_acc_map)
print('test_acc:', test_acc_map)
print('loss:', loss_map)

def plot(arr, name, z_range=100):
    global C_list, sigma_list
    Z = np.zeros([len(C_list), len(sigma_list)])
    for i, C in enumerate(C_list):
        for j, sigma in enumerate(sigma_list):
            Z[i][j] = arr[C][sigma]
    Z = np.array(Z).reshape(len(C_list), len(sigma_list))

    C_l, sigma_l = np.meshgrid(C_list, sigma_list)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(C_l, sigma_l, Z, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)

    ax.set_xlabel('C')
    ax.set_ylabel('sigma')
    ax.set_zlabel(name)
    ax.set_zlim(0, z_range)
    ax.zaxis.set_major_locator(LinearLocator(10))
    fig.colorbar(surf, shrink=0.5, aspect=5)
    # plt.savefig('figs/' + name + '.png')
    plt.show()

plot(train_acc_map, 'DP_exp_train_acc')
# plot(test_acc_map, 'DP_exp_test_acc')
# plot(epsilon_map, 'DP_exp_epsilon', 5)
