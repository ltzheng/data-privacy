import subprocess
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np
import copy
import math
import re


def compute_epsilon(C, sigma):
    return math.sqrt(2 * math.log(math.e, 1.25 / 1e-3)) * C / sigma

C_list = np.arange(0.1, 1.1, 0.2)
sigma_list = np.arange(0.1, 0.5, 0.1)

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

# C_list, sigma_list = np.meshgrid(C_list, sigma_list)
# accuracy = np.sqrt(C_list**2 + sigma_list**2)

# # Plot the surface.
# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# surf = ax.plot_surface(C_list, sigma_list, accuracy, cmap=cm.coolwarm,
#                        linewidth=0, antialiased=False)

# # Customize the z axis.
# ax.set_zlim(-1.01, 1.01)
# ax.zaxis.set_major_locator(LinearLocator(10))
# # A StrMethodFormatter is used automatically
# ax.zaxis.set_major_formatter('{x:.02f}')

# # Add a color bar which maps values to colors.
# fig.colorbar(surf, shrink=0.5, aspect=5)

# plt.show()
# save_name = "sigma influence to accuracy"
# plt.clf()
# plt.plot(x,y)
# plt.savefig(save_name)
