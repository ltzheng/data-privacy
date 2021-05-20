from main import main
import timeit
from utils import display_table, default_data_config
import matplotlib.pyplot as plt
import numpy as np


default_k = 10
default_maxsup = 20
k_s = np.arange(0, 500, 20)


config = {
    'k': default_k, 
    'maxsup': default_maxsup, 
    'samarati': True, 
    'mondrian': False, 
    'optimal_samarati': False
}
config['data'] = default_data_config

elapsed_time = {'samarati': [], 'mondrian': []}
lm_s = {'samarati': [], 'mondrian': []}
for k in k_s:
    start = timeit.default_timer()
    config['k'] = k
    config['maxsup'] = default_maxsup
    lm_s.append(main(config))
    stop = timeit.default_timer()
    elapsed_time.append(stop - start)

fig, ax = plt.subplots()
# ax.plot(k_s, elapsed_time)
# ax.set(xlabel='k', ylabel='runtime (s)',
#        title='Runtime with different k')
ax.plot(k_s, lm_s)
ax.set(xlabel='k', ylabel='loss metric',
       title='Loss metric with different k')
ax.grid()

# fig.savefig('figs/diff_k.png')
plt.show()