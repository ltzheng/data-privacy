import argparse
from main import main
import timeit
from utils import default_data_config
import matplotlib.pyplot as plt
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--samarati", action='store_true')
parser.add_argument("--mondrian", action='store_true')
args = parser.parse_args()

k_s = list(range(10, 200, 20)) + list(range(200, 350, 30))
# k_s = list(range(10, 200, 20)) + list(range(200, 500, 50))
maxsup = [0, 10, 20, 50, 80, 100, 150, 200, 300]

config = {
    'k': 10, 
    'maxsup': 20, 
    'samarati': False, 
    'mondrian': False, 
    'optimal_samarati': False
}
config['data'] = default_data_config

if args.samarati:
    samarati_elapsed_time = {k: [] for k in maxsup}
    samarati_lm_s = {k: [] for k in maxsup}

    for sup in maxsup:
        for k in k_s:
            config['k'] = k
            config['maxsup'] = sup
            config['samarati'] = True
            # count runtime & get loss metric
            start = timeit.default_timer()
            lm = main(config)
            samarati_lm_s[sup].append(lm)
            stop = timeit.default_timer()
            samarati_elapsed_time[sup].append(stop - start)

    plt.figure()
    plt.xlabel('k')
    plt.ylabel('runtime (s)')
    plt.title('Samarati runtime')
    for sup in maxsup:
        plt.plot(k_s, samarati_elapsed_time[sup], label='maxsup='+str(sup))
    plt.legend()
    plt.savefig('figs/samarati_runtime.png')
    plt.show()

    plt.figure()
    plt.xlabel('k')
    plt.ylabel('Samarati loss metric')
    plt.title('Samarati loss metric')
    for sup in maxsup:
        plt.plot(k_s, samarati_lm_s[sup], label='maxsup='+str(sup))
    plt.legend()
    plt.savefig('figs/samarati_lm.png')
    plt.show()

    # print('samarati_elapsed_time:', samarati_elapsed_time)
    # print('samarati_lm_s:', samarati_lm_s)

elif args.mondrian:
    mondrian_elapsed_time = []
    mondrian_lm_s = []

    for k in k_s:
        config['k'] = k
        config['mondrian'] = True
        # count runtime & get loss metric
        start = timeit.default_timer()
        lm = main(config)
        mondrian_lm_s.append(lm)
        stop = timeit.default_timer()
        mondrian_elapsed_time.append(stop - start)

    plt.figure()
    plt.xlabel('k')
    plt.ylabel('runtime (s)')
    plt.title('Mondrian runtime')
    plt.plot(k_s, mondrian_elapsed_time)
    plt.legend()
    plt.savefig('figs/mondrian_runtime.png')
    plt.show()

    plt.figure()
    plt.xlabel('k')
    plt.ylabel('loss metric')
    plt.title('Mondrian loss metric')
    plt.plot(k_s, mondrian_lm_s)
    plt.legend()
    plt.savefig('figs/mondrian_lm.png')
    plt.show()

    # print('mondrian_elapsed_time:', mondrian_elapsed_time)
    # print('mondrian_lm_s:', mondrian_lm_s)