# -*- coding: utf-8 -*-

"""
by Quentin Corlay
"""

import pickle
import sys

def to_pickle(obj, file):
    with open(file, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Adapted from https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console/14879561
# Print iterations progress
def printProgressBar (iteration, total, prefix = 'Progress: ', suffix = 'Complete', decimals = 1, bar_length=100):
    """
        Call in a loop to create terminal progress bar
        @params:
            iteration   - Required  : current iteration (Int)
            total       - Required  : total iterations (Int)
            prefix      - Optional  : prefix string (Str)
            suffix      - Optional  : suffix string (Str)
            decimals    - Optional  : positive number of decimals in percent complete (Int)
            bar_length  - Optional  : character length of bar (Int)
        """
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = '█' * filled_length + '-' * (bar_length - filled_length)

    # print('\r%s |%s| %s%% %s' % (prefix, bar, percents, suffix), end = '\r'),
    # if iteration == total: 
    #     print()

    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),
    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()