import glob
import json
import os
import pickle
from multiprocessing import Pool

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_fill_holes
from skimage.measure import find_contours, label, regionprops
from skimage.transform import resize
# from torchsummary import summary
import elasticdeform
import random


def get_ids(dir, index=None, pattern='_*'):
    '''获取编号'''
    if index is None:
        ids = glob.glob(dir + '*.h5')
    else:
        ids = []
        for i in index:
            id = glob.glob('{}{:02d}{}.h5'.format(dir, i, pattern))
            ids = ids + id
    return sorted(ids)

def create_dir(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

def load_pickle(file, mode='rb'):
    with open(file, mode) as f:
        a = pickle.load(f)
    return a


def save_pickle(obj, file, mode='wb'):
    with open(file, mode) as f:
        pickle.dump(obj, f)


def load_json(file):
    with open(file, 'r') as f:
        a = json.load(f)
    return a


def save_json(obj, file, indent=4, sort_keys=True):
    with open(file, 'w') as f:
        json.dump(obj, f, sort_keys=sort_keys, indent=indent)
