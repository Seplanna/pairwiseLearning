import sys
import numpy as np
from numpy.linalg import inv
from dataUtils import *
from Matrics import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--i', type = int)
parser.add_argument('--u', type = int)
parser.add_argument('--f', type = int)
parser.add_argument('--m', type = int)
FLAGS, unparsed = parser.parse_known_args()

#items_names_file = "../../DATA/ml-20m/movies.csv"
data_dir = "../GeneratedDataAmazone/data" + str(FLAGS.f)
#data_dir = "../GeneratedData/data1"

def GetNextItem(items, used_items, l):
    u_i = list(used_items)
    if (len(u_i) < 1):
        p = np.zeros([1,items.shape[1]])
    else:
        p = items[u_i]
        p = np.vstack([p, np.zeros(p.shape[1])])
    ones = np.eye(p.shape[0])
    ones *= l
    best_res = 0
    best_item = 0
    for item in range(len(items)):
        if item not in used_items:
            p[-1] = items[item]
            res = np.matrix.trace(inv(np.dot(p, p.T) + ones))
            if res < best_res:
                best_res = res
                best_item = item
    return best_item

def MostInformativeItems(items, n_items):
    used_items = []
    for i in range(n_items):
        print(i)
        used_items.append(GetNextItem(items, used_items, 0.01))
    return used_items

def Test():
    item_vecs, item_bias, user_vecs, user_bias, global_bias, user_vecs_train, user_bias_train = GetData(data_dir)
    popular_items = np.genfromtxt(data_dir + "/train_ratings.txt_").astype(int)
    informative_items = np.where(np.abs(item_bias[popular_items]) < 0.2)[0]
    questions = MostInformativeItems(item_vecs[popular_items[informative_items]], FLAGS.i)
    np.savetxt(data_dir + "/yahoo_items", informative_items[questions])

Test()
 
