from random import gauss
import numpy as np
import math
#import scipy.stats as stats
#from sklearn.cluster import KMeans


def SUCSESS():
    return 0.1


def recieveAnswer(user, user_bias, item, item_bias, global_bias, sigma):
    mean = np.dot(item, user)
    return mean + user_bias + item_bias + global_bias


def GetSimilarUsers(user, user_bias, users_estimation, user_bias_estimation):
    n_users = users_estimation.shape[0]
    distance_to_users = [[np.dot(user - users_estimation[i], (user - users_estimation[i]).T) +
                          (user_bias - user_bias_estimation[i])**2, i]
                          for i in range(n_users)]
    #distance_to_users = [[np.dot(user, users_estimation[i].T) +
    #                      (user_bias - user_bias_estimation[i]) ** 2, i]
    #                     for i in range(n_users)]
    distance_to_users.sort(key=lambda x:x[0])
    return distance_to_users

def GetReward(distance_to_users, users, users_bias, item, item_bias, global_bias, sigma):
    n_users = 0
    result = 0
    i = 0
    while i < users.shape[0] and distance_to_users[i][0] < 0.01:
        n_users += 1
        result += int(recieveAnswer(users[distance_to_users[i][1]], users_bias[distance_to_users[i][1]],
                                   item, item_bias, global_bias, sigma)
                     > SUCSESS())
        i += 1
    #print('n_users = ', n_users)
    return result / float(n_users)

def GetData(datadir):
    item_vecs = np.genfromtxt(datadir + "/items1.txt")
    item_bias = np.genfromtxt(datadir + "/items_bias1.txt")
    user_vecs_test = np.genfromtxt(datadir + "/users.txt")
    user_vecs_train = np.genfromtxt(datadir + "/users_train.txt")
    user_bias_train = np.genfromtxt(datadir + "/user_bias_train.txt")
    user_bias_test = np.genfromtxt(datadir + "/user_bias.txt")
    global_bias = 0.
    with open(datadir + "/global_bias.txt", 'r') as global_b:
        for line in global_b:
            global_bias = float(line.strip())
    return item_vecs, item_bias, user_vecs_test, user_bias_test, global_bias, user_vecs_train, user_bias_train

def ExpandData(user_vecs, user_bias, expand):
    n_users = user_vecs.shape[0]
    new_user_vecs = np.empty([expand * n_users, user_vecs.shape[1]])
    new_user_bias = np.empty(expand * n_users)
    for i in range(n_users):
        for j in range(expand):
            new_user_vecs[i * expand + j] = user_vecs[i]
            new_user_bias[i * expand + j] = user_bias[i]
    return [new_user_vecs, new_user_bias]


def appendAllArrays(arrays):
    c = np.array([])
    for ar in arrays:
        c = np.append(c, ar)
    return c


def make_input(item_vec, item_bias, user_vec, user_bias):
    return appendAllArrays([item_vec, item_bias, user_vec, user_bias, item_vec * user_vec])
    #return item_vec * user_vec




def OneStep(user, user_bias, item, item_bias, global_bias, r, learning_rate, user_bias_reg, user_fact_reg):
    prediction = global_bias + user_bias + item_bias
    prediction += np.dot(user, item)
    e = (r - prediction)  # error

    # Update biases
    user_bias += learning_rate * \
                         (e - user_bias_reg * user_bias)
    # Update latent factors
    user += learning_rate * \
                            (e * item - \
                             user_fact_reg * user)
    return [user, user_bias]


def GetBestItem(item_vecs, item_bias, user, user_bias, W, used_items):
    y1_arg = 0
    element = -1000
    for item1 in range(item_vecs.shape[0]):
        c = np.dot(W, make_input(item_vecs[item1], item_bias[item1],
                                                        user, user_bias))
        if (element < c and not item1 in used_items):
            element = c
            y1_arg = item1
    return y1_arg

def SortIItemByPopularity(users, users_bias, item, item_bias, global_bias, sigma):
    result = []
    n_users = users.shape[0]
    for i in range(item.shape[0]):
        result.append([
            sum(float(recieveAnswer(users[j], users_bias[j], item[i], item_bias[i], global_bias, sigma) > SUCSESS())
                for j in range(users.shape[0])) / n_users ,
            item_bias[i] , i])
    result.sort(key = lambda x:x[0])
    return result

def learning_rate(step):
    return 1./math.sqrt(step)

def GetItemsNames(file):
    res = {}
    with open(file) as f:
        for line in f:
            try:
                line = line.split(',')
                res[int(line[0])] = [line[1], line[-1]]
            except:
                continue
    return res

def PrintItemPopularity(user_vecs, user_bias, item_vecs, item_bias, global_bias, sigma):
    sort_items = SortIItemByPopularity(user_vecs, user_bias, item_vecs, item_bias, global_bias, sigma)
    items_test = np.genfromtxt("data/test_items.txt")
    items_names = GetItemsNames("../dataset/ml-1m/movies_mine.dat")
    ratings = []
    ratings1 = []
    with open("itemPopularity", 'w') as ip:
        for i in sort_items:
            try:
                if (float(items_names[items_test[i[2]] + 1][1]) > 0):
                    #print(i[-1], items_names[items_test[i[-1]] + 1])
                    ip.write(str(i[0]) + "\t" + str(i[1]) + "\t"  + str(items_test[i[2]]) +
                         "\t" + items_names[items_test[i[2]] + 1][0] +
                         "\t" + items_names[items_test[i[2]] + 1][1].strip() +
                         '\n')
                    ratings.append(float(items_names[items_test[i[2]] + 1][1]) / 10.)
                    ratings1.append(i[0])
            except:
                continue
    ratings = np.array(ratings)
    ratings1 = np.array(ratings1)
    ratings -= np.min(ratings)
    ratings /= np.max(ratings)
    a1 = np.arange(ratings.shape[0])
    a = np.argsort(ratings)
    print(np.corrcoef(ratings,ratings1)[0][1])
    return  np.corrcoef(ratings,ratings1)[0][1]

def GetItemPopularity():
    result = []
    with open("itemPopularity", 'r') as ip:
        for line in ip:
            line = int(line.strip().split("\t")[2])
            result.append(line)
    return list(reversed(result))

def DeleteMostPopularItems(n_delete_items):
    items_pop = GetItemPopularity()
    with open("data/items.txt") as items, \
         open("data/items_bias.txt") as i_b, \
         open("data/items1.txt", 'w') as n_items,\
         open("data/items_bias1.txt", 'w') as n_ib:
        n_line = 0
        for line in items:
            if n_line not in items_pop[:n_delete_items]:
                n_items.write(line)
            else:
                n_items.write('\t'.join(['0' for i in range(10)]) + '\n')
            n_line += 1
        n_line = 0
        for line in i_b:
            if n_line not in items_pop[:n_delete_items]:
                n_ib.write(line)
            else:
                n_ib.write('0' + '\n')
            n_line += 1

def GetNRatingsOfItems():
    result = {}
    with open('data/items_ratings.txt') as f:
        for line in f:
            line = line.strip().split()
            result[int(line[0])] = int(line[1])
    return result


def DeleteItemsWithoutManyRatings(threshold):
    itemsRatings = GetNRatingsOfItems()
    items = open("data/items.txt").readlines()
    items1 = open("data/items1.txt", 'w')
    items_b = open("data/items_bias.txt").readlines()
    items_b1 = open("data/items_bias1.txt", 'w')
    dictionaryToItems = {}
    newline_n = 0
    for line_n, line in enumerate(items):
        if itemsRatings[line_n] > threshold:
            dictionaryToItems[line_n] = newline_n
            items1.write(line)
            items_b1.write(items_b[line_n])
            newline_n += 1
    with open('itemPopularity') as iP, open('data/itemPopularity', 'w') as nIP:
        for line in iP:
            line = line.strip().split('\t')
            if (itemsRatings[int(line[2])] > threshold):
                line[2] = str(dictionaryToItems[int(line[2])])
                nIP.write('\t'.join(line) + '\n')


def VectorToString(vector):
    return "_".join(str(v) for v in vector)

def LearningRate(learning_rate, step):
    res = learning_rate / step
    if (res < 0.01):
        res = 0.01
    return res

def GetDistributionOfNorms():
    item_vecs, item_bias, user_vecs, user_bias, global_bias = GetData("data")
    norms = []
    for i in item_vecs:
        norms.append(math.sqrt(np.dot(i, i)))
    norms.sort()
    norms = np.array(norms)
    np.savetxt("data/item_norms", norms)

def GetInverseMatrix(previousMatrix, new_vector):
    v = np.dot(previousMatrix, new_vector)
    return previousMatrix - \
    (np.dot(previousMatrix, v) / (1. + np.trace(v)))


class StateStat(object):
    def __init__(self):
        self.nq = 0
        self.npositiv = 0

def ClusterItems():
    item_vecs, item_bias, user_vecs, user_bias, global_bias, user_vecs_train, user_bias_train = GetData("data")
    kmeans = KMeans(n_clusters=50).fit(item_vecs)
    np.savetxt("data/item_clusters", kmeans.cluster_centers_)

#ClusterItems()
#item_vecs, item_bias, user_vecs, user_bias, global_bias, user_vecs_train, user_bias_train = GetData("data")
#print(item_vecs.shape)
#PrintItemPopularity(user_vecs_train, user_bias_train, item_vecs, item_bias, global_bias, 0.001)