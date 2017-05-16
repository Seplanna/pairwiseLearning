import sys
from dataUtils import *
from Matrics import *
#from PairwiseRecommendation import FirstAlgorithm
#from PairwiseRecommendation import GetComparativeItem
#import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp
"""
Relative algorithm from Towards Conversational Recommender Systems
"""
#import argparse
#parser = argparse.ArgumentParser()
#parser.add_argument('--i', type = float)
#parser.add_argument('--u', type = int)
#parser.add_argument('--f', type = int)
#parser.add_argument('--m', type = int)
#FLAGS, unparsed = parser.parse_known_args()

#data_dir = "../GeneratedData/data" + str(FLAGS.f)

#data_dir = "../GeneratedDataAmazone/data" + str(FLAGS.f)



#n_points = 30
n_points = 30


def GetComparativeItemSimple(user_estim, items, item, questions, answers, items_bias, used_items, learning_rate):
    answers_new = list(answers)
    answers_new.append(-1)
    new_questions = list(questions)
    reshape = False
    if len(new_questions) < 1:
        reshape = True
    new_questions.append(items[item])
    new_user = UpdateUser(user_estim.copy(), items[item], items_bias[item], -1., learning_rate)
    # picking the comparative item
    comparative_item = BestItem(items, new_user, used_items, items_bias, 0.2)
    return comparative_item


def OneStep(user_estim, user, items, used_items, answers, questions, items_bias, users_ratings, learning_rate, mode):
    n_latent_factors = len(user)
    # for picking best item
    item = BestItem(items, user_estim, used_items, items_bias, 0.2)
    used_items.append(item)

    #make virtual step
    comparative_item = GetComparativeItemSimple(user_estim, items, item, questions, answers, items_bias, used_items, learning_rate)
    used_items.append(comparative_item)
    
    #update user
    user_answer = receive_answer(user, items[item] - items[comparative_item], -2, 2,
                                 items_bias[item] - items_bias[comparative_item])
    if (mode == 1):
        user_answer = trueAnswer(users_ratings, item, comparative_item)
    answers.append(user_answer - items_bias[item] + items_bias[comparative_item])

    it = items[item]
    questions.append(items[item] - items[comparative_item])
    questions1 = np.array(questions)

    user_estim = UpdateUser(user_estim.copy(), questions[-1], items_bias[item] - items_bias[comparative_item], answers[-1], learning_rate)

    return user_estim, answers, used_items, questions, [item, comparative_item]

def AllAlgorithm(items, user, user_estim, n_questions, items_bias, users_ratings, learning_rate, mode):
    inverse_matrix = np.linalg.inv(np.eye(user.shape[0]) * 0.001)
    answers = []
    used_items = []
    questions = []
    questions_item = []
    for q in range(n_questions):
        user_estim, answers, used_items, questions, item_question = OneStep(user_estim, user,
                                                                  items, used_items, answers, questions,
                                                                             items_bias, users_ratings, learning_rate, mode)
        questions_item.append(item_question)
    dif = user - user_estim
    return questions_item

import pandas as pd

def GetData1(directory, users):
    names = ['user_id', 'item_id', 'rating', 'timestamp']
    df = pd.read_csv(directory + "/" + 'u.data', sep='\t', names=names)
    n_users = df.user_id.unique().shape[0]
    n_items = df.item_id.unique().shape[0]
    ratings = np.zeros((n_users, n_items))

    for row in df.itertuples():
        ratings[row[1]-1, row[2]-1] = (int(row[3] > 3.5) - 0.5) * 2

    ratings2 = ratings[users]
    #filter items
    ratings1 = ratings2.T
    items = []
    for i in xrange(ratings1.shape[0]):
        #print(ratings1[i].nonzero(), len(ratings1[i].nonzero()[0]))
        if len(ratings1[i].nonzero()[0]) > 10:
            items.append(i)
    ratings = ratings.T[items]
    ratings = ratings.T

    print (str(n_users) + ' users')
    print (str(n_items) + ' items')
    sparsity = float(len(ratings.nonzero()[0]))
    sparsity /= (ratings.shape[0] * ratings.shape[1])
    sparsity *= 100
    print ('Sparsity: {:4.2f}%'.format(sparsity))
    return ratings

class BanditBaseline(object):
    def __init__(self, mode, learning_rate):
       self.learning_rate = learning_rate
       self.mode = mode
    def RecieveQuestions(self, item_vecs, user, user_estim, n_points, item_bias, ratings):
           return AllAlgorithm(item_vecs, user, user_estim, n_points, item_bias, ratings, self.learning_rate, self.mode)


def GetResultsFromFiles():
    mean_dist = [[] for i in range(n_points)]
    my_mean_dist = [[] for i in range(n_points)]
    NDCG_train = [[] for i in range(n_points)]
    my_NDCG = [[] for i in range(n_points)]
    correct_pairs = [[] for i in range(n_points)]
    my_correct_pairs = [[] for i in range(n_points)]
    precision10 = [[] for i in range(n_points)]
    my_precision10 = [[] for i in range(n_points)]
    for j in range(9):
        if (j == 3):
            continue
        i = j
        print(data_dir[:-1] + str(i+1) + "/mean_dist1")
        mdl = np.genfromtxt(data_dir[:-1] + str(i+1) + "/mean_dist1")
        mmdl = np.genfromtxt(data_dir[:-1] + str(i+1) + "/my_mean_dist1")
        NDCGl = np.genfromtxt(data_dir[:-1] + str(i+1) +"/NDCG1")
        mNDCGL = np.genfromtxt(data_dir[:-1] + str(i+1) + "/my_NDCG1")
        cpl = np.genfromtxt(data_dir[:-1] + str(i+1) +"/correct_pairs1")
        mcpl = np.genfromtxt(data_dir[:-1] + str(i+1) + "/my_correct_pairs1")
        p10 = np.genfromtxt(data_dir[:-1] + str(i+1) + "/precision101")
        mp10 = np.genfromtxt(data_dir[:-1] + str(i + 1) + "/my_precision101")
        for p in range(n_points):
            mean_dist[p].append(mdl[p])
            my_mean_dist[p].append(mmdl[p])
            NDCG_train[p].append(NDCGl[p])
            my_NDCG[p].append(mNDCGL[p])
            correct_pairs[p].append(cpl[p])
            my_correct_pairs[p].append(mcpl[p])
            precision10[p].append(p10[p])
            my_precision10[p].append(mp10[p])
    return np.array(mean_dist), np.array(my_mean_dist), np.array(NDCG_train), \
           np.array(my_NDCG), np.array(correct_pairs), np.array(my_correct_pairs), \
           np.array(precision10), np.array(my_precision10)



def Plot(mean_dist, my_mean_dist, NDCG_train, my_NDCG, correct_pairs, my_correct_pairs, precision10, my_precision10):
    """mean_dist = np.genfromtxt(data_dir + "/mean_dist")
    my_mean_dist = np.genfromtxt(data_dir + "/my_mean_dist")
    #random_mean_dist = np.genfromtxt("random_mean_dist")
    NDCG_train = np.genfromtxt(data_dir + "/NDCG")
    my_NDCG = np.genfromtxt(data_dir + "/my_NDCG")
    #random_NDCG = np.genfromtxt("random_NDCG")
    correct_pairs = np.genfromtxt(data_dir + "/correct_pairs")
    my_correct_pairs = np.genfromtxt(data_dir + "/my_correct_pairs")
    #random_correct_pairs = np.genfromtxt("random_correct_pairs")"""
    collection = [mean_dist, my_mean_dist, #random_mean_dist,
                  NDCG_train, my_NDCG, #random_NDCG,
                  correct_pairs, my_correct_pairs,
                  precision10, my_precision10]#, random_correct_pairs]
    #print(precision10)

    means = []
    variace = []
    mins = []
    maxs = []
    for i in range(len(collection)):
        print(i)
        means.append(np.mean(collection[i], axis = 1))
        variace.append(np.var(collection[i], axis = 1))
        mins.append(np.min(collection[i], axis = 1))
        maxs.append(np.max(collection[i], axis = 1))
    names = [data_dir + "/mean_dist.png", data_dir + "/NDCG.png", data_dir + "/correct_pairs.png",
             data_dir + "/precision10.png"]
    means = np.array(means)
    variace = np.array(variace)
    mins = np.array(mins)
    maxs = np.array(maxs)
    mins = means - mins
    maxs = maxs - means
    for i in range(4):
        for j in range(n_points):
            t_statistic, p_value = ttest_1samp(collection[2*i][j] - collection[2*i+1][j], 0)
            print(p_value)
        fig = plt.figure()
        #ax = fig.add_subplot(111)
        #ax.errorbar(np.arange(n_points),means[2*i], yerr= [mins[2*i], maxs[2*i]],
        #        fmt='o', ecolor='g', capthick=2, color='b' )
        #ax.errorbar(np.arange(n_points),np.array(means[2*i+1]), yerr= [mins[2*i+1], maxs[2*i + 1]],
        #        fmt='v', ecolor='b', capthick=2, color = 'r')
        plt.plot(np.arange(n_points-1),means[2*i][1:], 'bo',np.arange(n_points-1),means[2*i][1:], 'k')
        plt.plot(np.arange(n_points-1), means[2 * i + 1][1:], 'g^', np.arange(n_points-1), means[2 * i + 1][1:], 'k')
        plt.plot(np.arange(n_points - 1), [means[2 * i + 1][0] for j in range(n_points-1)], 'r*')
        #ax.errorbar(np.arange(n_points), np.array(means[3 * i + 2]), yerr=[mins[3*i+2], maxs[3*i+2]],
        #        fmt='x', ecolor='y', capthick=2, color='k')
        plt.grid()
        fig.savefig(names[i])

#mean_dist, my_mean_dist, NDCG_train, my_NDCG, \
#correct_pairs, my_correct_pairs, \
#precision10, my_precision10 = GetResultsFromFiles()

#Plot(mean_dist, my_mean_dist, NDCG_train, my_NDCG, correct_pairs, my_correct_pairs, precision10, my_precision10)
