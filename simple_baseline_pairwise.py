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
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--i', type = float)
parser.add_argument('--u', type = int)
parser.add_argument('--f', type = int)
parser.add_argument('--m', type = int)
FLAGS, unparsed = parser.parse_known_args()

data_dir = "../GeneratedData/data" + str(FLAGS.f)

#data_dir = "../GeneratedDataAmazone/data" + str(FLAGS.f)



#n_points = 30
n_points = 30


def GetComparativeItemSimple(user_estim, items, item, questions, answers, items_bias, used_items):
    answers_new = list(answers)
    answers_new.append(-1)
    new_questions = list(questions)
    reshape = False
    if len(new_questions) < 1:
        reshape = True
    new_questions.append(items[item])
    # print(np.array(new_questions).shape)
    new_questions = np.array(new_questions)
    #new_inverse_matrix = np.linalg.inv(np.dot(new_questions.T, new_questions) + 0.001 * np.eye(new_questions.shape[1]))
    #new_inverse_matrix = 0.001 * np.eye(new_questions.shape[1])
    #new_user = np.dot(new_inverse_matrix,
    #                  np.dot(np.array(answers_new - items_bias[item]), np.array(new_questions)))
    new_user = UpdateUser(user_estim.copy(), items[item], items_bias[item], -1., FLAGS.i)
    # picking the comparative item
    comparative_item = BestItem(items, new_user, used_items, items_bias, 0.2)
    return comparative_item


def OneStep(user_estim, user, items, used_items, answers, questions, items_bias, users_ratings, baseline=True):
    n_latent_factors = len(user)
    # for picking best item
    item = BestItem(items, user_estim, used_items, items_bias, 0.2)
    used_items.append(item)

    #make virtual step


    #it = items[item].reshape(items.shape[1], 1)
    #v = np.dot(it, it.T)
    #new_inverse_matrix = GetInverseMatrix(inverse_matrix, v)


    #print (np.array(new_questions).shape)
    if (baseline):
        comparative_item = GetComparativeItemSimple(user_estim, items, item, questions, answers, items_bias, used_items)
    used_items.append(comparative_item)

    user_answer = receive_answer(user, items[item] - items[comparative_item], -2, 2,
                                 items_bias[item] - items_bias[comparative_item])
    if (FLAGS.m == 1):
        user_answer = trueAnswer(users_ratings, item, comparative_item)
    answers.append(user_answer - items_bias[item] + items_bias[comparative_item])

    it = items[item]
    questions.append(items[item] - items[comparative_item])
    questions1 = np.array(questions)
    #if (reshape):
    #    questions1 = questions1.reshape(1, questions1.shape[0])

    #com_it = items[comparative_item].reshape([items.shape[1], 1])
    #v = np.dot(it-com_it, (it - com_it).T)
    #new_inverse_matrix1 = GetInverseMatrix(inverse_matrix, v)

    #new_inverse_matrix = np.linalg.inv(np.dot(questions1.T, questions1) + 0.001 * np.eye(questions1.shape[1]))
    #d = np.max(new_inverse_matrix - new_inverse_matrix1)
    #print (np.max(d))
    user_estim = UpdateUser(user_estim.copy(), questions[-1], items_bias[item] - items_bias[comparative_item], answers[-1], FLAGS.i)

    #user_estim = np.dot(new_inverse_matrix,
    #                     np.dot(np.array(answers), np.array(questions)))
    return user_estim, answers, used_items, questions, [item, comparative_item]

def AllAlgorithm(items, user, user_estim, n_questions, items_bias, users_ratings, baseline = True):
    inverse_matrix = np.linalg.inv(np.eye(user.shape[0]) * 0.001)
    answers = []
    used_items = []
    questions = []
    questions_item = []
    for q in range(n_questions):
        user_estim, answers, used_items, questions, item_question = OneStep(user_estim, user,
                                                                  items, used_items, answers, questions,
                                                                             items_bias, users_ratings, baseline)
        questions_item.append(item_question)
    dif = user - user_estim
    return answers, questions, used_items, questions_item

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
    def __init__(self):
       self.RecieveQuestions = AllAlgorithm

def main():
    #dir = "data"
    print(data_dir)
    item_vecs, item_bias, user_vecs, user_bias, global_bias, user_vecs_train, user_bias_train = GetData(data_dir)
    user_estim = np.mean(user_vecs_train, axis = 0)
    print(item_vecs.shape, user_vecs.shape,user_vecs_train.shape)
    popular_items = np.genfromtxt(data_dir + "/train_ratings.txt_").astype(int)
    
    non_informative_items = np.where(abs(item_bias) > 0.3)[0]

    mean_dist = [0. for i in range(n_points+2)]
    NDCG_train = [0. for i in range(n_points+2)]
    my_NDCG = [0. for i in range(n_points+2)]
    correct_pairs = [0. for i in range(n_points+2)]
    my_correct_pairs = [0. for i in range(n_points+2)]
    my_mean_dist = [0. for i in range(n_points+2)]
    precision10 = [0. for i in range(n_points+2)]
    my_precision10 = [0. for i in range(n_points+2)]
    n_users_in_test_ = 1000
    n_users_in_test = 1e-10
    n_users_in_test_ndcg = 1e-10
    n_NDCG = 0

    ratings = np.genfromtxt(data_dir + "/tes_ratings1.txt")

    #ratings = ratings.T
    #ratings[non_informative_items] = 0
    #ratings = ratings.T
    #ratings = ratings * (item_bias < 0.2) * (item_bias > -0.2)

    #FirstAlgorithm(40, 10, item_vecs, item_bias, dir)
    user_estim_first = np.mean(user_vecs_train, axis=0)

        
    #my_questions_item_ = np.genfromtxt(data_dir +  "/questions_items")
    my_questions_item_ = np.genfromtxt(data_dir +  "/questions_items_user_specific")
    for i in range(n_users_in_test_):
        sys.stdout.write("\r%d%%" % i)

        #my estimation
        my_questions_item = my_questions_item_
        if (FLAGS.m == 1):
             my_questions_item = my_questions_item_[i].reshape(my_questions_item_[i].shape[0] / 2, 2)
         


        if (len(ratings[i][popular_items].nonzero()[0]) < 20):
            continue
   

        user_estim = np.mean(user_vecs_train, axis = 0)
        
        if (FLAGS.m == 1):
            u_r = np.array(ratings[i][popular_items].nonzero()[0])
        if (FLAGS.m == 0):
            u_r = np.arange(popular_items.shape[0])

        answers, questions,used_items,questions_item = AllAlgorithm(item_vecs[popular_items[u_r]], user_vecs[i],
                                          user_estim, n_points, item_bias[popular_items[u_r]], ratings[i][popular_items[u_r]])
        questions_item = np.array(questions_item).astype(int)
        questions_item = questions_item.T
        #questions_item[0] = popular_items[questions_item[0]]
        #questions_item[1] = popular_items[questions_item[1]]
        questions_item[0] = popular_items[u_r[questions_item[0]]]
        questions_item[1] = popular_items[u_r[questions_item[1]]]
        questions_item = questions_item.T
        #answers_mine, my_questions, my_used_items, my_questions_item = AllAlgorithm(item_vecs, user_vecs[i],
        #                                  user_estim, n_points, item_bias, False)


        test_ratings = ratings[i].copy()
        #test_ratings[used_items] = 0

        my_questions_item = np.array(my_questions_item).astype(int)
        my_questions_item = my_questions_item.T
        my_questions_item[0] = popular_items[my_questions_item[0]]
        my_questions_item[1] = popular_items[my_questions_item[1]]
        my_questions_item = my_questions_item.T

        test_ratings[popular_items] = 0
        test_ratings[non_informative_items] = 0
        #test_ratings[my_questions_item.T[0]] = 0
        #test_ratings[my_questions_item.T[1]] = 0
        non_zerro_ratings = np.array(test_ratings.nonzero()[0])
        #print(non_zerro_ratings.shape)
        if (non_zerro_ratings.shape[0] < 15):
            continue
        #if (len(ratings[i].nonzero()[0]) < 10):
        #    continue
        n_users_in_test += 1
        #test
        """questions = (item_vecs[questions_item.T[0]] - item_vecs[questions_item.T[1]])
        my_questions = (item_vecs[my_questions_item.T[0]] - item_vecs[my_questions_item.T[1]])
        my_norm = [np.dot(q, q.T) for q in my_questions]
        print(my_norm)
        print(np.trace(np.linalg.inv(np.dot(questions[:1].T, questions[:1]) +
                                     0.01 * np.eye(questions.shape[1]))))
        print(np.trace(np.linalg.inv(np.dot(my_questions[:1].T, my_questions[:1]) +
                                     0.01 * np.eye(my_questions.shape[1]))))
        """
        """answers = np.dot(questions, user_vecs[i])
        my_answers = np.dot(my_questions, user_vecs[i])
        user_estim = np.dot(np.linalg.inv(np.dot(questions.T, questions) + 0.00 * np.eye(questions.shape[1])), np.dot(answers, questions))
        my_user_estim = np.dot(np.linalg.inv(np.dot(my_questions.T, my_questions) + 0.00 * np.eye(my_questions.shape[1])), np.dot(my_answers, my_questions))
        print('BASELINE =', 1 - abs(np.dot(user_vecs[i], user_estim) / math.sqrt(np.dot(user_vecs[i], user_vecs[i]) * np.dot(user_estim, user_estim))))
        print(1 - abs(np.dot(user_vecs[i], my_user_estim) / math.sqrt(np.dot(user_vecs[i], user_vecs[i]) * np.dot(my_user_estim, my_user_estim))))
        print(answers)
        print(my_answers)"""
    
        a = ratings[i][non_zerro_ratings]
        if (i % 100 == 1):
            print(n_users_in_test)
            print(" BASELINE DIST = ", np.array(mean_dist) / (n_users_in_test))
            print(" MY DIST = ", np.array(my_mean_dist) / (n_users_in_test))
            print(" BASELINE NDCG = ", np.array(NDCG_train) / (n_users_in_test_ndcg))
            print(" MY NDCG = ", np.array(my_NDCG) / (n_users_in_test_ndcg))
            print(" BASELINE CP = ", np.array(correct_pairs) / n_users_in_test)
            print(" MY CP = ", np.array(my_correct_pairs) / n_users_in_test)
            print(" BASELINE P10 = ", np.array(precision10) / n_users_in_test)
            print(" MY P10 = ", np.array(my_precision10) / n_users_in_test)

        for j in range(n_points + 2):
            
            dist_b, ndcg_b, correct_pairs_, precision10_ = Test(questions_item, (j),
                                                  item_vecs[non_zerro_ratings], item_bias[non_zerro_ratings],
                                                  user_vecs[i], user_estim_first.copy(), a, item_vecs, item_bias, ratings[i], FLAGS.i, FLAGS.m)           
            dist_b_my, ndcg_b_my, correct_pairs_my_, my_precision10_ = Test(my_questions_item, (j),
                                                           item_vecs[non_zerro_ratings], item_bias[non_zerro_ratings],
                                                           user_vecs[i], user_estim_first.copy(), a, item_vecs, item_bias, ratings[i], FLAGS.i, FLAGS.m)

            mean_dist[j] += dist_b
            correct_pairs[j] += correct_pairs_
            precision10[j] += precision10_
            my_mean_dist[j] += dist_b_my
            my_correct_pairs[j] += correct_pairs_my_
            my_precision10[j] += my_precision10_
            
            """
            for i1 in range(10):
                serp = GetSERP(a)
                if max(serp) > 0:
                    if (j == 0 and i1 == 0):
                        n_NDCG += 1
                    dist_b,ndcg_b, correct_pairs_, precision10_ = Test(questions_item, (j),
                        item_vecs[non_zerro_ratings[serp]], item_bias[non_zerro_ratings[serp]],
                        user_vecs[i], user_estim_first.copy(), a[serp], item_vecs, item_bias, FLAGS.i)
                    dist_b_my, ndcg_b_my, correct_pairs_my_, my_precision10_ = Test(my_questions_item, (j),
                                    item_vecs[non_zerro_ratings[serp]], item_bias[non_zerro_ratings[serp]],
                                    user_vecs[i], user_estim_first.copy(), a[serp], item_vecs, item_bias, FLAGS.i)
                    NDCG_train[j] += ndcg_b / 10.
                    my_NDCG[j] += ndcg_b_my / 10.
                    n_users_in_test_ndcg += 1. / (10 * n_points)
            """
            #test all
        sys.stdout.write("\r")
    mean_dist = np.array(mean_dist)
    my_mean_dist = np.array(my_mean_dist)
    NDCG_train = np.array(NDCG_train, float)
    my_NDCG = np.array(my_NDCG, float)
    correct_pairs = np.array(correct_pairs, float)
    my_correct_pairs = np.array(my_correct_pairs, float)
    precision10 = np.array(precision10, float)
    my_precision10 = np.array(my_precision10, float)
    print(n_users_in_test)
    print(" BASELINE = ", mean_dist / (n_users_in_test))
    print(" MY = ", my_mean_dist / (n_users_in_test))
    print(" BASELINE = ", NDCG_train / (n_users_in_test_ndcg))
    print(" MY = ", my_NDCG / (n_users_in_test_ndcg))
    print(" BASELINE = ", correct_pairs / n_users_in_test)
    print(" MY = ", my_correct_pairs/ n_users_in_test)
    print(" BASELINE = ", precision10 / n_users_in_test)
    print(" MY = ", my_precision10/ n_users_in_test)
    return mean_dist / (n_users_in_test), my_mean_dist / (n_users_in_test), NDCG_train / (n_users_in_test_ndcg), \
    my_NDCG / (n_users_in_test_ndcg), correct_pairs / n_users_in_test, my_correct_pairs/ n_users_in_test, \
    precision10 / n_users_in_test, my_precision10 / n_users_in_test




    #print(mean_dist / n_users_in_test)
#main("data0")

mean_dist = [[] for i in range(n_points)]
my_mean_dist = [[] for i in range(n_points)]
NDCG_train = [[] for i in range(n_points)]
my_NDCG = [[] for i in range(n_points)]
correct_pairs  = [[] for i in range(n_points)]
my_correct_pairs  = [[] for i in range(n_points)]
precision10  = [[] for i in range(n_points)]
my_precision10  = [[] for i in range(n_points)]
for i in range(1):
    if (i == 5):
        continue
    mean_dist_,my_mean_dist_,NDCG_train_,my_NDCG_,correct_pairs_,my_correct_pairs_, \
        precision10_, my_precision10_ = main()
    for i in range(n_points):
        mean_dist[i].append(mean_dist_[i])
        my_mean_dist[i].append(my_mean_dist_[i])
        NDCG_train[i].append(NDCG_train_[i])
        my_NDCG[i].append(my_NDCG_[i])
        correct_pairs[i].append(correct_pairs_[i])
        my_correct_pairs[i].append(my_correct_pairs_[i])
        precision10[i].append(precision10_[i])
        my_precision10[i].append(my_precision10_[i])


mean_dist = np.array(mean_dist)
my_mean_dist = np.array(my_mean_dist)
NDCG_train = np.array(NDCG_train)
my_NDCG = np.array(my_NDCG)
correct_pairs  = np.array(correct_pairs)
my_correct_pairs  = np.array(my_correct_pairs)
precision10 = np.array(precision10)
my_precision10 = np.array(my_precision10)

np.savetxt(data_dir + "/mean_dist1", mean_dist)
np.savetxt(data_dir + "/my_mean_dist1", my_mean_dist)
np.savetxt(data_dir + "/NDCG1", NDCG_train)
np.savetxt(data_dir + "/my_NDCG1", my_NDCG)
np.savetxt(data_dir + "/correct_pairs1", correct_pairs)
np.savetxt(data_dir + "/my_correct_pairs1", my_correct_pairs)
np.savetxt(data_dir + "/precision101", precision10)
np.savetxt(data_dir + "/my_precision101", my_precision10)


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
