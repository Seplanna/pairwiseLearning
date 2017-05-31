import sys
from dataUtils import *
from Matrics import *
#from PairwiseRecommendation import FirstAlgorithm
#from PairwiseRecommendation import GetComparativeItem
#import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp
from simple_baseline_pairwise import BanditBaseline
from StaticFromFile import *
from PairwiseRecommendation import StaticPairwiseSmart 
from PairwiseRecommendation import StaticPairwiseRandom
from BaselineClustering import ClusteringPairwise
from BaselineYahoo import Yahoo
from Plot import AllResultsForOneMethod
from Plot import Plot_All
  
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--i', type = float)
parser.add_argument('--u')
parser.add_argument('--f', type = int)
parser.add_argument('--m', type = int)
FLAGS, unparsed = parser.parse_known_args()


#data_dir = "../GeneratedData/data" + str(FLAGS.f)
#results_dir = "../GeneratedData/results/"

data_dir = "../GeneratedDataAmazone/data"+ str(FLAGS.f)
results_dir = "../GeneratedDataAmazone/results/"



#n_points = 30
n_points = 30

def GetItemsForTest(user, items, item_bias, n_samples):
    truth = np.dot(user, items.T) + item_bias
    item_list = np.argsort(-truth)
    step = items.shape[0] / n_samples
    result = []
    for i in range(n_samples):
        result.append(np.random.choice(item_list[step*i:step*(i+1)], 1)[0])
    return result



def TestAllData(method, mode):
    #dir = "data"
    print(data_dir)
    item_vecs, item_bias, user_vecs, user_bias, global_bias, user_vecs_train, user_bias_train = GetData(data_dir)
    print(item_vecs.shape, np.zeros((item_vecs.shape[1], 1)).shape)
    item_vecs = np.concatenate((item_vecs, np.zeros((1,item_vecs.shape[1]))), axis=0)
    item_bias = np.append(item_bias,[0])
    print(item_vecs.shape, user_vecs.shape,user_vecs_train.shape, item_bias.shape)
    popular_items = np.genfromtxt(data_dir + "/train_ratings.txt_").astype(int)
    
    non_popular_items = np.array([i for i in range(item_vecs.shape[0]) if i not in popular_items])
    
    non_informative_items = np.where(abs(item_bias) > 0.3)[0]

    mean_dist = [0. for i in range(n_points+2)]
    NDCG_train = [0. for i in range(n_points+2)]
    correct_pairs = [0. for i in range(n_points+2)]
    precision10 = [0. for i in range(n_points+2)]
    
    n_users_in_test_ = 1000
    n_users_in_test = 1e-10
    n_users_in_test_ndcg = 1e-10

    ratings = np.genfromtxt(data_dir + "/tes_ratings1.txt")
    print(ratings.shape)
    ratings = np.concatenate((ratings, np.zeros((ratings.shape[0], 1))), axis=1)
    print(ratings.shape)
    user_estim_first = np.mean(user_vecs_train, axis=0)
        
    for i in range(n_users_in_test_):
        if (len(ratings[i][popular_items].nonzero()[0]) < 20):
            continue
        

        if (mode == 1):
            u_r = np.array(ratings[i][popular_items].nonzero()[0])
        if (mode == 0):
            u_r = np.arange(popular_items.shape[0])
        questions_item = method.RecieveQuestions(item_vecs[popular_items[u_r]], user_vecs[i],
                                          user_estim_first.copy(), n_points, item_bias[popular_items[u_r]], ratings[i][popular_items[u_r]])
        questions_item = np.array(questions_item).astype(int)
        questions_item = questions_item.T
        questions_item[0] = popular_items[u_r[questions_item[0]]]
        # for absolute question the comparative item always -1.
        if (np.max(questions_item[1]) >= 0):
            questions_item[1] = popular_items[u_r[questions_item[1]]]
        questions_item = questions_item.T

        test_ratings = ratings[i].copy()
        test_ratings[popular_items] = 0        
        test_ratings[non_informative_items] = 0
        non_zerro_ratings = np.array(test_ratings.nonzero()[0])
        if (non_zerro_ratings.shape[0] < 15):
            continue
        n_users_in_test += 1
        #test
        if (mode == 0):
            non_zerro_ratings = GetItemsForTest(user_vecs[i], item_vecs[non_popular_items], item_bias[non_popular_items], 50)


        a = ratings[i][non_zerro_ratings]
        if (i % 10 == 1):
            print(n_users_in_test)
            print(" BASELINE DIST = ", np.array(mean_dist) / (n_users_in_test))
            print(" BASELINE CP = ", np.array(correct_pairs) / n_users_in_test)
            print(" BASELINE P10 = ", np.array(precision10) / n_users_in_test)
        
        
        for j in range(n_points + 2):
             
            dist_b, ndcg_b, correct_pairs_, precision10_ = Test(questions_item, (j),
                                                  item_vecs[non_zerro_ratings], item_bias[non_zerro_ratings],
                                                  user_vecs[i], user_estim_first.copy(), a, item_vecs, item_bias, ratings[i], FLAGS.i, mode)           

            mean_dist[j] += dist_b
            correct_pairs[j] += correct_pairs_
            precision10[j] += precision10_
        sys.stdout.write("\r")
    mean_dist = np.array(mean_dist)
    correct_pairs = np.array(correct_pairs, float)
    precision10 = np.array(precision10, float)
    print(n_users_in_test)
    print(" BASELINE = ", mean_dist / (n_users_in_test))
    print(" BASELINE = ", correct_pairs / n_users_in_test)
    print(" BASELINE = ", precision10 / n_users_in_test)
    return mean_dist / (n_users_in_test), \
    correct_pairs / n_users_in_test,\
    precision10 / n_users_in_test


def main(method, mode, result_files):
    mean_dist = [[] for i in range(n_points)]
    correct_pairs  = [[] for i in range(n_points)]
    precision10  = [[] for i in range(n_points)]
    mean_dist_,correct_pairs_,precision10_ = TestAllData(method, mode)
    for i in range(n_points):
        mean_dist[i].append(mean_dist_[i])
        correct_pairs[i].append(correct_pairs_[i])
        precision10[i].append(precision10_[i])


    mean_dist = np.array(mean_dist)
    correct_pairs  = np.array(correct_pairs)
    precision10 = np.array(precision10)

    np.savetxt(data_dir + "/" + result_files + "_mean_dist", mean_dist)
    np.savetxt(data_dir + "/" + result_files + "_correct_pairs", correct_pairs)
    np.savetxt(data_dir + "/" + result_files + "_precision10", precision10)


def RunTest(method):
    if (method == 'bandits'):
        main(BanditBaseline(FLAGS.m, FLAGS.i), 0, "bandits")
    if (method == 'my'):
        main(StaticPairwiseQuestions(data_dir +  "/questions_items"), 0, "my")
    if (method == 'random'):
        main(StaticPairwiseQuestions(data_dir +  "/random_questions_items"), 0, "random")
    if (method == 'yahoo'):
        main(StaticAbsoluteQuestions(data_dir + "/yahoo_items"), 0, "yahoo")
    if (method == 'IG'):
        main(ClusteringPairwise(data_dir + "/users_train.txt", data_dir + "/centroids_file", data_dir + "/clusters",80, n_points, 0), 0, "IG")
    if (method == 'my_real'):
        main(StaticPairwiseSmart(n_points, 5), 1, "static_pairwise_smart_per_user")
    if (method == 'random_real'):
        main(StaticPairwiseRandom(n_points), 1, "static_pairwise_random_per_user")
    if (method == 'bandits_real'):
        main(BanditBaseline(FLAGS.m, FLAGS.i), 1, "bandits_per_user")
    if (method == 'yahoo_real'):
        main(Yahoo(n_points), 1, "yahoo_per_user")
    if (method == 'IG_real'):
        main(ClusteringPairwise(data_dir + "/users_train.txt", data_dir + "/centroids_file", data_dir + "/clusters",80, n_points, 1), 1, "IG_real")
RunTest(FLAGS.u)
#main(ClusteringPairwise(data_dir + "/users_train.txt", data_dir + "/centroids_file", data_dir + "/clusters", 30, n_points), "interactive_clustering_pairwise")

#AllResultsForOneMethod(n_points, data_dir, "bandits", 9, results_dir)
#AllResultsForOneMethod(n_points, data_dir, "my", 9, results_dir)
#AllResultsForOneMethod(n_points, data_dir, "random", 9, results_dir)
#AllResultsForOneMethod(n_points, data_dir, "yahoo", 9, results_dir)
#AllResultsForOneMethod(n_points, data_dir, "IG", 9, results_dir)

#AllResultsForOneMethod(n_points, data_dir, "bandits_per_user", 9, results_dir, 'bandits_real')
#AllResultsForOneMethod(n_points, data_dir, "static_pairwise_smart_per_user", 9, results_dir, 'my_real')
#AllResultsForOneMethod(n_points, data_dir, "static_pairwise_random_per_user", 9, results_dir, 'random_real')
#AllResultsForOneMethod(n_points, data_dir, "yahoo_per_user", 9, results_dir, 'yahoo_real')
#Plot_All("../GeneratedData/results/", ["bandits", "my", "random", "yahoo"], "mean_dist")

