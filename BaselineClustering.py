import sys
import numpy as np
from nltk.cluster.kmeans import KMeansClusterer
from nltk.cluster.util import cosine_distance
from Matrics import *
from dataUtils import *
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--i', type = float)
parser.add_argument('--u', type = int)
parser.add_argument('--f', type = int)
parser.add_argument('--m', type = int)
FLAGS, unparsed = parser.parse_known_args()

#items_names_file = "../../interactive-recomendation/dataset/ml-20m/movies.csv"
#data_dir = "../../GeneratedData1/data" + str(FLAGS.f)
data_dir = "../GeneratedDataAmazone/data" + str(1)
n_points = 30
NUM_CLUSTERS = FLAGS.u

def ClusterItems(data_file, items_bias_file, index_file, clusters_file, centroids_file):

    data = np.genfromtxt(data_file)
    popular_items = np.genfromtxt(index_file).astype('int')
    data = data[popular_items]
    items_bias = np.genfromtxt(items_bias_file)
    important_items = np.where(np.abs(items_bias[popular_items]) < 0.2)[0]
    kclusterer = KMeansClusterer(NUM_CLUSTERS, distance=cosine_distance)
    print("end", data.shape)
    clusters = kclusterer.cluster(data[important_items], assign_clusters=True)
    np.savetxt(centroids_file, kclusterer.means())
    np.savetxt(clusters_file, clusters)

class LasyTree(object):
    def __init__(self, people):
        self.users = people
        self.question = ''
        self.sons = []

def SplitPeople(users, question):
    answers = np.dot(users, question)
    return [np.where(answers > 0)[0], np.where(answers < 0)[0]]


def BuildNewNodes(tree, items, users):
    items_len = items.shape[0]
    IG = 10e+5
    best_question = [0,0]
    best_split = [tree, tree, tree]
    min_n_same_users = 3
    for item in range(items_len):
        for com_item in range(item+1, items_len):
            #print(item, com_item)
            question = items[item] - items[com_item]
            split = SplitPeople(users[tree.users], question)
            split = [tree.users[s] for s in split]
            #print(split)
            lIG = 0.
            if (min(len(split[u]) for u in range(len(split))) < min_n_same_users):
                continue

            for u in range(len(split)):
                us = np.random.choice(split[u], min(100, len(split[u])), replace = False)
                lIG += np.linalg.det(np.cov(users[us], rowvar=False) + 0.01*np.eye(users.shape[1])) * len(split[u])
            lIG /= tree.users.shape[0]
            if (lIG < IG):
                IG = lIG
                best_question = [item, com_item]
                best_split = [LasyTree(s) for s in split]
    tree.sons = best_split
    tree.question = best_question

def RecieveRealItems(question, items, clusters, centroids):
    cluster1 = np.array(clusters[question[0]])
    cluster2 = np.array(clusters[question[1]])
    item1 = np.argmax(np.linalg.norm(items[cluster1] - centroids[0], axis=1))
    item2 = np.argmax(np.linalg.norm(items[cluster2] - centroids[1], axis=1))
    return[cluster1[item1], cluster2[item2]]

def OneStep(tree, centroids, items, items_bias, users, user, clusters):
    question = [0, 0]
    if (len(tree.users) < 100):
        return tree, question
    if (tree.question == ''):
        BuildNewNodes(tree, centroids, users)
    question = tree.question
    important_items = np.where(np.abs(items_bias) < 0.2)[0]
    question = RecieveRealItems(question, items[important_items], clusters, [centroids[question[0]], centroids[question[1]]])
    question[0] = important_items[question[0]]
    question[1] = important_items[question[1]]
    user_answer = receive_answer(user, items[question[0]] - items[question[1]], -1, 1, items_bias[question[0]] - items_bias[question[1]])
    user_answer = np.dot(user, items[question[0]] - items[question[1]]) + items_bias[question[0]] - items_bias[question[1]]
    user_answer = int(user_answer > 0)
    tree = tree.sons[int(user_answer)]
    return tree, question

def AllAlgorithm(users, n_iterations, centroids, items, items_bias, user, clusters, tree):

    tree1 = tree
    questions = []
    for i in range(n_iterations):
        tree1, question = OneStep(tree1, centroids, items, items_bias, users, user, clusters)
        questions.append(question)
    return questions


def RunTesting(centroid_file, clustering_file):
    #print(data_dir)
    item_vecs, item_bias, user_vecs, user_bias, global_bias, user_vecs_train, user_bias_train = GetData(data_dir)
    user_estim = np.mean(user_vecs_train, axis=0)
    #print(item_vecs.shape, user_vecs.shape, user_vecs_train.shape)
    popular_items = np.genfromtxt(data_dir + "/train_ratings.txt_").astype(int)
    non_informative_items = np.where(abs(item_bias) > 0.3)[0]


    mean_dist = [0. for i in range(n_points)]
    NDCG_train = [0. for i in range(n_points)]
    correct_pairs = [0. for i in range(n_points)]
    precision10 = [0. for i in range(n_points)]
    n_users_in_test_ = 1000
    n_users_in_test = 0
    n_NDCG = 0

    tree = LasyTree(np.arange(user_vecs_train.shape[0]))
    centroids = np.genfromtxt(data_dir + '/' + centroid_file)
    clusters_ = np.genfromtxt(data_dir + '/' + clustering_file).astype('int')
    clusters = {}
    for i in range(NUM_CLUSTERS):
        clusters[i] = []
    for i in range(len(clusters_)):
        clusters[clusters_[i]].append(i)



    ratings = np.genfromtxt(data_dir + "/tes_ratings1.txt")
    user_estim_first = np.mean(user_vecs_train, axis=0)
    for i in range(n_users_in_test_):
        sys.stdout.write("\r%d%%" % i)
        if (len(ratings[i].nonzero()[0]) < 10):
            continue

        user_estim = np.mean(user_vecs_train, axis=0)
        questions_item = AllAlgorithm(user_vecs_train, n_points, centroids,
                                      item_vecs[popular_items],
                                      item_bias[popular_items],
                                      user_vecs[i], clusters, tree)

        questions_item = np.array(questions_item).astype(int)
        questions_item = questions_item.T
        questions_item[0] = popular_items[questions_item[0]]
        questions_item[1] = popular_items[questions_item[1]]
        questions_item = questions_item.T
        test_ratings = ratings[i]
        test_ratings[popular_items] = 0
        test_ratings[non_informative_items] = 0

        non_zerro_ratings = np.array(test_ratings.nonzero()[0])

        if (len(ratings[i].nonzero()[0]) < 15):
            continue
        n_users_in_test += 1
        # test

        a = ratings[i][non_zerro_ratings]
        #if (i % 100 == 1):
        #    print(n_users_in_test)
        #    print(" BASELINE = ", np.array(mean_dist) / (n_users_in_test))
        #    print(" BASELINE = ", np.array(correct_pairs) / n_users_in_test)
        #    print(" BASELINE = ", np.array(precision10) / n_users_in_test)

        for j in range(n_points):

            dist_b, ndcg_b, correct_pairs_, precision10_ = Test(questions_item, (j),
                                                                    item_vecs[non_zerro_ratings], item_bias[non_zerro_ratings],
                                                                    user_vecs[i], user_estim_first.copy(), a, item_vecs, item_bias, ratings[i], FLAGS.i, FLAGS.m)

            
            mean_dist[j] += dist_b
            correct_pairs[j] += correct_pairs_
            precision10[j] += precision10_
        sys.stdout.write("\r")
    mean_dist = np.array(mean_dist)
    NDCG_train = np.array(NDCG_train, float)
    correct_pairs = np.array(correct_pairs, float)
    precision10 = np.array(precision10, float)
    print(n_users_in_test)
    print(" BASELINE = ", FLAGS.u, mean_dist / (n_users_in_test))
    print(" BASELINE = ", FLAGS.u, correct_pairs / n_users_in_test)
    print(" BASELINE = ", FLAGS.u, precision10 / n_users_in_test)
    return mean_dist / (n_users_in_test), \
               correct_pairs / n_users_in_test, \
               precision10 / n_users_in_test

ClusterItems(data_dir + "/items1.txt", data_dir + "/items_bias1.txt", data_dir + "/train_ratings.txt_",data_dir + "/clusters", data_dir + "/centroids_file")
RunTesting("centroids_file", "clusters")
