import sys
import numpy as np
from nltk.cluster.kmeans import KMeansClusterer
from nltk.cluster.util import cosine_distance
from Matrics import *
from dataUtils import *
import argparse


#parser = argparse.ArgumentParser()
#parser.add_argument('--i', type = float)
#parser.add_argument('--u', type = int)
#parser.add_argument('--f', type = int)
#parser.add_argument('--m', type = int)
#FLAGS, unparsed = parser.parse_known_args()

#items_names_file = "../../interactive-recomendation/dataset/ml-20m/movies.csv"
#data_dir = "../../GeneratedData1/data" + str(FLAGS.f)
#data_dir = "../GeneratedData/data" + str(FLAGS.f)
#n_points = 30
#NUM_CLUSTERS = FLAGS.u

def ClusterItems(data_file, items_bias_file, index_file, clusters_file, centroids_file):

    data = np.genfromtxt(data_file)
    popular_items = np.genfromtxt(index_file).astype('int')
    data = data[popular_items]
    items_bias = np.genfromtxt(items_bias_file)
    important_items = np.where(np.abs(items_bias[popular_items]) < 0.2)[0]
    kclusterer = KMeansClusterer(NUM_CLUSTERS, distance=cosine_distance)
    print(NUM_CLUSTERS, important_items.shape)
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
    return [np.where(answers > 0.01)[0], np.where(answers < -0.01)[0]]

def GetIG(split, users):
    lIG = 0.
    n_users = 0
    for u in range(len(split)):
        n_users += len(split[u])
        us = np.random.choice(split[u], min(200, len(split[u])), replace = False)
        lIG += np.linalg.det(np.cov(users[us], rowvar=False)) * len(split[u])
    lIG /= n_users
    return lIG

def GetIGVariance(split, users):
    lIG = 0.
    n_users = 0
    for u in range(len(split)):
        n_users += len(split[u])
        lIG += np.sum(np.var(split[u], axis=0)) * len(split[u])
    lIG /= n_users
    return lIG

def BuildNewNodes(tree, items, users):
    items_len = items.shape[0]
    IG = 10e+10
    best_question = [0,0]
    best_split = [tree, tree, tree]
    min_n_same_users = 1
    best_lens = []
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

            lIG = GetIGVariance(split, users)
            #lIG = GetIG(split, users)
            #lIG = 1. / (item+com_item) 
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
    if (len(tree.users) < 10):
        return tree, question
    if (tree.question == ''):
        BuildNewNodes(tree, centroids, users)
    question = tree.question
    question = RecieveRealItems(question, items, clusters, [centroids[question[0]], centroids[question[1]]])
    user_answer = receive_answer(user, items[question[0]] - items[question[1]], -1, 1, items_bias[question[0]] - items_bias[question[1]])
    user_answer = np.dot(user, items[question[0]] - items[question[1]]) + items_bias[question[0]] - items_bias[question[1]]
    user_answer = int(user_answer > 0)
    tree = tree.sons[int(user_answer)]
    return tree, question

def AllAlgorithm(users, n_iterations, centroids, items, items_bias, user, clusters, tree):

    tree1 = tree
    questions = []
    last_q_random = False
    for i in range(n_iterations):
        question = [0, 0]
        if(last_q_random == False):
            tree1, question = OneStep(tree1, centroids, items, items_bias, users, user, clusters)
        if (last_q_random == True or question[0] == question[1]):
            question[0] = np.random.randint(0, items.shape[0], 1)
            question[1] = np.random.randint(0, items.shape[0], 1)
        questions.append(question)
    return questions

class ClusteringPairwise():
    def __init__(self, users_vecs_train_file, centroid_file, clustering_file, num_clusters, n_iteration):
       self.num_clusters = num_clusters
       self.users = np.genfromtxt(users_vecs_train_file)
       self.tree = LasyTree(np.arange(self.users.shape[0]))
       self.centroids = np.genfromtxt(centroid_file)
       clusters_ = np.genfromtxt(clustering_file).astype('int')
       self.clusters = {}
       for i in range(num_clusters):
           self.clusters[i] = []
       for i in range(len(clusters_)):
           self.clusters[clusters_[i]].append(i)
       self.n_iteration = n_iteration
       self.kclusterer = KMeansClusterer(num_clusters, distance=cosine_distance, initial_means=list(self.centroids))

    def RecieveQuestions(self, item_vecs, user, user_estim, n_points, item_bias, ratings):
        clusters_ = self.kclusterer.cluster(item_vecs, assign_clusters=True)
        clusters = {}
        for i in range(self.num_clusters):
           clusters[i] = []
        for i in range(len(clusters_)):
           clusters[clusters_[i]].append(i)
        return  AllAlgorithm(self.users, self.n_iteration, self.centroids,
                                      item_vecs,
                                      item_bias,
                                      user, clusters, self.tree)

#ClusterItems(data_dir + "/items1.txt", data_dir + "/items_bias1.txt", data_dir + "/train_ratings.txt_",data_dir + "/clusters", data_dir + "/centroids_file")
#RunTesting("centroids_file", "clusters")
