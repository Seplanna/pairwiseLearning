"""
This algorithm is implemented from http://ieeexplore.ieee.org/abstract/document/6212388/?reload=true paper
"""
import random
from dataUtils import *
from sklearn.cluster import KMeans
from numpy.linalg import inv

class Result(object):
    def __init__(self, best_items, user_division):
        self.best_items = best_items.copy()
        self.user_division = user_division.copy()

class Node(object):
    def __init__(self, users = []):
        self.value = [0,0]
        self.users = users
        self.leaf = True
        self.sons = []


def ChooseRandomSample(n_users_to_choose, n_users):
    return random.sample(range(n_users), n_users_to_choose)

def AlgorithmFromThePaperOneStep(users, items, ratings, min_rating, max_rating, used_items):
    """

    :param users: set of vectors of similar users
    :param items: set of vectors of availible items
    :param user: the vector of the user
    :param ratings: dictionary {rating: rating_in_list}
    :return: new_pair
    """
    n_users = len(users)
    n_items = len(items)
    best_res = 1e+100
    best_items = [0, 1]
    users_division = [[] for i in range(max_rating - min_rating + 1)]
    for i1 in range(n_items):
        if i1 not in used_items:
            for i2 in range(i1+1, n_items):
                if (i2 not in used_items):
                    similar_users = [[] for i in range(max_rating - min_rating + 1)]
                    res = 0
                    for u in range(n_users):
                        u_answer = receive_answer(users[u], items[i1] - items[i2], min_rating, max_rating)
                        similar_users[ratings[u_answer]].append(u)
                    not_to_use_these_pair = False
                    for r in similar_users:
                        if (len(r) < 10):
                            not_to_use_these_pair = True
                    if not_to_use_these_pair:
                        continue
                    for r in similar_users:
                        weight = float(len(r)) / n_users
                        res += np.linalg.det(np.dot(users[r].T, users[r]) + np.eye(users.shape[1])) * weight
                    if res < best_res or best_res == 1e+100:
                        best_res = res
                        best_items = [i1, i2]
                        users_division = np.array(similar_users)
    return best_items, users_division

def AllAlgorithmFromPaper(users, items, user, n_questions, lasy_tree, cluster_mode = False):
    if cluster_mode:
        kmeans = KMeans(n_clusters = 50).fit(items)
        np.savetxt("data/item_clusters", kmeans.cluster_centers_)
    ratings = {-1:0, 0:1, 1:2}
    min_rating = -1
    max_rating = 1
    user_set = np.arange(len(users))
    used_items = []
    user_answers = []
    questions = []
    node = lasy_tree
    for q in range(n_questions):
        #print(q)
        if (node.leaf):
            best_items,users_division = \
                AlgorithmFromThePaperOneStep(users[user_set], items,
                                             ratings, min_rating, max_rating, used_items)
            if (len(users_division[0]) == 0):
                break
            node.value = best_items
            for i in range(len(users_division)):
                node.sons.append(Node(users_division[i]))
            node.leaf = False
        else:
            best_items = node.value
        user_answer = receive_answer(user, items[best_items[0]] - items[best_items[1]], min_rating, max_rating)
        same_users = node.sons[ratings[user_answer]].users
        try:
            user_set = user_set[np.array(same_users)]
        except:
            print (np.array(same_users))
            print(q)
        #print(best_items)
        node = node.sons[ratings[user_answer]]
        used_items.append(best_items[0])
        used_items.append(best_items[1])
        questions.append(items[best_items[0]] - items[best_items[1]])
        user_answers.append(user_answer)
        if (len(user_set) < 30):
            break
    questions = np.array(questions)
    user_answers = np.array(user_answers)
    inv_questions = inv(np.dot(questions.T, questions) + 0.001 * np.eye(questions.shape[1]))
    #user_estim = np.mean(users[same_users])
    user_estim = np.dot(inv_questions,
                        np.dot(user_answers, questions))
    dif = user - user_estim
    return np.dot(dif, dif.T)



def main():
    item_vecs, item_bias, user_vecs, user_bias, global_bias = GetData("data")
    items = np.genfromtxt("data/item_clusters")
    array = np.arange(len(user_vecs))
    np.random.shuffle(array)
    n_users_in_test = len(user_vecs) / 3
    users_in_train = user_vecs[array[n_users_in_test:]]
    mean_dist = 0
    node = Node(users_in_train)
    for i in range(n_users_in_test):
        #print(i)
        mean_dist += AllAlgorithmFromPaper(users_in_train, items, user_vecs[array[i]], 20, node)
    print(mean_dist / n_users_in_test)

for i in range(100):
    main()