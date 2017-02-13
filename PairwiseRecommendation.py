import numpy as np
import math
from numpy.linalg import inv
import random
from dataUtils import *
from Matrics import *
import sys
#from sklearn.decomposition import PCA
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--i', type = int)
parser.add_argument('--u', type = int)
parser.add_argument('--f', type = int)
FLAGS, unparsed = parser.parse_known_args()

items_names_file = "../../DATA/ml-20m/movies.csv"
data_dir = "../../GeneratedData1/data" + str(FLAGS.f)
#data_dir = "../../GeneratedData/data1"


class Vector(object):
    def __init___(self):
        self.parallel = np.empty()
        self.ortogonal = np.empty()
        self.parallel_norm = 0
        self.ortogonal_norm = 0

class Point(object):
    def __init__(self):
        self.x = 0
        self.y = 0

def GetDistanse(x, y):
    a = x-y
    x_norm = np.dot(x, x)
    x_norm1 = np.dot(x, x.T)
    y_norm = np.dot(y, y)
    return np.dot(a, a)

def GetFarthersPoint(points, x):
    d = 0
    point = 0
    for i in range(len(points)):
        p = points[i]
        new_d = GetDistanse(p.ortogonal, x)
        if (d < new_d):
            d = new_d
            point = i
    return point, d

def GetDiametrOfSet(points):
    result = Point()
    diametr = 0
    for i in range(len(points)):
        point, current_diametr = GetFarthersPoint(points[i+1:], points[i].ortogonal)
        if (diametr < current_diametr):
            diametr = current_diametr
            result.x = i
            result.y = point + i + 1
    #print(diametr)
    return result

def GetDiametrOfSetEasyApproximation(points, n_steps = 2):
    result = Point()
    diametr = 0
    i = random.randint(0,len(points) - 1)
    for j in range(n_steps):
        point, current_diametr = GetFarthersPoint(points, points[i].ortogonal)
        if (diametr < current_diametr):
            diametr = current_diametr
            result.x = i
            result.y = point
        i = point
    return result, diametr

def GetOrtogonalBasis(oldBasis, old_basis_norm, new_v):
    new_el_in_basis = new_v.copy()
    for i in range(len(oldBasis)):
        b = oldBasis[i]
        b_norm = old_basis_norm[i]
        new_el_in_basis -= b * (np.dot(b, new_v) / b_norm)
    oldBasis.append(new_el_in_basis)
    old_basis_norm.append(np.dot(new_el_in_basis, new_el_in_basis))

def GetOrtogonalComponent(vectors, new_element_in_bas, new_element_norm):
    max_norm = 0
    min_norm = 100000
    for v in vectors:
        new_parallel = np.dot(v.ortogonal, new_element_in_bas) / new_element_norm
        v.parallel = np.append(v.parallel, new_parallel)
        v.parallel_norm += new_parallel * new_parallel * new_element_norm
        v.ortogonal_norm -= new_parallel * new_parallel * new_element_norm
        v.ortogonal -= new_parallel * new_element_in_bas
        if (min_norm > v.parallel_norm):
            min_norm = v.parallel_norm
        if (max_norm < v.parallel_norm):
            max_norm = v.parallel_norm
    return (min_norm, max_norm)

def GetSetOfPoints(vectors, min_norm, max_norm, items_bias, threshold_):
    result = []
    threshold = min_norm + (max_norm - min_norm) / math.sqrt(len(vectors))
    if (threshold < threshold_):
        threshold = threshold_
    #print(threshold, min_norm, max_norm)
    for i in range(len(vectors)):
        parallel_norm = vectors[i].parallel_norm
        if (parallel_norm < threshold and abs(items_bias[i]) < 0.2):
            result.append(i)
    #if (len(result) > math.sqrt(len(vectors))):
    #    np.random.shuffle(result)
    #    result = result[:int(math.sqrt(len(vectors)))]
    return  result

def GetSetOfPointWithSort(vectors):
    a = []
    n_points = int(math.sqrt(len(vectors)))
    for i in range(len(vectors)):
        a.append([vectors[i].parallel_norm, i])
    a.sort(key = lambda x:x[0])
    return [a[i][1] for i in range(n_points)]

def GetSet(pool_of_points, items_by_basis):
    result = []
    for i in pool_of_points:
        result.append(items_by_basis[i])
    return result

def FirstAlgorithm(n_iterations, n_random_steps, items_original, item_bias, data_dir, threshold = 0.):
    #items_names = GetItemsNames(items_names_file)
    items_test = np.genfromtxt(data_dir + "/test_items.txt")
    comparative_items = open(data_dir + "/questions_name", 'w')
    print("End")
    #print(items[0][0])
    items = items_original.copy()
    items1 = items.copy()
    items_by_basis = []
    for i in range(len(items)):
        el = Vector()
        el.ortogonal = items[i].copy()
        el.ortogonal_norm = np.dot(items[i], items[i].T)
        el.parallel = np.empty(0)
        el.parallel_norm = 0
        items_by_basis.append(el)

    basis = []
    basis_norm = []
    questions = []
    arr = np.arange(len(items))
    np.random.shuffle(arr)
    pool_of_points = arr[:int(math.sqrt(len(items)))]
    question_item = []
    for i in range(n_iterations):
        diametr = 0
        question = ''
        items_set = GetSet(pool_of_points, items_by_basis)
        #question =  GetDiametrOfSet(items_set)
        for i_ in range(n_random_steps):
            question1, diametr_ =  GetDiametrOfSetEasyApproximation(items_set, 2)#GetDiametrOfSet(items_set)
            if (diametr < diametr_):
                diametr = diametr_
                question = question1
        item1 = pool_of_points[question.x]
        item2 = pool_of_points[question.y]
        #if (items_test[question.x]  + 1 in items_names and items_test[item2]  + 1 in items_names):
            #print(items_names[items_test[item1] + 1][0], items_names[items_test[item2] + 1][0])
            #comparative_items.write(items_names[items_test[item1] + 1][0] + "\t" +
            #                        items_names[items_test[item2] + 1][0] + '\n')
        try:

            new_vector = items1[item1] - items1[item2]
            #if (i == 0):
            #    print(item1, items1[0][0])
            #    print(items1[item1][0])
            #print(np.dot(new_vector, new_vector.T))
        except:
            print("EXCEPT")
            print(i)
            print(len(items), len(pool_of_points))

        questions.append(new_vector)
        question_item.append([item1, item2])
        GetOrtogonalBasis(basis, basis_norm, new_vector)
        min_norm, max_norm = GetOrtogonalComponent(items_by_basis, basis[-1], basis_norm[-1])
        pool_of_points = GetSetOfPoints(items_by_basis, min_norm, max_norm, item_bias, threshold)
        if (len(pool_of_points) < 300):
            pool_of_points = GetSetOfPoints(items_by_basis, min_norm, 1000000, item_bias, 10)

    questions = np.array(questions)
    my_A = np.dot(questions.T, questions)
    res = np.trace(np.linalg.inv(my_A + 0.001 * np.eye(my_A.shape[0])))
    np.savetxt(data_dir + "/questions", questions)
    np.savetxt( data_dir + "/questions_items", np.array(question_item))
    return res

def GetRandomQuestion(items, n):
    array = np.arange(len(items))
    np.random.shuffle(array)
    questions = []
    questions_items = array[:2*n]
    questions_items = questions_items.reshape([n, 2])
    np.savetxt(data_dir + "/quest_item_random", questions_items)
    for i in range(n):
        questions.append(items[array[i]] - items[array[i + n]])
        #questions.append(items[array[i]])
    questions = np.array(questions)
    my_A = np.dot(questions.T , questions)
    print(np.trace(np.linalg.inv(my_A + 0.001 * np.eye(my_A.shape[0]))))
    return np.array(questions)

def Test1(questions):
    item_vecs, item_bias, user_vecs, user_bias, global_bias, user_vecs_train, user_bias_train = GetData("data")
    mean_dif = 0
    my_NDCG = 0
    inv_questions = inv(np.dot(questions.T, questions) + 1e-8 * np.eye(questions.shape[1]))
    user_n = 0
    ratings = np.genfromtxt(data_dir + "/tes_ratings.txt")
    for user in user_vecs:
        answers = np.rint(np.dot(questions, user))
        answers[answers > 1] = 1.
        answers[answers < -1] = -1.
        user_estim = np.dot(inv_questions,
                           np.dot(answers, questions))
        non_zerro_ratings = ratings[user_n].nonzero()
        my_recommendation_list = GetRecommendetList(item_vecs[non_zerro_ratings], user_estim,
                                                           item_bias[non_zerro_ratings], user_vecs[user_n])
        a = ratings[user_n][non_zerro_ratings][my_recommendation_list]
        dif = user_estim - user_vecs[user_n]
        my_NDCG += NDCG(ratings[user_n][non_zerro_ratings][my_recommendation_list] + 1.)

        answers1 = np.rint(np.dot(questions, user_estim))
        dif = user - user_estim
        mean_dif += np.dot(dif, dif.T)
        user_n += 1
        #print(mean_dif / user_n)
    #print("MEAN ", mean_dif / len(user_vecs), my_NDCG / len(user_vecs))
    #print(len(user_vecs))
    return my_NDCG / len(user_vecs)

def RunTest(n_iter = 20):
    my = Test1(np.genfromtxt(data_dir + "/questions"))
    print("MY ", my, " MY")
    #yahoo = Test(np.genfromtxt("data/questions_yahoo"))
    r = 0.
    random_result = []
    for i in range(20):
        sys.stdout.write("\r%d%%" % i)
        random = Test1(GetRandomQuestion(np.genfromtxt("data//items.txt"), n_iter))
        random_result.append(random)
        r += random > my
    random_result = np.array(random_result)

    sys.stdout.write("\r")
    print("random ", np.mean(random_result), np.var(random_result))
    print(r)

"""def GetComparativeItem(items, item, A, used_items):
    A_2 = np.dot(A, A.T)
    new_A = A
    res = 0
    res_item = 0
    for j in range(items.shape[0]):
        v = items[item] - items[j]
        v = v.reshape(v.shape[0], 1)
        #del_ = np.dot(v.T, np.dot(A_2,v))
        #print(del_.shape)
        current_A_tr =np.dot(v.T, np.dot(A_2,v)) / (1. + np.dot(v.T, np.dot(A,v)))
        if (current_A_tr > res):
            res = current_A_tr
            res_item = j
    return res_item
"""


def GetComparativeItem(items, item, A, used_items, item_bias):
    new_A = A
    res = np.trace(new_A)
    res_item = 0
    for j in range(items.shape[0]):
        if (abs(item_bias[j] - item_bias[item]) > 0.2):
            continue
        v = items[item] - items[j]
        v = v.reshape(v.shape[0], 1)
        y = np.dot(v, v.T)
        y1 = np.dot(A, y)
        current_A = A - (np.dot(y1, A) / (1. + np.trace(y1)))
        #current_A = np.linalg.inv(np.linalg.inv(A) + y)
        current_A_tr = np.trace(current_A)
        if (current_A_tr < res):
            res = current_A_tr
            new_A = current_A
            res_item = j
    return res_item

def GetBestPair(items, A, used_items):
    new_A = np.linalg.inv(np.linalg.inv(A))
    res = np.trace(new_A)
    res_items = [0, 0]
    for i in range(0,items.shape[0]):
        for j in range(items.shape[0]):
            v = items[i] - items[j]
            v = v.reshape(v.shape[0], 1)
            y = np.dot(v, v.T)
            current_A = np.linalg.inv(np.linalg.inv(A) + y)
            current_A_tr = np.trace(current_A)
            if (current_A_tr < res):
                res = current_A_tr
                # new_A = A - np.dot(y, A) / (np.trace(A1) + 1.)
                # b = np.linalg.inv(np.linalg.inv(A) + y)
                #print(np.max(new_A - b), np.min(new_A - b))
                new_A = current_A
                res_items = [i, j]
    print(res)
    return res_items, new_A

def GetWorstPair(items, A):
    new_A = np.linalg.inv(np.linalg.inv(A))
    res = 1e+10
    res_items = [0, 0]
    print(res)
    for i in range(0,items.shape[0]):
        v = items[i]
        v = v.reshape([v.shape[0], 1])
        y = np.dot(v, v.T)
        current_A = np.linalg.inv(np.linalg.inv(A) - y)
        current_A_tr = np.trace(current_A)
        if (current_A_tr < res):
            res = current_A_tr
            # new_A = A - np.dot(y, A) / (np.trace(A1) + 1.)
            # b = np.linalg.inv(np.linalg.inv(A) + y)
            #print(np.max(new_A - b), np.min(new_A - b))
            new_A = current_A
            res_items = i
    print(res)
    return res_items, new_A,  np.delete(items, res_items, axis=0)

def BackGwardGreedy(items, n_q):
    items1 = items.copy()
    A = np.eye(items.shape[1]) * 0.001
    dic = {}
    iter = 0
    Items = np.zeros([items.shape[0]*(items.shape[0] - 1) / 2, items.shape[1]])
    for i in range(items.shape[0]):
        for j in range(i+1, items.shape[0]):
            v = items[i] - items[j]
            Items[iter] = v
            v = v.reshape(v.shape[0], 1)
            y = np.dot(v, v.T)
            A += y
            dic[iter] = [[i, j], True]
            iter += 1
    A = np.linalg.inv(A)
    for i in range(iter - n_q):
        print(i)
        item,A,Items =  GetWorstPair(Items, A)
        print(Items.shape)
        k = 0
        for j in range(iter):
            if (dic[j][1] == True):
                k += 1
            if (k > item):
                dic[j][1] = False
                break
    res = []
    for k in dic.keys():
        if dic[k][1] == True:
            res.append(dic[k][0])
    return res


def GetProperAlgorithm(items, n_q):
    A = np.linalg.inv(np.eye(items.shape[1]) * 0.001)
    questions = []
    used_items = []
    for i  in range(n_q):
        res_item, A = GetBestPair(items, A, used_items)
        used_items.append((res_item[0] + res_item[1],
                           res_item[0]**2 + res_item[1]**2))
        print(res_item)
        questions.append(res_item)
    return np.array(questions)

def main(n_random = 5, n_iter = FLAGS.i):
    item_vecs, item_bias, user_vecs, user_bias, global_bias, user_vecs_train, user_bias_train = GetData(data_dir)
    popular_items = np.genfromtxt(data_dir + "/train_ratings.txt_").astype(int)
    #pca = PCA(n_components=100)
    #items = pca.fit_transform(item_vecs)
    #GetRandomQuestion(item_vecs, 40)
    res = 1e+10
    THRESHOLD_ = 1.
    """for i in range(20):
        THRESHLOD = 0.1 * i + 3.
        print(THRESHLOD)
        res_ = FirstAlgorithm(n_iter, n_random, item_vecs, item_bias, data_dir, THRESHLOD)
        print(res_)
        if (res_ < res):
            THRESHOLD_ = THRESHLOD
            res = res_
    print(THRESHOLD_, res)
    THRESHLOD = THRESHOLD_"""
    print(FirstAlgorithm(n_iter, n_random, item_vecs[popular_items[:200]], item_bias[popular_items[:200]], data_dir, 10.))

    """items = np.genfromtxt(data_dir + "/questions_items")
    items = items.astype(int)
    items = items.reshape((items.shape[0]*2, ))
    print(items)
    res = GetProperAlgorithm(item_vecs[items], n_iter)
    res1 = res.T
    my_quetions = item_vecs[items[res1[0]]] - item_vecs[items[res1[1]]]
    my_A = np.dot(my_quetions.T, my_quetions)
    print(np.trace(np.linalg.inv(my_A + 0.001 * np.eye(my_A.shape[0]))))


    print(res.shape)
    items = items.astype(int)
    np.savetxt(data_dir + "/questions1", item_vecs[items[res1[0]]] - item_vecs[items[res1[1]]])
    questions = [[items[res[i][0]], items[res[i][1]]] for i in range(n_iter)]
    np.savetxt(data_dir + "/questions_items", np.array(questions))"""
        #q = np.genfromtxt("data/questions")
        #print (q.shape, np.max(q[-10:]))
        #Test(q)"""

"""for i in range(10):
    main(n_iter=10*(i+1))
    q = np.genfromtxt("data/questions")
    print (q.shape, np.max(q[-10:]))
    Test(q)"""
#main()
#if __name__ == "main":
#    main()
#for i in range(10):
#    print(i)
#    #main(2)
#    #RunTest()
#    main(n_iter=10*(i + 1))
#    RunTest(10 * (i+1))




