import sys
from dataUtils import *
from Matrics import *
from PairwiseRecommendation import FirstAlgorithm
from PairwiseRecommendation import GetComparativeItem

"""
Relative algorithm from Towards Conversational Recommender Systems
"""
def GetComparativeItemSimple(items, item, questions, answers, items_bias, used_items):
    answers_new = list(answers)
    answers_new.append(-1)
    new_questions = list(questions)
    reshape = False
    if len(new_questions) < 1:
        reshape = True
    new_questions.append(items[item])
    # print(np.array(new_questions).shape)
    new_questions = np.array(new_questions)
    new_inverse_matrix = np.linalg.inv(np.dot(new_questions.T, new_questions) + 0.001 * np.eye(new_questions.shape[1]))
    new_user = np.dot(new_inverse_matrix,
                      np.dot(np.array(answers_new - items_bias[item]), np.array(new_questions)))

    # picking the comparative item
    comparative_item = BestItem(items, new_user, used_items, items_bias)
    return comparative_item


def OneStep(user_estim, user, items, used_items, inverse_matrix, answers, questions, items_bias, baseline=True):
    n_latent_factors = len(user)
    # for picking best item
    item = BestItem(items, user_estim, used_items, items_bias)
    used_items.append(item)

    #make virtual step


    #it = items[item].reshape(items.shape[1], 1)
    #v = np.dot(it, it.T)
    #new_inverse_matrix = GetInverseMatrix(inverse_matrix, v)


    #print (np.array(new_questions).shape)
    if (baseline):
        comparative_item = GetComparativeItemSimple(items, item, questions, answers, items_bias, used_items)
    else:
        if len(questions) < 1:
            A = np.linalg.inv(0.001 * np.eye(items.shape[1]))
        else:
            A = np.linalg.inv(np.dot(np.array(questions).T, np.array(questions)) + 0.001 * np.eye(np.array(questions).shape[1]))
        comparative_item = GetComparativeItem(items, item, A, used_items, items_bias)
    used_items.append(comparative_item)

    user_answer = receive_answer(user, items[item] - items[comparative_item], -1, 1,
                                 items_bias[item] - items_bias[comparative_item])
    answers.append(user_answer - items_bias[item] + items_bias[comparative_item])

    it = items[item]
    questions.append(items[item] - items[comparative_item])
    questions1 = np.array(questions)
    #if (reshape):
    #    questions1 = questions1.reshape(1, questions1.shape[0])

    #com_it = items[comparative_item].reshape([items.shape[1], 1])
    #v = np.dot(it-com_it, (it - com_it).T)
    #new_inverse_matrix1 = GetInverseMatrix(inverse_matrix, v)

    new_inverse_matrix = np.linalg.inv(np.dot(questions1.T, questions1) + 0.001 * np.eye(questions1.shape[1]))
    #d = np.max(new_inverse_matrix - new_inverse_matrix1)
    #print (np.max(d))

    user_estim = np.dot(new_inverse_matrix,
                         np.dot(np.array(answers), np.array(questions)))
    return user_estim, answers, new_inverse_matrix, used_items, questions, [item, comparative_item]

def AllAlgorithm(items, user, user_estim, n_questions, items_bias, baseline = True):
    inverse_matrix = np.linalg.inv(np.eye(user.shape[0]) * 0.001)
    answers = []
    used_items = []
    questions = []
    questions_item = []
    for q in range(n_questions):
        user_estim, answers, inverse_matrix, used_items, questions, item_question = OneStep(user_estim, user,
                                                                  items, used_items, inverse_matrix, answers, questions,
                                                                             items_bias, baseline)
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

    print str(n_users) + ' users'
    print str(n_items) + ' items'
    sparsity = float(len(ratings.nonzero()[0]))
    sparsity /= (ratings.shape[0] * ratings.shape[1])
    sparsity *= 100
    print 'Sparsity: {:4.2f}%'.format(sparsity)
    return ratings

def main(dir):
    #dir = "data"
    print(dir)
    item_vecs, item_bias, user_vecs, user_bias, global_bias, user_vecs_train, user_bias_train = GetData(dir)
    user_estim = np.mean(user_vecs_train, axis = 0)

    n_points = 40
    mean_dist = [0. for i in range(n_points)]
    NDCG_train = [0. for i in range(n_points)]
    my_NDCG = [0. for i in range(n_points)]
    correct_pairs = [0. for i in range(n_points)]
    my_correct_pairs = [0. for i in range(n_points)]
    my_mean_dist = [0. for i in range(n_points)]
    test = np.genfromtxt(dir + '/test.txt')
    test = test.astype(int)
    n_users_in_test_ = 300
    n_users_in_test = 0
    #ratings = GetData1("../movieLens/ml-100k", test[:n_users_in_test])

    ratings = np.genfromtxt(dir + "/tes_ratings.txt")
    #ratings = np.zeros([300, item_vecs.shape[0]])
    #for i in xrange(ratings_.shape[0]):
    #    ratings[ratings_[i][0]][ratings_[i][1]] = ratings_[i][2]
    ratings = ratings * (item_bias < 0.2) * (item_bias > -0.2)

    #FirstAlgorithm(40, 10, item_vecs, item_bias, dir)
    for i in range(n_users_in_test_):
        sys.stdout.write("\r%d%%" % i)
        #if (user_bias[i] > 0.2 or user_bias[i] < -0.2):
        #    continue
        my_questions_item = np.genfromtxt(dir + "/questions_items")[0:n_points]
        user_estim = np.mean(user_vecs_train, axis = 0)
        answers, questions,used_items,questions_item = AllAlgorithm(item_vecs, user_vecs[i],
                                          user_estim, n_points, item_bias)
        #answers_mine, my_questions, my_used_items, my_questions_item = AllAlgorithm(item_vecs, user_vecs[i],
        #                                  user_estim, n_points, item_bias, False)
        test_ratings = ratings[i]
        test_ratings[used_items] = 0

        questions_item = np.array(questions_item)
        my_questions_item = np.array(my_questions_item)
        my_questions_item = my_questions_item.astype(int)

        test_ratings[my_questions_item.T[0]] = 0
        test_ratings[my_questions_item.T[1]] = 0
        non_zerro_ratings = np.array(test_ratings.nonzero()[0])
        #print(non_zerro_ratings.shape)
        if (non_zerro_ratings.shape[0] < 15):
            continue
        n_users_in_test += 1
        #test

        #print(np.trace(np.linalg.inv(np.dot(questions.T, questions) +
        #                             0.001 * np.eye(questions.shape[1]))))
        #print(np.trace(np.linalg.inv(np.dot(my_questions.T, my_questions) +
        #                             0.001 * np.eye(my_questions.shape[1]))))

        a = ratings[i][non_zerro_ratings]
        for j in range(n_points):
            dist_b, ndcg_b, correct_pairs_ = Test(questions_item, (j + 1),
                                                  item_vecs[non_zerro_ratings], item_bias[non_zerro_ratings],
                                                  user_vecs[i], a, item_vecs, item_bias)
            dist_b_my, ndcg_b_my, correct_pairs_my_ = Test(my_questions_item, (j + 1),
                                                           item_vecs[non_zerro_ratings], item_bias[non_zerro_ratings],
                                                           user_vecs[i], a, item_vecs, item_bias)

            #print(dist_b, dist_b_my)
            mean_dist[j] += dist_b
            correct_pairs[j] += correct_pairs_
            my_mean_dist[j] += dist_b_my
            my_correct_pairs[j] += correct_pairs_my_
            for i1 in range(10):
                serp = GetSERP(a)
                if max(serp) > 0:
                    dist_b,ndcg_b, correct_pairs_ = Test(questions_item, (j+1),
                        item_vecs[non_zerro_ratings[serp]], item_bias[non_zerro_ratings[serp]],
                        user_vecs[i], a[serp], item_vecs, item_bias)
                    dist_b_my, ndcg_b_my, correct_pairs_my_ = Test(my_questions_item, (j+1),
                                    item_vecs[non_zerro_ratings[serp]], item_bias[non_zerro_ratings[serp]],
                                    user_vecs[i], a[serp], item_vecs, item_bias)
                    NDCG_train[j] += ndcg_b / 10.
                    my_NDCG[j] += ndcg_b_my / 10.
            #test all
        sys.stdout.write("\r")
    mean_dist = np.array(mean_dist)
    my_mean_dist = np.array(my_mean_dist)
    NDCG_train = np.array(NDCG_train, float)
    my_NDCG = np.array(my_NDCG, float)
    correct_pairs = np.array(correct_pairs, float)
    my_correct_pairs = np.array(my_correct_pairs, float)
    print(n_users_in_test)
    print(" BASELINE = ", mean_dist / (n_users_in_test))
    print(" MY = ", my_mean_dist / (n_users_in_test))
    print(" BASELINE = ", NDCG_train / (n_users_in_test))
    print(" MY = ", my_NDCG / (n_users_in_test))
    print(" BASELINE = ", correct_pairs / n_users_in_test)
    print(" MY = ", my_correct_pairs/ n_users_in_test)




    #print(mean_dist / n_users_in_test)
#main("data0")
for i in range(9):
    if (i == 5):
        continue
    main("data" + str(i))