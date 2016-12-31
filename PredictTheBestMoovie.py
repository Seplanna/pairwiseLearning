import sys
from dataUtils import *
from simple_baseline_pairwise import  *
from Play import Update
from Play import OneIteration


def NStepsTOgetBEstItem(user, baseline = True):
    items_names = GetItemsNames("../dataset/ml-1m/movies_mine.dat")
    items_test = np.genfromtxt("data/test_items.txt")
    item_vecs, item_bias, user_vecs, user_bias, global_bias, user_vecs_train, user_bias_train = GetData(
        "data")
    inverse_matrix = np.linalg.inv(np.eye(item_vecs.shape[1]) * 0.001)
    answers = []
    used_items = []
    questions = []
    user_estim = np.mean(user_vecs_train, axis=0)
    #print(user_estim.shape)
    best = BestItem(item_vecs, user, [], item_bias)
    predication = best + 1
    n_steps = 0
    while (predication != best and n_steps < 100):
        if (baseline):
            item, comparative_item, used_items = OneIteration(item_vecs, item_bias, user_estim, used_items, questions,
                 answers)
        else:
            questions1 = np.genfromtxt("data/questions_items")
            item, comparative_item = questions1[n_steps]
            used_items.append(item)
            used_items.append(comparative_item)
        user_answer = receive_answer(user, item_vecs[item] - item_vecs[comparative_item], -1, 1,
                                 item_bias[item] - item_bias[comparative_item])

        answers, inverse_matrix, questions = \
            Update(answers, user_answer,
                   item_vecs, item_bias, item, comparative_item,
                   questions)
        user_estim = np.dot(inverse_matrix,
                                 np.dot(np.array(answers),
                                        np.array(questions)))
        predication = BestItem(item_vecs, user_estim, [], item_bias)
        n_steps += 1
    print (n_steps)
    return n_steps

def main():
    item_vecs, item_bias, user_vecs, user_bias, global_bias, user_vecs_train, user_bias_train  = \
        GetData("data")
    print(user_vecs.shape)
    n_steps = 0
    user_n = 0
    for user in user_vecs:
        print(user_n)
        n_steps += NStepsTOgetBEstItem(user, True)
        user_n += 1
        if (user_n > 200):
            break
    print(float(n_steps) / user_n)
main()
