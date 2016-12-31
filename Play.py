from PIL import Image
import numpy as np
import requests
import json
import math
import os.path
import wget
from dataUtils import *
from Matrics import *
from PairwiseRecommendation import GetComparativeItem

"""headers = {'Accept': 'application/json'}
payload = {'api_key': "e16fe7a4d1f7e73c8d9a611656c980c8"}
response1 = requests.get("http://api.themoviedb.org/3/configuration", \
                       params=payload, \
                       headers=headers)
response1 = json.loads(response1.text)
base_url = response1['images']['base_url'] + 'w185'
"""

def get_poster(imdb_url, api_key="e16fe7a4d1f7e73c8d9a611656c980c8"):
    # Get IMDB movie ID
    # images = 'images/'
    # response = requests.get(imdb_url)
    # movie_id = response.url.split('/')[-2]

    # Query themoviedb.org API for movie poster path.
    # movie_url = 'http://api.themoviedb.org/3/movie/{:}/'.format(movie_id)
    headers = {'Accept': 'application/json'}
    payload = {'api_key': api_key}
    movie_title = imdb_url.split('?')[-1].split('(')[0]
    payload['query'] = movie_title
    response = requests.get('http://api.themoviedb.org/3/search/movie', \
                            params=payload, \
                            headers=headers, timeout=10)
    # print (response.text)
    rating = json.loads(response.text)['results'][0]['vote_average']
    file_path = json.loads(response.text)['results'][0]['poster_path']
    poster = base_url + file_path
    filename = wget.download(poster)
    #im = Image.open(filename)
    #im.show()
    #os.remove(filename)
    #print(rating, json.loads(response.text)['results'][0]["overview"])
    return filename

def OneIteration(items, items_bias, user_estim, used_items, questions,
                 answers, baseline=True):
    n_latent_factors = items.shape[1]
    # for picking best item
    #self.user_estim = np.dot(self.inverse_matrix,
    #                         np.dot(np.array(self.answers),
    #                                np.array(self.questions)))


    item = BestItem(items, user_estim, used_items, items_bias)
    used_items.append(item)

    # make virtual step
    if (baseline):
        answers_new = list(answers)
        answers_new.append(-1 - items_bias[item])

        new_questions = list(questions)
        reshape = False
        if len(new_questions) < 1:
            reshape = True
        new_questions.append(items[item])
        new_questions = np.array(new_questions)
        new_inverse_matrix = np.linalg.inv(
            np.dot(new_questions.T, new_questions) + 0.001 * np.eye(new_questions.shape[1]))
        new_user = np.dot(new_inverse_matrix,
                          np.dot(np.array(answers_new), np.array(new_questions)))

        # picking the comparative item
        comparative_item = BestItem(items, new_user, used_items, items_bias)
    else:
        if len(questions) < 1:
            A = np.linalg.inv(0.001 * np.eye(items.shape[1]))
        else:
            A = np.linalg.inv(np.dot(np.array(questions).T, np.array(questions)) + 0.001 * np.eye(np.array(questions).shape[1]))
        comparative_item = GetComparativeItem(items, item, A, used_items, items_bias)
    used_items.append(comparative_item)
    return item, comparative_item, used_items

def Get_posters(items_test, items_names, item, comparative_item):
    try:
        img1 = get_poster(items_names[items_test[item] + 1][0], api_key="e16fe7a4d1f7e73c8d9a611656c980c8")
        img2 = get_poster(items_names[items_test[comparative_item] + 1][0], api_key="e16fe7a4d1f7e73c8d9a611656c980c8")
        return img1, img2
    except:
        pass

def Update(answers, user_answer,
            items, items_bias, item, comparative_item,
            questions):
    answers.append(user_answer - items_bias[item] + items_bias[comparative_item])
    it = items[item]
    questions.append(items[item] - items[comparative_item])
    questions1 = np.array(questions)
    new_inverse_matrix = np.linalg.inv(np.dot(questions1.T, questions1) + 0.001 * np.eye(questions1.shape[1]))
    return answers, new_inverse_matrix, questions

#def MyStep():

def RunIterativeGame():
    items_names = GetItemsNames("../dataset/ml-1m/movies_mine.dat")
    items_test = np.genfromtxt("data/test_items.txt")
    item_vecs, item_bias, user_vecs, user_bias, global_bias, user_vecs_train, user_bias_train = GetData("data")
    inverse_matrix = np.linalg.inv(np.eye(item_vecs.shape[1]) * 0.001)
    answers = []
    used_items = []
    questions = []
    user_estim = np.mean(user_vecs_train, axis=0)
    for q in range(20):
        #print(user_estim)
        item, comparative_item, used_items = OneIteration(item_vecs, item_bias, user_estim, used_items, questions,
                     answers)
        Get_posters(items_test, items_names, item, comparative_item)
        answer = float(raw_input("how do you like items " + "\n"))
        answers, inverse_matrix, questions = Update(answers, answer,
               item_vecs, item_bias, item, comparative_item,
               questions)


        #user_estim, answers, inverse_matrix, used_items, questions = \
        #OneIteration(item_vecs, item_bias, user_estim, used_items, questions, answers, items_names, items_test)

        user_estim = np.dot(inverse_matrix,
                        np.dot(np.array(answers), np.array(questions)))

#RunIterativeGame()
def GetPictures():
    items_names = GetItemsNames("../dataset/ml-1m/movies_mine.dat")
    for name in items_names.keys():
        try:
            n = items_names[name][0].split('?')[-1].split('(')[0]
            if  not os.path.isfile("images/" + n + ".jpg"):
                print (name)
                file_name = get_poster(items_names[name][0], api_key="e16fe7a4d1f7e73c8d9a611656c980c8")
                n = items_names[name][0].split('?')[-1].split('(')[0]
                print(n)
                os.rename(file_name, "images/" + n + ".jpg")
        except:
            continue
#GetPictures()