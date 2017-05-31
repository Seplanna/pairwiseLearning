import sys
import numpy as np

class StaticPairwiseQuestions():
    def __init__(self, file_with_questions):
        self.questions = np.array(np.genfromtxt(file_with_questions).astype('int'))
    def RecieveQuestions(self, item_vecs, user, user_estim, n_points, item_bias, ratings):
        return self.questions

class StaticAbsoluteQuestions():
    def __init__(self, file_with_questions):
        questions = np.array(np.genfromtxt(file_with_questions).astype('int'))
        self.questions = np.zeros((2, questions.shape[0]))
        self.questions -= 1
        self.questions[0] = questions
        self.questions = self.questions.T

    def RecieveQuestions(self, item_vecs, user, user_estim, n_points, item_bias, ratings):
        return self.questions
