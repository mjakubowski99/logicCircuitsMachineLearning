from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from classifier.formula_creators.formula_creator import formula_creator

from classifier.formula_checkers.get_score import get_formulas_score
from classifier.formula_checkers.get_score import formula_is_positive
from classifier.formula_checkers.best_formulas import get_most_satisfiable_formulas

import time

import numpy as np

class LogicClassifier(BaseEstimator):

    def __init__(self, literal_number_in_clause=3, clause_number_in_formula=5, formula_multiples_number=100, max_number_of_training_loop=20):
        self.literal_number_in_clause = literal_number_in_clause
        self.clause_number_in_formula = clause_number_in_formula
        self.formula_multiples_number = formula_multiples_number
        self.max_number_of_training_loop = max_number_of_training_loop
        self.classes_count = 0

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes = np.unique(y)

        X = self.to_numpy_array(X)
        y = self.to_numpy_array(y)

        data = self.group_rows_by_classes(X,y)

        return self.learn(X,y,data)

    def learn(self, X, y, data):

        i = 0
        positive = np.array([])
        negative = np.array([])
        early_break = False

        while i < self.max_number_of_training_loop:
            print("Iteracja: ", i)
            expected_positive_formula_number = self.formula_multiples_number - len(positive)
            expected_negative_formula_number = self.formula_multiples_number - len(negative)

            generated_positive_formulas = formula_creator(data[1], data[0], expected_positive_formula_number, self.clause_number_in_formula, self.literal_number_in_clause)
            generated_negative_formulas = formula_creator(data[0], data[1], expected_negative_formula_number, self.clause_number_in_formula, self.literal_number_in_clause)

            if positive.size == 0:
                positive = generated_positive_formulas
            else:
                positive = np.concatenate((positive,generated_positive_formulas), axis=0)

            if negative.size == 0:
                negative = generated_negative_formulas
            else:
                negative = np.concatenate((negative, generated_negative_formulas), axis=0)

            best_positive_formulas = get_most_satisfiable_formulas(positive, get_formulas_score(data[1], positive, True))
            best_negative_formulas = get_most_satisfiable_formulas(negative, get_formulas_score(data[0], negative, False))

            early_break = len(best_positive_formulas) > self.formula_multiples_number and len(best_negative_formulas) >= self.formula_multiples_number

            positive = best_positive_formulas
            negative = best_negative_formulas

            if early_break:
                break
            i+=1

        self.positive_formulas = positive
        self.negative_formulas = negative

    def predict(self, X):
        check_is_fitted(self)

        return 0

    def score(self, X, y):
        check_X_y(X, y)

        X = self.to_numpy_array(X)
        y = self.to_numpy_array(y)

        positive = [get_formulas_score(np.array([x]), self.positive_formulas, True)[0] for x in X]
        negative = [get_formulas_score(np.array([x]), self.negative_formulas, False)[0] for x in X]

        positive_score = np.array([x[1][1]+x[0][0] for x in positive])
        negative_score = np.array([x[1][1]+x[0][0] for x in negative])

        index = 0
        guessed = 0
        for expected in y:
            predicted = positive_score[index] > negative_score[index] 

            if predicted == bool(expected):
                guessed+=1
            index+=1

        return guessed/len(y)


            


    def to_numpy_array(self, value):
        if not isinstance(value, np.ndarray):
            return np.array(value)
        return value

    def group_rows_by_classes(self, X, y):
        grouped = {label: X[y==label] for label in self.classes}

        return grouped
            




        

        
    