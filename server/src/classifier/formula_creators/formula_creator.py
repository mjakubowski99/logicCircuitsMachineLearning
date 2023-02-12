import numpy as np
from classifier.formula_creators.positive_formula_creator import positive_formula_creator
from classifier.formula_creators.negative_formula_creator import negative_formula_creator

def formula_creator(positive_data, negative_data, formulas_count, clauses_count, literals_count):
    positive = positive_formula_creator(positive_data, formulas_count, clauses_count, literals_count)
    negative = negative_formula_creator(negative_data, formulas_count, clauses_count, literals_count)
    
    return np.concatenate((positive, negative), axis=0)