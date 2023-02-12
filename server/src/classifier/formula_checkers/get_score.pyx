import numpy as np

def get_formulas_score(X, formulas, goal):
    return np.array([get_formula_score(X, formula, goal) for formula in formulas])

def get_formula_score(X, formula, goal):
    is_positive = np.array([formula_is_positive(row, formula) for row in X])
    
    true_positive = np.sum((is_positive == goal) & (goal == True), axis=0)
    true_negative = np.sum((is_positive == goal) & (goal == False), axis=0)
    false_positive = np.sum((is_positive != goal) & (goal == True), axis=0)
    false_negative = np.sum((is_positive != goal) & (goal == False), axis=0)

    return np.array([
        [true_negative, false_positive], 
        [false_negative, true_positive]
    ])


def formula_is_positive(row, formula):
    for clause in formula:
        clause_is_true = False

        for literal in clause:
            value = row[literal[0]]
            clause_is_true = literal[1] == value

            if clause_is_true:
                break
        
        if not clause_is_true:
            return False
    return True