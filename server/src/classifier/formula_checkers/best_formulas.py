import numpy as np

def get_most_satisfiable_formulas(formulas, answers):
    return np.array([formulas[i] for i in range(formulas.shape[0]) if formula_is_satisfiable(answers[i], i)])

def formula_is_satisfiable(answer, index):
    true_positive = answer[1][1]
    false_positive = answer[0][1]
    true_negative = answer[0][0]
    false_negative = answer[1][0]

    if (false_positive+false_negative) == 0:
        return True
    return (true_positive+true_negative) / (false_positive+false_negative) > 2.0