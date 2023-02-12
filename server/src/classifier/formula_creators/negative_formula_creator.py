import numpy as np
from classifier.formula_randomizer import random
from classifier.formula_randomizer import repeat_random_until_got_new_value

def negative_formula_creator(data, formulas_count, clauses_count, literals_count):
    formulas = np.zeros((formulas_count, clauses_count, literals_count, 2), dtype=int)

    row_number, col_number = data.shape

    for formula in formulas:
        for clause in formula:
            drawn_cols = []
            random_row = random(0,row_number-1)

            for literal in clause:
                random_col = repeat_random_until_got_new_value(0, col_number-1, drawn_cols)
                drawn_cols.append(random_col)

                literal[0] = random_col
                if data[random_row][random_col]:
                    literal[1] = 0
                else:
                    literal[1] = 1

    return formulas