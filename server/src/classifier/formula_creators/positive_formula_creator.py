import numpy as np
from classifier.formula_randomizer import repeat_random_until_got_new_value

def positive_formula_creator(X, formulas_count, clauses_count, literals_count):
    formulas = np.zeros((formulas_count, clauses_count, literals_count, 2), dtype=int)

    row_number, col_number = X.shape

    for formula in formulas:
        for clause in formula:
            drawn_rows = []
            drawn_cols = []

            for literal in clause:
                random_row = repeat_random_until_got_new_value(0, row_number-1, drawn_rows)
                random_col = repeat_random_until_got_new_value(0, col_number-1, drawn_cols)

                drawn_rows.append(random_row)
                drawn_cols.append(random_col)

                literal[0] = random_col
                literal[1] = X[random_row][random_col]

    return formulas