import os
import time
import numpy as np
import pandas as pd
import json
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from datetime import datetime

# # DATASET 1
data_set_url = "../datasets/dataset1/heart.csv"
goal_column = 13
config_file_url = "../datasets/dataset1/heart.config.json"
config_file_url = None
column_numbers_array = None

# # DATASET 2
#data_set_url = "../datasets/dataset2/weatherAUS.csv"
#goal_column = 22
#config_file_url = "../datasets/dataset2/weatherAUS.config.json"
#column_numbers_array = None

# DATASET 3
#data_set_url = "../datasets/dataset3/county_results.csv"
#goal_column = 54
#config_file_url = None
#column_numbers_array = None

# Wartości dla generowanych formuł
literal_number_in_clause = 3
clause_number_in_formula = 10
formula_multiples_number = 100

# Wartość powtarzania pętli poprawiającej formuły
max_number_of_training_loop = 2


def add_missing_values_or_removed(data_frame, goal_label):
    delete_column_percent = 0.6
    change_to_mean = 0.9
    all_rows = len(data_frame.index)
    nulls_array = data_frame.isnull().sum()
    for label in data_frame.columns.values:
        if label == goal_label:
            continue
        if nulls_array[label] < all_rows * delete_column_percent:
            data_frame.drop(label, axis=1)
        if nulls_array[label] < all_rows * change_to_mean:
            data_frame.dropna(subset=[label])
        if data_frame[label].dtype != 'object':
            data_frame[label] = data_frame[label].fillna(data_frame[label].mean())
    return data_frame


def add_changes_from_config_files(data_frame, config_file_url):
    def convert_changes_from_config(variable):
        change_type = config.get(label).get('type')
        values = config.get(label).get('values')
        default = config.get(label).get('default')
        if change_type == 'range':
            for x in values:
                if x.get('gte') <= variable < x.get('lower'):
                    return x.get('value')
        if change_type == 'equal':
            for x in values:
                if variable == x.get('equal'):
                    return x.get('value')
        if default is not None:
            return default
        return variable

    with open(config_file_url) as json_file:
        config = json.load(json_file)
        for label in data_frame.columns.values:
            if config.get(label) is not None:
                data_frame[label] = data_frame[label].apply(convert_changes_from_config)
    return data_frame


def add_tags_for_object_values(main_data, goal_label):
    max_unique_object_value = 5
    dtypes_array = main_data.dtypes
    label_encoder = LabelEncoder()
    for label in main_data.columns.values:
        if label == goal_label:
            continue
        if dtypes_array[label] == 'object':
            nun = main_data[label].nunique()
            if nun > max_unique_object_value:
                main_data = main_data.drop(label, axis=1)
                continue
            label_encoder.fit(main_data[label])
            main_data[label] = label_encoder.transform(main_data[label])
    return main_data


def change_values_to_integer(main_data, goal_label):
    def multiply_variable(variable):
        return variable * 100

    dtypes_array = main_data.dtypes
    for label in main_data.columns.values:
        if label == goal_label:
            continue
        if dtypes_array[label] != 'int64':
            if dtypes_array[label] != 'float64':
                main_data[label] = main_data[label].apply(multiply_variable)
            main_data[label] = main_data[label].round(0).astype(int)
    return main_data


def change_variables_to_binary_value(data_frame, goal_label, config_file_url):
    min_value = 0
    max_value = 1
    range_count = 5

    def convert_multiple_to_range(variable):
        if max_value - min_value <= range_count:
            return variable
        range_number = (max_value - min_value) / range_count
        variable -= min_value
        for index in range(range_count):
            if variable >= index * range_number:
                return index

    config = None
    if config_file_url is not None:
        with open(config_file_url) as json_file:
            config = json.load(json_file)
    data_frame_with_binary_values = pd.DataFrame()
    for label in data_frame.columns.values:
        min_value = data_frame[label].min()
        max_value = data_frame[label].max()
        if config is None or (config and config.get(label) is None):
            data_frame[label] = data_frame[label].apply(convert_multiple_to_range)
        nun = data_frame[label].nunique()
        if nun <= 2 or label == goal_label:
            data_frame.loc[data_frame[label] == max_value, label] = 1
            data_frame.loc[data_frame[label] == min_value, label] = 0
            data_frame_with_binary_values[label] = data_frame[label].copy()
        if nun > 2:
            for x in range(nun):
                new_label = label + str(x)
                data_frame.loc[data_frame[label] - min_value == x, new_label] = 1
                data_frame.loc[data_frame[label] - min_value != x, new_label] = 0
                data_frame_with_binary_values[new_label] = data_frame[new_label]
                data_frame_with_binary_values[new_label] = data_frame_with_binary_values[new_label].astype(int)
    return data_frame_with_binary_values, data_frame_with_binary_values.columns.get_loc(goal_label)


def create_formulas(data_frame, opposite_data_frame, literal_number, clause_number, formula_number):
    def create_positive_formula_array(data_frame, literal_number, clause_number, formula_number):
        row_number = len(data_frame)
        column_number = len(data_frame[0])
        formulas_array = []
        for form in range(formula_number):
            formula = []
            for clause in range(clause_number):
                clause = []
                column_used = []
                row_used = []
                row_index = random.randint(0, row_number - 1)
                column_index = random.randint(0, column_number - 1)
                for literal in range(literal_number):
                    while row_index in row_used:
                        row_index = random.randint(0, row_number - 1)
                    row_used.append(row_index)
                    while column_index in column_used:
                        column_index = random.randint(0, column_number - 1)
                    column_used.append(column_index)
                    values = data_frame[row_index]
                    clause.append([column_index, values[column_index]])
                formula.append(clause)
            formulas_array.append(formula)
        return formulas_array

    def create_negative_formula_array(df, literal_number, clause_number, formula_number):
        row_number = len(df)
        column_number = len(df[0])
        formulas_array = []
        for form in range(formula_number):
            formula = []
            for clause in range(clause_number):
                clause = []
                row_index = random.randint(0, row_number - 1)
                column_used = []
                column_index = random.randint(0, column_number - 1)
                for literal in range(literal_number):
                    while column_index in column_used:
                        column_index = random.randint(0, column_number - 1)
                    column_used.append(column_index)
                    values = df[row_index]
                    literal_value = 0
                    if values[column_index] == 0:
                        literal_value += 1
                    clause.append([column_index, literal_value])
                formula.append(clause)
            formulas_array.append(formula)
        return formulas_array

    first = create_positive_formula_array(data_frame, literal_number, clause_number, int(formula_number / 2))
    second = create_negative_formula_array(opposite_data_frame, literal_number, clause_number, int(formula_number / 2))
    return first + second


def split_data_frame_to_positives_and_negatives_rows(data_frame, new_goal_index):
    positive = []
    negative = []

    for x in range(len(data_frame)):
        row = data_frame.iloc[x, :].values.tolist()
        value_of_correct = row[new_goal_index]
        del row[new_goal_index]
        if value_of_correct == 1:
            positive.append(row)
        else:
            negative.append(row)

    return positive, negative


def create_array_with_init_value(array_length, init_value):
    array = []
    for index in range(array_length):
        array.append(init_value)
    return array


def check_formulas_correctness_on_data_frame(data_frame, formulas, goal_index, goal_label_value):
    formulas_correctness_count = np.zeros(len(data_frame))
    formulas_effective_count = np.zeros(len(formulas))
    formulas_statistic = np.zeros((len(formulas), 4))

    for y in range(len(data_frame)):
        x = data_frame.iloc[y, :].values.tolist()
        index = 0
        for form in formulas:
            form_is_false = False
            for clause in form:
                value_true = False
                for literal in clause:
                    if x[literal[0]] == literal[1]:
                        value_true = True
                        break
                if not value_true:
                    form_is_false = True
                    break
            if not form_is_false:
                formulas_correctness_count[y] += 1
                if bool(x[goal_index]) is goal_label_value:
                    formulas_effective_count[index] += 1
                    formulas_statistic[index][0] += 1
                else:
                    formulas_effective_count[index] -= 1
                    formulas_statistic[index][1] += 1
            else:
                if bool(x[goal_index]) is not goal_label_value:
                    formulas_effective_count[index] -= 1
                    formulas_statistic[index][2] += 1
                else:
                    formulas_effective_count[index] += 1
                    formulas_statistic[index][3] += 1
            index += 1
    return formulas_correctness_count, formulas_effective_count, formulas_statistic


def get_best_formulas(formulas, formulas_statistics):
    best_formulas = []

    for formula_index in range(len(formulas)):
        TP = formulas_statistics[formula_index][0]
        FP = formulas_statistics[formula_index][1]
        TN = formulas_statistics[formula_index][2]
        FN = formulas_statistics[formula_index][3]
        if (TP + TN) / (FP + FN) > 2:
            best_formulas.append(formulas[formula_index])
    return best_formulas


def create_files_with_formulas(literal_number, clause_number, multi_number, formulas_labels, formulas_array,
                               filename="formulas.cnf"):
    dirname = "exportedData"
    if not os.path.exists(dirname):
        try:
            os.mkdir(dirname)
        except OSError as e:
            return
    formulas_file = open(dirname + "/" + filename, "a")
    formulas_file.write("c " + filename + "\n")
    formulas_file.write("c" + "\n")
    for labels_index in range(len(formulas_labels)):
        formulas_file.write("c " + str(labels_index) + " = " + str(formulas_labels[labels_index]) + "\n")
    formulas_file.write("c" + "\n")
    formulas_file.write("p cnf " + str(literal_number) + " " + str(clause_number) + " " + str(multi_number) + "\n")
    for formulas in formulas_array:
        for clause in formulas:
            for literal in clause:
                if literal[1] == 0:
                    formulas_file.write("-" + str(literal[0]))
                else:
                    formulas_file.write(str(literal[0]))
                formulas_file.write(" ")
            formulas_file.write("\n")
    formulas_file.close()


def all():
    data = pd.read_csv(data_set_url)
    goal_label = data.columns.values[goal_column]
    main_data = data
    if column_numbers_array:
        labels = []
        for i in range(len(data.columns.values)):
            if i in column_numbers_array:
                labels.append(data.columns.values[i])
        labels.append(goal_label)
        main_data = data[labels]

    main_data = add_missing_values_or_removed(main_data, goal_label)
    if config_file_url is not None:
        main_data = add_changes_from_config_files(main_data, config_file_url)

    # Sprawdzenie poprawności danych w kolumnie celu
    if main_data[goal_label].dtype != 'int64' or main_data[goal_label].nunique() > 2:
        print("Goal column is not correct.")
        return 0

    main_data = add_tags_for_object_values(main_data, goal_label)
    main_data = change_values_to_integer(main_data, goal_label)
    main_data, new_goal_index = change_variables_to_binary_value(main_data, goal_label, config_file_url)

    train_data_frame, test_data_frame = train_test_split(main_data, test_size=0.5)
    positive_rows, negative_rows = split_data_frame_to_positives_and_negatives_rows(train_data_frame, new_goal_index)

    positive_formulas = []
    negative_formulas = []

    best_positive_formulas_end = False
    best_negative_formulas_end = False

    inter = 0
    while (not best_positive_formulas_end or not best_negative_formulas_end) and (inter < max_number_of_training_loop):
        inter += 1
        positive_formulas += create_formulas(positive_rows, negative_rows, literal_number_in_clause,
                                             clause_number_in_formula,
                                             int(formula_multiples_number - len(positive_formulas)))
        negative_formulas += create_formulas(negative_rows, positive_rows, literal_number_in_clause,
                                             clause_number_in_formula,
                                             int(formula_multiples_number - len(negative_formulas)))

        positive_correctness, positive_count, positive_formulas_statistic = check_formulas_correctness_on_data_frame(
            train_data_frame, positive_formulas, new_goal_index, True)
        negative_correctness, negative_count, negative_formulas_statistic = check_formulas_correctness_on_data_frame(
            train_data_frame, negative_formulas, new_goal_index, False)

        best_positive_formulas = get_best_formulas(positive_formulas, positive_formulas_statistic)
        best_negative_formulas = get_best_formulas(negative_formulas, negative_formulas_statistic)

        if len(best_positive_formulas) >= formula_multiples_number:
            best_positive_formulas_end = True
        if len(best_negative_formulas) >= formula_multiples_number:
            best_negative_formulas_end = True

        if inter < max_number_of_training_loop:
            positive_formulas = best_positive_formulas
            negative_formulas = best_negative_formulas

    positive_correctness, positive_count, positive_formulas_statistic = check_formulas_correctness_on_data_frame(
        test_data_frame, positive_formulas, new_goal_index, True)
    negative_correctness, negative_count, negative_formulas_statistic = check_formulas_correctness_on_data_frame(
        test_data_frame, negative_formulas, new_goal_index, False)

    effectiveness = 0
    for index in range(len(test_data_frame)):
        proposal_value = positive_correctness[index] > negative_correctness[index]
        x = test_data_frame.iloc[index, :].values.tolist()
        if bool(x[new_goal_index]) == proposal_value:
            effectiveness += 1
    precision_value = round(((effectiveness * 100) / len(test_data_frame)), 2)
    now = datetime.now()
    create_files_with_formulas(literal_number_in_clause, clause_number_in_formula, formula_multiples_number,
                               test_data_frame.columns.values, positive_formulas,
                               now.strftime("%Y%m%d%H%M%S") + "-positive_formulas.cnf")
    create_files_with_formulas(literal_number_in_clause, clause_number_in_formula, formula_multiples_number,
                               test_data_frame.columns.values, negative_formulas,
                               now.strftime("%Y%m%d%H%M%S") + "-negative_formulas.cnf")
    return precision_value


number_of_cases = 1
maximum_time = 0
minimum_time = -1
sum_time = 0
maximum_precision = 0
minimum_precision = 100
sum_precision = 0
predict_array = []
time_array = []
for x in range(number_of_cases):
    start_time = time.time()
    precision_value = all()
    end_time = time.time()
    total_time = end_time - start_time
    if total_time > maximum_time:
        maximum_time = total_time
    if total_time < minimum_time or minimum_time == -1:
        minimum_time = total_time
    sum_time += total_time
    if precision_value > maximum_precision:
        maximum_precision = precision_value
    if precision_value < minimum_precision:
        minimum_precision = precision_value
    sum_precision += precision_value
    predict_array.append(precision_value)
    time_array.append(total_time)

print("--- RESULTS FOR " + str(number_of_cases) + " CASES ---")
print("Minimum execution time: " + str(minimum_time))
print("Average execution time: " + str(sum_time / number_of_cases))
print("Maximum execution time: " + str(maximum_time))
print("Minimum precision: " + str(minimum_precision) + "%")
print("Average precision: " + str(round(sum_precision / number_of_cases)) + "%")
print("Maximum precision: " + str(maximum_precision) + "%")
print("---------------")
