import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer

def standardize_value(x, min_val, max_val, new_min=0, new_max=5):
    return round((x - min_val) * (new_max - new_min) / (max_val - min_val) + new_min)
    
def to_binary(df, target):
    new_df = pd.DataFrame() 

    for column in df.columns:
        if column == target:
            continue

        df[column] = df[column].apply(standardize_value, min_val=df[column].min(), max_val=df[column].max())

        encoder = LabelBinarizer()
        encoded = encoder.fit_transform(df[column])

        if len(encoder.classes_) > 2:
            binary_df = pd.DataFrame(encoded, columns=[column+str(class_) for class_ in encoder.classes_])
        else:
            binary_df = pd.DataFrame(encoded, columns=[column+"0"])
    
        if len(new_df) == 0:
            new_df = binary_df
        else:
            new_df = pd.concat([new_df, binary_df], axis=1)
    
    new_df[target] = df[target]

    return new_df


