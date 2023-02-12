import numpy as np

def encode_decimal_to_integers(df, target):
    float_columns = df.select_dtypes(include=[np.float64])

    for column in float_columns:
        if column == target:
            continue
        
        df[column] = df[column] * 100 
        df[column] = df[column].round(0).astype(np.int64)

    return df