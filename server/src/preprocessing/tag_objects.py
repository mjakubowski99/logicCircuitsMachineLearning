from sklearn.preprocessing import LabelEncoder

def tag_objects(df, target):
    max_unique = 5

    for column in df.columns:
        if column == target:
            continue

        if df[column].dtype != "object":
            continue

        if df[column].nunique() > max_unique:
            df = df.drop(columns=[column])
            continue 

        encoder = LabelEncoder()
        df[column] = encoder.fit_transform(df[column])

    return df 
