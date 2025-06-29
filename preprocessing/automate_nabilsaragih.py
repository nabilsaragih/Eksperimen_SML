from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib
import numpy as np
import os

def preprocess_data(data, target_column, save_path, file_path):
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)


    data.drop_duplicates(inplace=True)
    data.dropna(inplace=True)

    binary_map = {
        "Stage_fear": {"Yes": 1, "No": 0},
        "Drained_after_socializing": {"Yes": 1, "No": 0},
        target_column: {"Extrovert": 1, "Introvert": 0}
    }
    for col, mapping in binary_map.items():
        if col in data.columns:
            data[col] = data[col].map(mapping)

    numeric_features = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
    if target_column in numeric_features:
        numeric_features.remove(target_column)

    header_df = pd.DataFrame(columns=data.columns.drop(target_column))
    header_df.to_csv(file_path, index=False)
    print(f"Header kolom disimpan di: {file_path}")

    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features)
        ],
        remainder='passthrough'
    )

    X = data.drop(columns=[target_column])
    y = data[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    X_train = pd.DataFrame(X_train, columns=numeric_features + [col for col in X.columns if col not in numeric_features])
    X_test = pd.DataFrame(X_test, columns=numeric_features + [col for col in X.columns if col not in numeric_features])

    X_train.to_csv("output/X_train_scaled.csv", index=False)
    X_test.to_csv("output/X_test_scaled.csv", index=False)

    joblib.dump(preprocessor, save_path)
    print(f"Pipeline disimpan di: {save_path}")

    return X_train, X_test, y_train, y_test

def inference(data, preprocessor_path):
    preprocessor = joblib.load(preprocessor_path)
    print(f"Pipeline dimuat dari: {preprocessor_path}")

    transformed_data = preprocessor.transform(data)
    return transformed_data

def inverse_transform(transformed_data, preprocessor_path, original_columns):
    preprocessor = joblib.load(preprocessor_path)
    
    numeric_transformer = preprocessor.named_transformers_['num']['scaler']
    num_cols_len = len(numeric_transformer.mean_)

    numeric_columns = original_columns[:num_cols_len]
    original_numeric_data = numeric_transformer.inverse_transform(transformed_data[:, :num_cols_len])
    inversed_df = pd.DataFrame(original_numeric_data, columns=numeric_columns)

    print("Berhasil inverse transform data.")
    return inversed_df



df = pd.read_csv('personality/personality_datasert.csv')
X_train, X_test, y_train, y_test = preprocess_data(
    data=df,
    target_column='Personality',
    save_path='output/preprocessor_nabilsaragih.joblib',
    file_path='output/ekstro_intro_preprocessing.csv'
)

pipeline_path = 'output/preprocessor_nabilsaragih.joblib'
col = pd.read_csv('output/ekstro_intro_preprocessing.csv')

new_data = df.drop(columns=['Personality']).iloc[0].values
new_data = pd.DataFrame([new_data], columns=col.columns)

transformed_data = inference(new_data, pipeline_path)

inversed_data = inverse_transform(transformed_data, pipeline_path, new_data.columns)

print("\nData asli:")
print(new_data)
print("\nSetelah transformasi (scaling):")
print(transformed_data)
print("\nSetelah inverse transform (kembali ke skala awal):")
print(inversed_data)
