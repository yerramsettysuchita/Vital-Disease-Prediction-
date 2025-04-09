import pandas as pd
df_loaded = pd.read_csv("C:/Users/DELL/Downloads/Vital Disease Prediction/vital_disease_prediction_dataset.csv")

# Basic data analysis
summary = {
    "Shape": df_loaded.shape,
    "Columns": df_loaded.columns.tolist(),
    "Sample Rows": df_loaded.head(),
    "Null Values": df_loaded.isnull().sum(),
    "Disease Distribution": df_loaded["Disease_Prediction"].value_counts().head(10)
}
print(summary)
