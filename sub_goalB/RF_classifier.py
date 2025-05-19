from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def file_initialise(csv_file : str, target : str) -> tuple[float]:
    """ Opens files and initialises data-frame by encoding categorical features and initialising expected input and output

    Args:
        csv_file (str): string name of file
        target (str): specific output feature being analysed

    Returns:
        X: Data-frame of input features
        Y: Data-frame of output features
        df_merged: General data-frame
        le_comp: Encoder for access complexity
        le_auth: Encoder for access authentication
        le_prod: Encoder for product being hacked
        le_vend: Encoder for vendor protecting the product
    """
    df = pd.read_csv(csv_file)
    df = df.rename(columns={"Unnamed: 0": "cve_id"})
    df_cleaned = df.dropna()
    
    df_products = pd.read_csv('data/cleaned_products.csv')
    df_vendor_only = pd.read_csv('data/cleaned_vendors.csv').rename(columns={"Unnamed: 0": "cve_id"})

    df_products = df_products.drop_duplicates(subset='cve_id')
    df_vendor_only = df_vendor_only.drop_duplicates(subset='cve_id')

    # Merge df_main with products.csv on 'cve_id'
    df_merged = pd.merge(df_cleaned, df_products, on='cve_id', how='inner')

    # Merge with vendor-only file (may duplicate vendor column — that's okay)
    df_merged = pd.merge(df_merged, df_vendor_only[['cve_id', 'vendor']], on='cve_id', how='inner', suffixes=('', '_vendor2'))
    
    # Encode the vulnerable_products values
    le_prod = LabelEncoder()
    df_merged['vulnerable_product'] = le_prod.fit_transform(df_merged['vulnerable_product'])
    
    le_comp = LabelEncoder() 
    df_merged["access_complexity"] = le_comp.fit_transform(df_merged["access_complexity"])
    
    le_auth = LabelEncoder() 
    df_merged["access_authentication"] = le_auth.fit_transform(df_merged["access_authentication"])
    
    # le_conf = LabelEncoder()
    # df_merged["impact_confidentiality"] = le_conf.fit_transform(df_merged["impact_confidentiality"])
    
    le_vend = LabelEncoder()
    df_merged["vendor"] = le_vend.fit_transform(df_merged["vendor"])

    #print(df_merged[['cve_id','cwe_code', 'vendor']])
    #Y = df_merged["impact_confidentiality"]
    Y = df_merged[target]
    #Y=df_merged["impact_integrity"]
    X = df_merged[["cwe_code", "cvss", "access_complexity", "access_authentication", "vulnerable_product", "vendor"]]
    
    return X, Y, df_merged, le_comp, le_auth, le_prod, le_vend

def predict_access_vector(model : BaseEstimator, df : pd.DataFrame, input_sample : list, target : str, le_comp : LabelEncoder, 
                          le_auth : LabelEncoder, le_prod : LabelEncoder, le_vend : LabelEncoder):
    """
    Predicts the risk impact level of a vulnerability on a specified target based on input features.

    Args:
        model (sklearn.base.BaseEstimator): A trained classification model that predicts the risk level.
        df (pd.DataFrame): The main dataset containing vulnerability information, including CWE codes and names.
        input_sample (list or array-like): A list containing input features in the following order:
            [cwe_code (int), cvss_score (float), access_complexity (int), authentication (int), product (int), vendor (int)].
        target (str): The impact target to assess (e.g., 'impact_confidentiality', 'impact_integrity').
        le_comp (sklearn.preprocessing.LabelEncoder): LabelEncoder used to encode access complexity.
        le_auth (sklearn.preprocessing.LabelEncoder): LabelEncoder used to encode authentication.
        le_prod (sklearn.preprocessing.LabelEncoder): LabelEncoder used to encode the product.
        le_vend (sklearn.preprocessing.LabelEncoder): LabelEncoder used to encode the vendor.

    Returns:
        tuple:
            - prediction (str): The predicted risk level (e.g., 'Low', 'Medium', 'High').
            - input_conversion (list): A human-readable version of the input features for interpretation.
    """
    target_name = target.replace('impact_', '')
    df_mapped = df[['cwe_code', 'cwe_name']].drop_duplicates()  # Remove duplicate cwe_code/cwe_name pairs
    df_with_names = pd.merge(df, df_mapped, on='cwe_code', how='left', suffixes=('', '_mapped'))
    cwe_name = df.loc[df['cwe_code'] == input_sample[0], 'cwe_name'].iloc[0] if input_sample[0] in df['cwe_code'].values else "Unknown CWE"
    access_comp = le_comp.inverse_transform([input_sample[2]])[0]
    access_auth = le_auth.inverse_transform([input_sample[3]])[0]
    product = le_prod.inverse_transform([input_sample[4]])[0]
    vendor = le_vend.inverse_transform([input_sample[5]])[0]
    prediction = model.predict(np.array(input_sample).reshape(1, -1))[0]
    
    input_conversion = [cwe_name, input_sample[1], access_comp, access_auth, product, vendor]
        
    # print(
    #     f"The vulnerability '{cwe_name}' affecting the product '{product}' from vendor '{vendor}'"
    #     f"with a CVSS score of {input_sample[1]}, which has '{access_comp}' access complexity and '{access_auth}' authentication,"
    #     f"is predicted to pose a '{prediction}' risk to {target_name}.")
    
    return prediction, input_conversion

def print_prediction(predictions: list, targets: list, input_conversion: list):
    """
    Prints the predictions in a human-readable format after classification.

    Args:
        predictions (list): List of predictions for the output features in order.
        targets (list): List of target feature names that the model is predicting.
        input_conversion (list): Human-readable version of the input features.
    """
    
    print("=== Input Features ===")
    print(f"Vulnerability           : {input_conversion[0]}")
    print(f"CVSS Score              : {input_conversion[1]}")
    print(f"Access Complexity       : {input_conversion[2]}")
    print(f"Access Authentication   : {input_conversion[3]}")
    print(f"Product                 : {input_conversion[4]}")
    print(f"Vendor                  : {input_conversion[5]}")
    
    print("\n=== Predicted Risk Impacts ===")
    for target, prediction in zip(targets, predictions):
        target_name = target.replace('impact_', '').capitalize()
        print(f"{target_name:<20}: {prediction} risk")

def graph_importance(X_train):
    # Feature Importance graph
    importances = pd.Series(forest.feature_importances_, index=X_train.columns)
    importances.sort_values(ascending=False).plot(kind='barh', figsize=(8, 5))
    plt.title("Feature Importance (Random Forest)")
    plt.xlabel("Absolute Weight (Importance)")
    plt.ylabel("Features")
    plt.xlim(0, 0.7)
    plt.tight_layout()
    plt.show()
    

def forest_classifier(csv_file, target):
    X, Y, df, le_comp, le_auth, le_prod, le_vend = file_initialise(csv_file, target)

    # First: split 60% training, 40% temporary (to become val and test)
    X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, train_size=0.6, random_state=42)

    # Then: split the 40% temporary into 50/50 (i.e., 20% val, 20% test)
    X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)
    
    forest = RandomForestClassifier()
    forest.fit(X_train, Y_train)
    Y_pred = forest.predict(X_test)  

    graph_importance(X_train)
    
    print(f"\nClassification Report for {target}:\n", classification_report(Y_test, Y_pred, digits=4, zero_division=0))
    
    return forest, df, le_comp, le_auth, le_prod, le_vend

warnings.filterwarnings("ignore")
targets = ["impact_confidentiality", "impact_availability", "impact_integrity"]
predictions = []
for target in targets:
    forest, df_merged, le_comp, le_auth, le_prod, le_vend = forest_classifier("data/cleaned_cve.csv", target)

    # Input sample structure ["cwe_code", "cvss", "access_complexity", "access_authentication", "vulnerable_product", "vendor"]
    input_sample = [79, 7, 1, 0, 40, 10]
    pred, input_conversion = predict_access_vector(forest, df_merged, input_sample, target, le_comp, le_auth, le_prod, le_vend)
    predictions.append(pred)

print_prediction(predictions, targets, input_conversion)