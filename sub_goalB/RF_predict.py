from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns
import warnings
import pandas as pd
import numpy as np
import shap
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
    
    print(X)
    return X, Y, df_merged, le_comp, le_auth, le_prod, le_vend

def multioutput_forest_classifier(csv_file, targets):
    # Load data
    X, _, df, le_comp, le_auth, le_prod, le_vend = file_initialise(csv_file, targets[0])
    
    le_target = LabelEncoder()
    # Encode all target labels
    Y = df[targets].apply(le_target.fit_transform)  # You can keep a list of encoders if needed

    # Split the data
    X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, train_size=0.6, random_state=42)
    X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)

    # Define and train the multi-output classifier
    base_clf = RandomForestClassifier(n_estimators =  200, 
                                      min_samples_split = 5, 
                                      min_samples_leaf = 1, 
                                      max_features = 'log2', 
                                      max_depth = 20
                                      )
    clf = MultiOutputClassifier(base_clf)
    clf.fit(X_train, Y_train)

    # Evaluate
    Y_pred = clf.predict(X_test)
    Y_pred_df = pd.DataFrame(Y_pred, columns=targets)

    for idx, target in enumerate(targets):
        print(f"\n=== Classification Report for {target} ===")
        print(classification_report(Y_test[target], Y_pred_df[target], digits = 4))
        print(confusion_matrix(Y_test[target], Y_pred_df[target]))

    return clf, X_train, X_test, Y_test, df, le_comp, le_auth, le_prod, le_vend, le_target

def forest_classifier(csv_file, target):
    X, Y, df, le_comp, le_auth, le_prod, le_vend = file_initialise(csv_file, target)

    # Encode target if it's categorical
    le_target = LabelEncoder()
    Y_encoded = le_target.fit_transform(Y)

    # Split the data
    X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y_encoded, train_size=0.6, random_state=42)
    X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)

    # Train model
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, Y_train)

    # Evaluate
    Y_pred = clf.predict(X_test)
    print(classification_report(Y_test, Y_pred, target_names=le_target.classes_))
    print(confusion_matrix(Y_test, Y_pred))

    # Graphs
    # graph_importance(clf, X_train)
    # plot_shap(clf, X_train)

    return clf, le_target, X_train, X_test, Y_test, df, le_comp, le_auth, le_prod, le_vend


def predict_multioutput(model, input_sample):
    prediction_idxs = model.predict(np.array(input_sample).reshape(1, -1))[0]  # One sample, multiple outputs
    
    pred_new = le_target.inverse_transform(prediction_idxs)
    
    # Decode input features
    cwe_name = df.loc[df['cwe_code'] == input_sample[0], 'cwe_name'].iloc[0] if input_sample[0] in df['cwe_code'].values else "Unknown CWE"
    access_comp = le_comp.inverse_transform([input_sample[2]])[0]
    access_auth = le_auth.inverse_transform([input_sample[3]])[0]
    product = le_prod.inverse_transform([input_sample[4]])[0]
    vendor = le_vend.inverse_transform([input_sample[5]])[0]

    input_conversion = [cwe_name, input_sample[1], access_comp, access_auth, product, vendor]
    return pred_new, input_conversion  # Return list of predictions


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
    for i in range(len(predictions[0])):
        target_name = targets[i].replace('impact_', '').capitalize()
        print(f"{target_name:<20}: {predictions[0][i]} risk")

def graph_importance(forest, X_train):
    # Feature Importance graph
    importances = pd.Series(forest.feature_importances_, index=X_train.columns)
    importances.sort_values(ascending=False).plot(kind='barh', figsize=(8, 5))
    plt.title("Feature Importance (Random Forest)")
    plt.xlabel("Absolute Weight (Importance)")
    plt.ylabel("Features")
    plt.xlim(0, 0.7)
    plt.tight_layout()
    plt.show()

def plot_shap(model, X_train):
    # Your existing code for data preparation
    print("Shape of X_train:", X_train.shape)
    print("Columns in X_train:", X_train.columns.tolist())

    if type(X_train) == pd.DataFrame:
        print("Yes is DataFrame")

    # Create TreeExplainer for your model
    explainer = shap.TreeExplainer(model)
    X_sample = X_train.sample(1000, random_state=42)

    print("Shape of X_sample:", X_sample.shape)
    print("Columns in X_sample:", X_sample.columns.tolist())

    # Calculate SHAP values
    shap_values = explainer.shap_values(X_sample)

    # Debug information
    print("SHAP values shape:", shap_values[0, :, 0].shape)
    print("Expected value:", explainer.expected_value)

    # Initialize SHAP JS visualizations
    shap.initjs()

    # Fix 1: Force plot for binary classification
    # For binary classification, use class 1 (positive class) SHAP values
    shap.force_plot(
        explainer.expected_value[1], 
        shap_values[0, :, 0],  # First sample, all features, class 1
        X_sample.iloc[0]
    )

    # Fix 2: Summary plot for binary classification
    # Use only the positive class SHAP values
    shap.summary_plot(shap_values[:, :, 0], X_sample)

    # Fix 3: Dependence plot for binary classification  
    shap.dependence_plot(
        "cvss", 
        shap_values[:, :, 0],  # All samples, all features, class 1
        X_sample
)
targets = ["impact_confidentiality", "impact_integrity", "impact_availability"]

clf, X_train, X_test, Y_test, df, le_comp, le_auth, le_prod, le_vend, le_target = \
    multioutput_forest_classifier("data/cleaned_cve.csv", targets)
input_sample = [79, 7, 1, 0, 4, 2]
prediction, input_conv= predict_multioutput(clf, input_sample)
print_prediction([prediction], targets, input_conv)
