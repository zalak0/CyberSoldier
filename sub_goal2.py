from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import make_scorer, accuracy_score, roc_auc_score, classification_report
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import warnings
import pandas as pd
import numpy as np

# def evaluate_model(model, X_test, y_test):
#     y_pred = model.predict(X_test)
#     y_proba = model.predict_proba(X_test)[:, 1]

#     acc = accuracy_score(y_test, y_pred)
#     auc = roc_auc_score(y_test, y_proba)

#     tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
#     specificity = tn / (tn + fp)

#     return acc, auc, specificity

def file_initialise(csv_file): 
    df = pd.read_csv(csv_file)
    df_cleaned = df.dropna()
    
    df_products = pd.read_csv('cleaned_products.csv')
    df_vendor_only = pd.read_csv('cleaned_vendors.csv')

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
    Y = df_merged["impact_availability"]
    #Y=df_merged["impact_integrity"]
    X = df_merged[["cwe_code", "cvss", "access_complexity", "access_authentication", "vulnerable_product", "vendor"]]
    
    return X, Y, df_merged

def predict_access_vector(knn_model, df, input_sample, label_encoder=None):
    """
    Predicts the access_vector for a single input sample using a trained KNN model.

    Parameters:
    - knn_model: trained KNeighborsClassifier model
    - input_sample: list or array of input features (same order as training data)
    - label_encoder: optional, to inverse-transform encoded labels if needed

    Returns:
    - predicted class (str or int depending on label encoder)
    """
    
    df_mapped = df[['cwe_code', 'cwe_name']].drop_duplicates()  # Remove duplicate cwe_code/cwe_name pairs
    df_with_names = pd.merge(df, df_mapped, on='cwe_code', how='left', suffixes=('', '_mapped'))
    
    if input_sample[0] in df['cwe_code'].values:
        cwe_name = df.loc[df['cwe_code'] == input_sample[0], 'cwe_name'].iloc[0]
        vendor = df.loc[df['cwe_code'] == input_sample[0], 'vendor'].iloc[0]
        acc_comp = df.loc[df['cwe_code'] == input_sample[0], 'access_complexity'].iloc[0]
        acc_auth = df.loc[df['cwe_code'] == input_sample[0], 'vendor'].iloc[0]
        

    # Reshape input to match expected shape: (1, n_features)
    input_array = np.array(input_sample).reshape(1, -1)

    # Make prediction
    prediction = knn_model.predict(input_array)
    
    #print(f"A vulnerability of {cwe_name} which has a vulnerability score of {input_sample[1]} from a {vendor} vendor, will result in a {prediction[0]} risk on data confidentiality")
    print(f"A vulnerability of {cwe_name} which has a vulnerability score of {input_sample[1]} from a {vendor} vendor, will result in a {prediction[0]} risk to system availability")
    #print(f"A vulnerability of {cwe_name} which has a vulnerability score of {input_sample[1]} from a {vendor} vendor, will result in a {prediction[0]} risk to data integrity")


def knn_classifier(csv_file):
    X, Y, df = file_initialise(csv_file)
    
    X_train, X_test, Y_train, Y_test = \
        train_test_split(X, Y, test_size = 0.2)

    knn = KNeighborsClassifier(n_neighbors=5, weights = "distance")
    knn.fit(X_train, Y_train)
    Y_pred = knn.predict(X_test)
    
    print("\nClassification Report using KNN:\n", classification_report(Y_test, Y_pred, digits=4, zero_division=0))
    
    return knn, df


def mlp_classifier(csv_file):
    X, Y, df = file_initialise(csv_file)
    
    X_train, X_test, Y_train, Y_test = \
        train_test_split(X, Y, test_size = 0.2)

    pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('mlp', MLPClassifier(hidden_layer_sizes=(20,), activation='logistic'))
    ])
    pipeline.fit(X_train, Y_train)
    Y_pred = pipeline.predict(X_test)  
    
    print("\nClassification Report using MLP: \n", classification_report(Y_test, Y_pred, digits=4, zero_division=0))
    
    return pipeline, df


def xg_classifier(csv_file):
    X, Y, df = file_initialise(csv_file)

    X_train, X_test, Y_train, Y_test = \
        train_test_split(X, Y, test_size = 0.2)

    xgboost = GradientBoostingClassifier()
    xgboost.fit(X_train, Y_train)
    Y_pred = xgboost.predict(X_test)  
    
    print("\nClassification Report using Gradient Boost:\n", classification_report(Y_test, Y_pred, digits=4, zero_division=0))
    
    return xgboost, df

def forest_classifier(csv_file):
    X, Y, df = file_initialise(csv_file)

    X_train, X_test, Y_train, Y_test = \
        train_test_split(X, Y, test_size = 0.2)

    forest = RandomForestClassifier()
    forest.fit(X_train, Y_train)
    Y_pred = forest.predict(X_test)  

    #mlp_results = evaluate_model(pipeline, X_test, Y_test)
    
    print("\nClassification Report for Random Forest:\n", classification_report(Y_test, Y_pred, digits=4, zero_division=0))
    
    return forest, df


def ens_classifier(csv_file):
    X, Y, df = file_initialise(csv_file)

    X_train, X_test, Y_train, Y_test = \
        train_test_split(X, Y, test_size = 0.2)
        
    # Example models
    knn = KNeighborsClassifier(n_neighbors=5)
    mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300)
    gbc = GradientBoostingClassifier()

    # Voting ensemble (can use 'hard' or 'soft' voting)
    ensemble = VotingClassifier(
        estimators=[('knn', knn), ('mlp', mlp), ('gbc', gbc)],
        voting='soft'  # 'soft' works better if all classifiers provide predict_proba
    )

    # Fit and evaluate
    ensemble.fit(X_train, Y_train)
    Y_pred = ensemble.predict(X_test)
    
    print("\nClassification Report for CWE Code:\n", classification_report(Y_test, Y_pred, digits=4, zero_division=0))
    
    return ensemble, df

warnings.filterwarnings("ignore")
knn, df = knn_classifier("cleaned_cve.csv")
mlp, df = mlp_classifier("cleaned_cve.csv")
xgboost, df = xg_classifier("cleaned_cve.csv")
forest, df = forest_classifier("cleaned_cve.csv")
#ensemble, df = ens_classifier("cleaned_cve.csv")
predict_access_vector(forest, df, [79, 5, 1, 1, 1, 4])
