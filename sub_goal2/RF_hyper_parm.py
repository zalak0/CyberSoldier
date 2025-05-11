from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator
from scipy.stats import randint
import warnings
import pandas as pd
import numpy as np

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
    
    return X, Y

def optim_param(rf, X_train, Y_train, X_test, Y_test):
    # 3. Define hyperparameter search space
    param_grid = {
        'n_estimators': [100, 200, 500],
        'max_depth': [10, 15, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }


    # 4. Set up randomized search
    random_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_grid,
        n_iter=25,                 # Try 50 random combinations
        cv=3,                      # 5-fold cross-validation
        scoring='accuracy',       # Use another metric if needed
        verbose=1,
        random_state=42,
        n_jobs=-1                 # Use all processors
    )
    
    # 5. Fit the search to the data
    random_search.fit(X_train, Y_train)

    # 6. Evaluate
    best_model = random_search.best_estimator_
    Y_pred = best_model.predict(X_test)
    print("Best Parameters:", random_search.best_params_)
    
    return Y_pred

def forest_classifier(csv_file, target):
    X, Y = file_initialise(csv_file, target)

    X_train, X_test, Y_train, Y_test = \
        train_test_split(X, Y, test_size = 0.2)

    forest = RandomForestClassifier()
    forest.fit(X_train, Y_train)
    
    return forest, X_train, Y_train, X_test, Y_test

warnings.filterwarnings("ignore")
targets = ["impact_confidentiality", "impact_availability", "impact_integrity"]
predictions = []
for target in targets:
    forest, X_train, Y_train, X_test, Y_test \
        = forest_classifier("data/cleaned_cve.csv", target)
    Y_pred = optim_param(forest, X_train, Y_train, X_test, Y_test)
    
    print(f"\nClassification Report for {target}:\n", classification_report(Y_test, Y_pred, digits=4, zero_division=0))
