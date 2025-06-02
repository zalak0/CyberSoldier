from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.multioutput import MultiOutputClassifier
from scipy.stats import randint
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def file_initialise(csv_file : str, targets : list) -> tuple[float]:
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
    
    le_vend = LabelEncoder()
    df_merged["vendor"] = le_vend.fit_transform(df_merged["vendor"])

    le_target = LabelEncoder()
    Y = df_merged[targets].apply(le_target.fit_transform)  # You can keep a list of encoders if needed
    Y_single = df_merged[targets[0]]

    X = df_merged[["cwe_code", "cvss", "access_complexity", "access_authentication", "vulnerable_product", "vendor"]]

    return X, Y, Y_single
    
def random_optim(rf, X_train, Y_train):
    # Selected parameters that will be analysed 
    param_grid = {
        'n_estimators': [100, 200, 500],
        'max_depth': [10, 15, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }

    # Set Up parameter optimiser and identifier
    random_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_grid,
        n_iter=50,                 # Try 50 random combinations
        cv=3,                      # 3-fold cross-validation
        scoring='accuracy',       # Use another metric if needed
        verbose=2,
        random_state=42,
        n_jobs=-1                 # Use all processors
    )
    
    # Train all the models
    random_search.fit(X_train, Y_train)
    
    return random_search.best_params_

def forest_optimiser(X_train, Y_train):
    forest = RandomForestClassifier()
    forest.fit(X_train, Y_train)
    
    # Extract best parameters
    best_params = random_optim(forest, X_train_sin, Y_train_sin)

    return best_params

def multi_forest_hyp_comp(targets, X_train, Y_train, X_test, Y_test, params):
    # Define and train the multi-output classifier
    b_clf = RandomForestClassifier()
    base_clf = MultiOutputClassifier(b_clf)
    base_clf.fit(X_train, Y_train)
    
    h_clf = RandomForestClassifier(**params)
    hyp_clf = MultiOutputClassifier(h_clf)
    hyp_clf.fit(X_train, Y_train)

    # Evaluate predictions for base and hyper
    Y_pred_base = base_clf.predict(X_test)
    Y_pred_df_base = pd.DataFrame(Y_pred_base, columns=targets)

    Y_pred_hyp = hyp_clf.predict(X_test)
    Y_pred_df_hyp = pd.DataFrame(Y_pred_hyp, columns=targets)
    
    acc_before = []
    acc_after = []
    
    for idx, target in enumerate(targets):
        acc_before.append(accuracy_score(Y_test[target], Y_pred_df_base[target]))
        acc_after.append(accuracy_score(Y_test[target], Y_pred_df_hyp[target]))

    return acc_before, acc_after

def plot_hyper_parm(acc_before: list, acc_after: list):
    # Plotting Accuracy Comparison
    x = np.arange(len(targets))
    width = 0.35

    fig, ax = plt.subplots()
    bars1 = ax.bar(x - width/2, acc_before, width, label='Before Optimization')
    bars2 = ax.bar(x + width/2, acc_after, width, label='After Optimization')

    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy Before vs After Hyperparameter Optimization')
    ax.set_xticks(x)
    ax.set_xticklabels(targets)
    ax.set_ylim(0.9, 1)
    ax.legend()

    for bar in bars1 + bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

    plt.tight_layout()
    plt.show()

# For plotting
targets = ["impact_confidentiality", "impact_integrity", "impact_availability"]

# Loading data to intialise X and Y
X, Y, Y_single = file_initialise("data/cleaned_cve.csv", targets)

# Split the data 60/20/20 split
X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, train_size=0.6, random_state=42)
X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)

# Split the data 60/20/20 split for single output for hyperparameterisation
X_train_sin, X_temp, Y_train_sin, Y_temp = train_test_split(X, Y_single, train_size=0.6, random_state=42)
X_val_sin, X_test_sin, Y_val_sin, Y_test_sin = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)

# Hyper parameterise using one of the targets and using single output classifier
best_params = forest_optimiser(X_train_sin, Y_train_sin)

# Use above hyper-parms for multi-classifier
# Extract accuracy before and after hyper-parameterisation
acc_before_list, acc_after_list = \
    multi_forest_hyp_comp(targets, X_train, Y_train, X_test, Y_test, best_params)
    
plot_hyper_parm(acc_before_list, acc_after_list)