from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from scipy.stats import randint
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def file_initialise(csv_file : str, targets : str) -> tuple[float]:
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
    df_cleaned = df.dropna()
    
    df_products = pd.read_csv('data/cleaned_products.csv')
    df_vendor_only = pd.read_csv('data/cleaned_vendors.csv')

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

    Y = df_merged[[target]]

    X = df_merged[["cwe_code", "cvss", "access_complexity", "access_authentication", "vulnerable_product", "vendor"]]

    return X, Y, df_merged

def optim_param(rf, X_train, Y_train, X_test, Y_test):
    param_grid = {
        'n_estimators': [100, 300, 500],
        'max_depth': [10, 15, 20],
        'min_samples_leaf': [2, 4],
        'criterion': ['gini', 'entropy'],
    }

    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=5,  # 5-fold cross-validation
        scoring='accuracy',  # or 'accuracy', 'roc_auc', etc.
        n_jobs=-1,  # Use all available CPUs
        verbose=2
    )

    # 5. Fit the search to the data
    grid_search.fit(X_train, Y_train)

    # 6. Evaluate
    best_model = grid_search.best_estimator_
    Y_pred = best_model.predict(X_test)
    print("Best Parameters:", grid_search.best_params_)
    
def random_optim(rf, X_train, Y_train, X_test, Y_test):
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
        n_iter=50,                 # Try 50 random combinations
        cv=3,                      # 3-fold cross-validation
        scoring='accuracy',       # Use another metric if needed
        verbose=2,
        random_state=42,
        n_jobs=-1                 # Use all processors
    )
    
    # 5. Fit the search to the data
    random_search.fit(X_train, Y_train)

    # 6. Evaluate
    best_model = random_search.best_estimator_
    Y_pred = best_model.predict(X_test)
    print("Best Parameters:", random_search.best_params_)
    print("Test Accuracy:", accuracy_score(Y_test, Y_pred))
        
    return Y_pred

def forest_classifier(csv_file, target):
    X, Y = file_initialise(csv_file, target)
    
    # Split the data
    X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, train_size=0.6, random_state=42)
    X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)

    forest = RandomForestClassifier()
    forest.fit(X_train, Y_train)

    # Predict using default parameters (before optimization)
    Y_pred_before = forest.predict(X_test)
    acc_before = accuracy_score(Y_test, Y_pred_before)

    return forest, X_train, Y_train, X_val, Y_val, acc_before

# For plotting
targets = ["impact_confidentiality", "impact_availability", "impact_integrity"]
acc_before_list = []
acc_after_list = []

for target in targets:
    forest, X_train, Y_train, X_val, Y_val, acc_before = forest_classifier("data/cleaned_cve.csv", target)
    Y_pred_after = random_optim(forest, X_train, Y_train, X_val, Y_val)
    acc_after = accuracy_score(Y_val, Y_pred_after)

    acc_before_list.append(acc_before)
    acc_after_list.append(acc_after)

    print(f"\nClassification Report for {target}:\n{classification_report(Y_val, Y_pred_after, digits=4, zero_division=0)}")
    print(f"Accuracy Before Optimization: {acc_before:.4f}")
    print(f"Accuracy After Optimization:  {acc_after:.4f}")

# Plotting Accuracy Comparison
x = np.arange(len(targets))
width = 0.35

fig, ax = plt.subplots()
bars1 = ax.bar(x - width/2, acc_before_list, width, label='Before Optimization')
bars2 = ax.bar(x + width/2, acc_after_list, width, label='After Optimization')

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