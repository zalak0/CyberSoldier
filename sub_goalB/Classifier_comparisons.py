from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import pandas as pd
import numpy as np

def plot_pca(X, Y):
    # PCA graph to show data clusters and data separability
    X_scaled = StandardScaler().fit_transform(X)
    X_pca = PCA(n_components=2).fit_transform(X_scaled)
    
    le = LabelEncoder()
    Y_encoded = le.fit_transform(Y)

    plt.scatter(X_pca[:,0], X_pca[:,1], c=Y_encoded, cmap='viridis', alpha=0.5)
    plt.title("PCA Visualization Colored by Target")
    plt.colorbar(label="Encoded Target")

    # Example using 3 classes
    labels = le.classes_  # ['COMPLETE', 'INCOMPLETE', 'PENDING']
    colors = plt.cm.viridis([0, 0.5, 1])
    handles = [mpatches.Patch(color=colors[i], label=labels[i]) for i in range(len(labels))]
    plt.legend(handles=handles)
    plt.show()
    
def plot_corr_matrix(df):
    # Correlation Matrix (Pre-Processing)
    corr_matrix = df.corr(numeric_only=True)  # Only numeric columns

    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.show()

def plot_class_comp(X_train, X_test, Y_train, Y_test):
    acc_knn, f1_kn = knn_classifier(X_train, X_test, Y_train, Y_test)
    acc_mlp, f1_mlp = mlp_classifier(X_train, X_test, Y_train, Y_test)
    acc_gb, f1_gb = gboost_classifier(X_train, X_test, Y_train, Y_test)
    acc_frst, f1_frst = forest_classifier(X_train, X_test, Y_train, Y_test)
    acc_log_reg, f1_log_reg = log_reg_classifier(X_train, X_test, Y_train, Y_test)
    acc_svm, f1_svm = svm_classifier(X_train, X_test, Y_train, Y_test)
    # predict_access_vector(forest, df, [79, 5, 1, 1, 1, 4])

    # Classifier names
    models = ['KNN', 'MLP', 'GBoost', 'Random Forest', 'Logistic Regression', 'SVM']

    # Accuracy and F1-score values
    accuracy = [acc_knn, acc_mlp, acc_gb, acc_frst, acc_log_reg, acc_svm]
    f1_score = [f1_kn, f1_mlp, f1_gb, f1_frst, f1_log_reg, f1_svm]

    # accuracy = [acc_log_reg, acc_svm]
    # f1_score = [f1_log_reg, f1_svm]
    
    # Set position of bar on X axis
    x = np.arange(len(models))
    width = 0.35  # Width of the bars

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, accuracy, width, label='Accuracy', color='skyblue')
    bars2 = ax.bar(x + width/2, f1_score, width, label='F1 Score', color='salmon')

    # Labels and title
    ax.set_ylabel('Score')
    ax.set_title('Classifier Accuracy and F1 Score Performance')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0.5, 1.05)
    ax.legend()

    # Annotate bars
    for bar in bars1 + bars2:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f'{yval:.4f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.show()

def file_initialise(csv_file): 
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
    
    # le_conf = LabelEncoder()
    # df_merged["impact_confidentiality"] = le_conf.fit_transform(df_merged["impact_confidentiality"])
    
    le_vend = LabelEncoder()
    df_merged["vendor"] = le_vend.fit_transform(df_merged["vendor"])

    #print(df_merged[['cve_id','cwe_code', 'vendor']])
    #Y = df_merged["impact_confidentiality"]
    Y = df_merged["impact_availability"]
    #Y=df_merged["impact_integrity"]
    X = df_merged[["cwe_code", "cvss", "access_complexity", "access_authentication", "vulnerable_product", "vendor"]]
    
    plot_corr_matrix(df_merged)
    
    plot_pca(X, Y)

    return X, Y, df_merged

def knn_classifier(X_train, X_test, Y_train, Y_test):
    knn = KNeighborsClassifier(n_neighbors=5, weights = "distance")
    
    knn.fit(X_train, Y_train)
    Y_pred = knn.predict(X_test)
    acc = accuracy_score(Y_test, Y_pred)
    f1 = f1_score(Y_test, Y_pred, average='weighted')

    
    print("\nClassification Report using KNN:\n", classification_report(Y_test, Y_pred, digits=4, zero_division=0))
    
    return acc, f1


def mlp_classifier(X_train, X_test, Y_train, Y_test):
    pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('mlp', MLPClassifier(hidden_layer_sizes=(100,),
                    max_iter=500,
                    random_state=42)
)
    ])
    pipeline.fit(X_train, Y_train)
    Y_pred = pipeline.predict(X_test)  
    acc = accuracy_score(Y_test, Y_pred)
    f1 = f1_score(Y_test, Y_pred, average='weighted')
    
    print("\nClassification Report using MLP: \n", classification_report(Y_test, Y_pred, digits=4, zero_division=0))
    
    return acc, f1


def gboost_classifier(X_train, X_test, Y_train, Y_test):
    gboost = GradientBoostingClassifier()
    gboost.fit(X_train, Y_train)
    Y_pred = gboost.predict(X_test)  
    acc = accuracy_score(Y_test, Y_pred)
    f1 = f1_score(Y_test, Y_pred, average='weighted')

    
    print("\nClassification Report using Gradient Boost:\n", classification_report(Y_test, Y_pred, digits=4, zero_division=0))
    
    return acc, f1

def forest_classifier(X_train, X_test, Y_train, Y_test):

    forest = RandomForestClassifier()
    forest.fit(X_train, Y_train)
    Y_pred = forest.predict(X_test)
    acc = accuracy_score(Y_test, Y_pred)
    f1 = f1_score(Y_test, Y_pred, average='weighted')


    #mlp_results = evaluate_model(pipeline, X_test, Y_test)
    
    print("\nClassification Report for Random Forest:\n", classification_report(Y_test, Y_pred, digits=4, zero_division=0))
    
    return acc, f1

def log_reg_classifier(X_train, X_test, Y_train, Y_test):
    log_reg = LogisticRegression(max_iter= 20000)

    log_reg.fit(X_train, Y_train)
    Y_pred = log_reg.predict(X_test)
    acc = accuracy_score(Y_test, Y_pred)
    f1 = f1_score(Y_test, Y_pred, average='weighted')

    print("\nClassification Report for Logistic Regression:\n", \
        classification_report(Y_test, Y_pred, digits=4, zero_division=0))
    
    return acc, f1

def svm_classifier(X_train, X_test, Y_train, Y_test):

    svm = SVC()

    scaler = StandardScaler()
    # 2. Scale entire data (fit only on training set!)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # 3. Encode labels (once)
    le = LabelEncoder()
    Y_train_enc = le.fit_transform(Y_train)
    Y_test_enc  = le.transform(Y_test)

    # Fit and evaluate
    svm.fit(X_train_scaled, Y_train_enc)
    Y_pred = svm.predict(X_test_scaled)
    acc = accuracy_score(Y_test_enc, Y_pred)
    f1 = f1_score(Y_test_enc, Y_pred, average='weighted')

    
    print("\nClassification Report for CWE Code:\n", classification_report(Y_test_enc, Y_pred, digits=4, zero_division=0))
    
    return acc, f1


X, Y, df = file_initialise("data/cleaned_cve.csv")

X_train, X_temp, Y_train, Y_temp = \
    train_test_split(X, Y, train_size = 0.6)

X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, train_size = 0.5)

plot_class_comp(X_train, X_test, Y_train, Y_test)

