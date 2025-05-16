from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, normalize
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import Input
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Avoid importing to_categorical since we're using sparse categorical encoding

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
    
    return X, Y, df_merged
    
X, Y, df = file_initialise('data/cleaned_cve.csv')    

# First: split 60% training, 40% temporary (to become val and test)
X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, train_size=0.6, random_state=42)

# Then: split the 40% temporary into 50/50 (i.e., 20% val, 20% test)
X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)
    
# Create MLPClassifier (scikit-learn's neural network)
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('mlp', MLPClassifier(hidden_layer_sizes=(100,),
                alpha=0.01,     # Increase this to apply stronger regularisation
                max_iter=300,
                random_state=42))
    ])

# Train model
pipeline.fit(X_train, Y_train)

# Evaluate
Y_pred = pipeline.predict(X_test)
train_acc = pipeline.score(X_train, Y_train)
test_acc = pipeline.score(X_test, Y_test)

print(f"Training Accuracy: {train_acc:.4f}")
print(f"Testing Accuracy:  {test_acc:.4f}")
print("\nClassification Report:")
print(classification_report(Y_test, Y_pred, digits = 4))

# No history to plot with scikit-learn, so we'll create a simple bar chart
plt.figure(figsize=(8, 5))
plt.bar(['Training', 'Testing'], [train_acc, test_acc], color=['blue', 'orange'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
for i, v in enumerate([train_acc, test_acc]):
    plt.text(i, v + 0.01, f"{v:.4f}", ha='center')
plt.tight_layout()
plt.show()

# Create confusion matrix
cm = confusion_matrix(Y_test, Y_pred, labels=pipeline.classes_)

# Display it
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=pipeline.classes_)
disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# 2. Scale entire data (fit only on training set!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)

# 3. Encode labels (once)
le = LabelEncoder()
Y_train_enc = to_categorical(le.fit_transform(Y_train))
Y_val_enc   = to_categorical(le.transform(Y_val))
Y_test_enc  = to_categorical(le.transform(Y_test))

print(X_train_scaled)      # Should be float32 or float64
print(Y_val_enc)      # Should be float32 or float64


# Build model
model = Sequential()
model.add(Input(shape=(X_train_scaled.shape[1],)))
model.add(Dense(100, activation='relu'))
model.add(BatchNormalization())     # <- BatchNorm
model.add(Dropout(0.3))             # Optional: to further prevent overfitting
model.add(Dense(Y_val_enc.shape[1], activation='softmax'))

# Compile
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train
history = model.fit(X_train_scaled, Y_train_enc,
          validation_data=(X_val_scaled, Y_val_enc),
          epochs=30,
          batch_size=32,
          callbacks=[EarlyStopping(patience=5, restore_best_weights=True)])

# Encode and one-hot the split Y sets to match model expectations
# Y_train_enc = to_categorical(le.transform(Y_train))
# Y_test_enc = to_categorical(le.transform(Y_test))

train_loss, train_acc = model.evaluate(X_train_scaled, Y_train_enc)
val_loss, val_acc     = model.evaluate(X_val_scaled, Y_val_enc)
test_loss, test_acc   = model.evaluate(X_test_scaled, Y_test_enc)

print(f"Train Accuracy: {train_acc:.4f}")
print(f"Val Accuracy:   {val_acc:.4f}")
print(f"Test Accuracy:  {test_acc:.4f}")

# Plot learning curves
import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title("Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()