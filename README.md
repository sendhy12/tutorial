# Tutorial Lengkap: Tugas Machine Learning 4 Kriteria

## Gambaran Umum

Tutorial ini akan membantu Anda menyelesaikan tugas Machine Learning yang terdiri dari 4 kriteria utama:
1. **Eksperimen Dataset** - Eksplorasi dan preprocessing data
2. **Membangun Model ML** - Training model dengan MLflow
3. **Workflow CI** - Continuous Integration dengan GitHub Actions
4. **Monitoring & Logging** - Prometheus dan Grafana

---

## 🔬 KRITERIA 1: Melakukan Eksperimen terhadap Dataset Pelatihan

### Persiapan Awal

1. **Buat Repository GitHub**
   ```bash
   # Buat folder lokal
   mkdir Eksperimen_SML_[Nama-Anda]
   cd Eksperimen_SML_[Nama-Anda]
   
   # Inisialisasi Git
   git init
   git remote add origin https://github.com/username/Eksperimen_SML_[Nama-Anda].git
   ```

2. **Struktur Folder**
   ```
   Eksperimen_SML_Nama-siswa/
   ├── .github/workflows/          # untuk advance
   ├── dataset_raw/               # dataset mentah
   ├── preprocessing/
   │   ├── Eksperimen_Nama-siswa.ipynb
   │   ├── automate_Nama-siswa.py  # untuk skilled
   │   └── dataset_preprocessing/  # hasil preprocessing
   ```

### Langkah 1: Eksplorasi Data (Basic - 2 pts)

**Buat Notebook Eksperimen**

```python
# Eksperimen_[Nama-Anda].ipynb

# 1. DATA LOADING
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv('../dataset_raw/your_dataset.csv')
print("Dataset Shape:", df.shape)
print("\nDataset Info:")
df.info()

# 2. EXPLORATORY DATA ANALYSIS (EDA)
print("\n=== BASIC STATISTICS ===")
print(df.describe())

print("\n=== MISSING VALUES ===")
print(df.isnull().sum())

print("\n=== DATA TYPES ===")
print(df.dtypes)

# Visualisasi distribusi
plt.figure(figsize=(12, 8))
for i, col in enumerate(df.select_dtypes(include=[np.number]).columns[:4]):
    plt.subplot(2, 2, i+1)
    plt.hist(df[col], bins=30, alpha=0.7)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# Correlation matrix
plt.figure(figsize=(10, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix')
plt.show()

# 3. DATA PREPROCESSING
# Handle missing values
df_clean = df.copy()

# Contoh: Fill missing values
for col in df_clean.select_dtypes(include=[np.number]).columns:
    df_clean[col].fillna(df_clean[col].median(), inplace=True)

for col in df_clean.select_dtypes(include=['object']).columns:
    df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)

# Encoding categorical variables
label_encoders = {}
for col in df_clean.select_dtypes(include=['object']).columns:
    if col != 'target_column':  # ganti dengan nama target Anda
        le = LabelEncoder()
        df_clean[col] = le.fit_transform(df_clean[col])
        label_encoders[col] = le

# Feature scaling
scaler = StandardScaler()
numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
df_clean[numeric_cols] = scaler.fit_transform(df_clean[numeric_cols])

# Split data
X = df_clean.drop('target_column', axis=1)  # ganti dengan target Anda
y = df_clean['target_column']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training set shape:", X_train.shape)
print("Test set shape:", X_test.shape)

# Save preprocessed data
X_train.to_csv('../preprocessing/X_train.csv', index=False)
X_test.to_csv('../preprocessing/X_test.csv', index=False)
y_train.to_csv('../preprocessing/y_train.csv', index=False)
y_test.to_csv('../preprocessing/y_test.csv', index=False)
```

### Langkah 2: Otomatisasi Preprocessing (Skilled - 3 pts)

**Buat File automate_[Nama-Anda].py**

```python
# automate_[Nama-Anda].py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
import os

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def load_data(self, file_path):
        """Load raw dataset"""
        return pd.read_csv(file_path)
    
    def handle_missing_values(self, df):
        """Handle missing values"""
        df_clean = df.copy()
        
        # Numeric columns: fill with median
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df_clean[col].fillna(df_clean[col].median(), inplace=True)
            
        # Categorical columns: fill with mode
        categorical_cols = df_clean.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
            
        return df_clean
    
    def encode_categorical(self, df, target_column):
        """Encode categorical variables"""
        df_encoded = df.copy()
        
        for col in df_encoded.select_dtypes(include=['object']).columns:
            if col != target_column:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col])
                self.label_encoders[col] = le
                
        return df_encoded
    
    def scale_features(self, X_train, X_test):
        """Scale numerical features"""
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled
    
    def preprocess_data(self, file_path, target_column, test_size=0.2):
        """Complete preprocessing pipeline"""
        print("Loading data...")
        df = self.load_data(file_path)
        
        print("Handling missing values...")
        df_clean = self.handle_missing_values(df)
        
        print("Encoding categorical variables...")
        df_encoded = self.encode_categorical(df_clean, target_column)
        
        print("Splitting data...")
        X = df_encoded.drop(target_column, axis=1)
        y = df_encoded[target_column]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        print("Scaling features...")
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)
        
        # Convert back to DataFrame
        X_train_df = pd.DataFrame(X_train_scaled, columns=X.columns)
        X_test_df = pd.DataFrame(X_test_scaled, columns=X.columns)
        
        return X_train_df, X_test_df, y_train, y_test
    
    def save_preprocessed_data(self, X_train, X_test, y_train, y_test, output_dir):
        """Save preprocessed data"""
        os.makedirs(output_dir, exist_ok=True)
        
        X_train.to_csv(f"{output_dir}/X_train.csv", index=False)
        X_test.to_csv(f"{output_dir}/X_test.csv", index=False)
        y_train.to_csv(f"{output_dir}/y_train.csv", index=False)
        y_test.to_csv(f"{output_dir}/y_test.csv", index=False)
        
        # Save preprocessing objects
        with open(f"{output_dir}/scaler.pkl", 'wb') as f:
            pickle.dump(self.scaler, f)
        with open(f"{output_dir}/label_encoders.pkl", 'wb') as f:
            pickle.dump(self.label_encoders, f)
        
        print(f"Preprocessed data saved to {output_dir}")

# Main execution
if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    
    # Sesuaikan dengan dataset Anda
    raw_data_path = "../dataset_raw/your_dataset.csv"
    target_column = "target_column_name"
    output_directory = "./dataset_preprocessing"
    
    # Run preprocessing
    X_train, X_test, y_train, y_test = preprocessor.preprocess_data(
        raw_data_path, target_column
    )
    
    # Save results
    preprocessor.save_preprocessed_data(
        X_train, X_test, y_train, y_test, output_directory
    )
    
    print("Preprocessing completed successfully!")
```

### Langkah 3: GitHub Actions Workflow (Advanced - 4 pts)

**Buat .github/workflows/preprocess.yml**

```yaml
name: Data Preprocessing Workflow

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]
  workflow_dispatch:

jobs:
  preprocess:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v3
      with:
        python-version: '3.8'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install pandas numpy scikit-learn matplotlib seaborn
    
    - name: Run preprocessing
      run: |
        cd preprocessing
        python automate_[Nama-Anda].py
    
    - name: Upload preprocessed data
      uses: actions/upload-artifact@v3
      with:
        name: preprocessed-data
        path: preprocessing/dataset_preprocessing/
```

---

## 🤖 KRITERIA 2: Membangun Model Machine Learning

### Persiapan Struktur Folder

```
Membangun_model/
├── modelling.py
├── modelling_tuning.py          # untuk skilled/advanced
├── dataset_preprocessing/       # dari kriteria 1
├── screenshoot_dashboard.jpg
├── screenshoot_artifak.jpg
├── requirements.txt
├── DagsHub.txt                 # untuk advanced
```

### Langkah 1: Setup MLflow (Basic - 2 pts)

**requirements.txt**
```
mlflow
scikit-learn
pandas
numpy
matplotlib
seaborn
```

**modelling.py (Basic)**
```python
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Enable autologging
mlflow.sklearn.autolog()

def load_data():
    """Load preprocessed data"""
    X_train = pd.read_csv('./dataset_preprocessing/X_train.csv')
    X_test = pd.read_csv('./dataset_preprocessing/X_test.csv')
    y_train = pd.read_csv('./dataset_preprocessing/y_train.csv').values.ravel()
    y_test = pd.read_csv('./dataset_preprocessing/y_test.csv').values.ravel()
    
    return X_train, X_test, y_train, y_test

def train_model():
    """Train model with MLflow tracking"""
    
    # Set experiment
    mlflow.set_experiment("Basic_ML_Experiment")
    
    with mlflow.start_run():
        # Load data
        X_train, X_test, y_train, y_test = load_data()
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy}")
        
        # MLflow will automatically log model and metrics
        print("Model training completed with autolog!")

if __name__ == "__main__":
    train_model()
```

### Langkah 2: Hyperparameter Tuning (Skilled - 3 pts)

**modelling_tuning.py (Skilled)**
```python
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

def load_data():
    """Load preprocessed data"""
    X_train = pd.read_csv('./dataset_preprocessing/X_train.csv')
    X_test = pd.read_csv('./dataset_preprocessing/X_test.csv')
    y_train = pd.read_csv('./dataset_preprocessing/y_train.csv').values.ravel()
    y_test = pd.read_csv('./dataset_preprocessing/y_test.csv').values.ravel()
    
    return X_train, X_test, y_train, y_test

def train_model_with_tuning():
    """Train model with hyperparameter tuning and manual logging"""
    
    # Set experiment
    mlflow.set_experiment("Skilled_ML_Experiment")
    
    with mlflow.start_run():
        # Load data
        X_train, X_test, y_train, y_test = load_data()
        
        # Hyperparameter tuning
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10]
        }
        
        # Grid search
        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        
        # Best model
        best_model = grid_search.best_estimator_
        
        # Predictions
        y_pred = best_model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Manual logging (same as autolog)
        mlflow.log_param("n_estimators", best_model.n_estimators)
        mlflow.log_param("max_depth", best_model.max_depth)
        mlflow.log_param("min_samples_split", best_model.min_samples_split)
        
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        
        # Log model
        mlflow.sklearn.log_model(best_model, "model")
        
        # Log best parameters
        mlflow.log_params(grid_search.best_params_)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1}")

if __name__ == "__main__":
    train_model_with_tuning()
```

### Langkah 3: Setup DagsHub (Advanced - 4 pts)

**Buat akun DagsHub dan setup**

1. Daftar di [DagsHub.com](https://dagshub.com)
2. Buat repository baru
3. Dapatkan MLflow tracking URI

**modelling_tuning.py (Advanced)**
```python
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Setup DagsHub
os.environ['MLFLOW_TRACKING_URI'] = 'https://dagshub.com/YOUR_USERNAME/YOUR_REPO.mlflow'
os.environ['MLFLOW_TRACKING_USERNAME'] = 'YOUR_USERNAME'
os.environ['MLFLOW_TRACKING_PASSWORD'] = 'YOUR_TOKEN'

def load_data():
    """Load preprocessed data"""
    X_train = pd.read_csv('./dataset_preprocessing/X_train.csv')
    X_test = pd.read_csv('./dataset_preprocessing/X_test.csv')
    y_train = pd.read_csv('./dataset_preprocessing/y_train.csv').values.ravel()
    y_test = pd.read_csv('./dataset_preprocessing/y_test.csv').values.ravel()
    
    return X_train, X_test, y_train, y_test

def create_confusion_matrix_plot(y_true, y_pred):
    """Create confusion matrix plot"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.close()
    return 'confusion_matrix.png'

def train_advanced_model():
    """Train model with advanced logging to DagsHub"""
    
    # Set experiment
    mlflow.set_experiment("Advanced_ML_Experiment")
    
    with mlflow.start_run():
        # Load data
        X_train, X_test, y_train, y_test = load_data()
        
        # Hyperparameter tuning
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        # Grid search
        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        
        # Best model
        best_model = grid_search.best_estimator_
        
        # Predictions
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)
        
        # Calculate metrics (autolog + 2 additional)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Additional metrics
        try:
            auc_score = roc_auc_score(y_test, y_pred_proba[:, 1])
        except:
            auc_score = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
        
        feature_importance_std = np.std(best_model.feature_importances_)
        
        # Manual logging
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("auc_score", auc_score)  # Additional 1
        mlflow.log_metric("feature_importance_std", feature_importance_std)  # Additional 2
        
        # Log model
        mlflow.sklearn.log_model(best_model, "model")
        
        # Log confusion matrix
        cm_path = create_confusion_matrix_plot(y_test, y_pred)
        mlflow.log_artifact(cm_path)
        
        # Log dataset info
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("test_samples", len(X_test))
        mlflow.log_param("n_features", X_train.shape[1])
        
        print(f"Model logged to DagsHub!")
        print(f"Accuracy: {accuracy}")
        print(f"AUC Score: {auc_score}")

if __name__ == "__main__":
    train_advanced_model()
```

**DagsHub.txt**
```
https://dagshub.com/YOUR_USERNAME/YOUR_REPO
```

---

## 🔄 KRITERIA 3: Membuat Workflow CI

### Struktur Folder

```
Workflow-CI/
├── .github/workflows/
├── MLProject/
│   ├── modelling.py
│   ├── conda.yaml
│   ├── MLProject
│   ├── dataset_preprocessing/
│   └── Dockerfile              # untuk advanced
```

### Langkah 1: Setup MLflow Project (Basic - 2 pts)

**MLProject/MLProject**
```yaml
name: ML_Training_Project

conda_env: conda.yaml

entry_points:
  main:
    command: "python modelling.py"
```

**MLProject/conda.yaml**
```yaml
name: ml_env
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.8
  - pip
  - pip:
    - mlflow
    - scikit-learn
    - pandas
    - numpy
    - matplotlib
    - seaborn
```

**MLProject/modelling.py**
```python
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train_model():
    mlflow.set_experiment("CI_ML_Experiment")
    
    with mlflow.start_run():
        # Load data
        X_train = pd.read_csv('./dataset_preprocessing/X_train.csv')
        X_test = pd.read_csv('./dataset_preprocessing/X_test.csv')
        y_train = pd.read_csv('./dataset_preprocessing/y_train.csv').values.ravel()
        y_test = pd.read_csv('./dataset_preprocessing/y_test.csv').values.ravel()
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Log
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(model, "model")
        
        print(f"Model trained with accuracy: {accuracy}")

if __name__ == "__main__":
    train_model()
```

**GitHub Workflow: .github/workflows/ci.yml**
```yaml
name: ML Training CI

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]
  workflow_dispatch:

jobs:
  train:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v3
      with:
        python-version: '3.8'
    
    - name: Install MLflow
      run: |
        pip install mlflow
    
    - name: Run MLflow Project
      run: |
        mlflow run ./MLProject --no-conda
```

### Langkah 2: Simpan Artifacts (Skilled - 3 pts)

**Update workflow untuk menyimpan artifacts:**

```yaml
name: ML Training CI with Artifacts

on:
  push:
    branches: [ main ]
  workflow_dispatch:

jobs:
  train:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v3
      with:
        python-version: '3.8'
    
    - name: Install dependencies
      run: |
        pip install mlflow scikit-learn pandas numpy
    
    - name: Run MLflow Project
      run: |
        cd MLProject
        python modelling.py
    
    - name: Upload MLflow artifacts
      uses: actions/upload-artifact@v3
      with:
        name: mlflow-artifacts
        path: mlruns/
    
    - name: Commit and push artifacts
      run: |
        git config --local user.email "action@github.com"
        git config --local user.name "GitHub Action"
        git add mlruns/
        git commit -m "Add MLflow artifacts" || exit 0
        git push
```

### Langkah 3: Docker Integration (Advanced - 4 pts)

**MLProject/Dockerfile**
```dockerfile
FROM python:3.8-slim

WORKDIR /app

COPY conda.yaml .
COPY modelling.py .
COPY dataset_preprocessing/ ./dataset_preprocessing/

RUN pip install mlflow scikit-learn pandas numpy matplotlib seaborn

CMD ["python", "modelling.py"]
```

**Update modelling.py untuk Docker**
```python
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os

def train_model():
    # Set tracking URI if available
    if os.getenv('MLFLOW_TRACKING_URI'):
        mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))
    
    mlflow.set_experiment("Docker_ML_Experiment")
    
    with mlflow.start_run():
        # Load data
        X_train = pd.read_csv('./dataset_preprocessing/X_train.csv')
        X_test = pd.read_csv('./dataset_preprocessing/X_test.csv')
        y_train = pd.read_csv('./dataset_preprocessing/y_train.csv').values.ravel()
        y_test = pd.read_csv('./dataset_preprocessing/y_test.csv').values.ravel()
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Log
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(model, "model")
        
        print(f"Model trained with accuracy: {accuracy}")

if __name__ == "__main__":
    train_model()
```

**Advanced CI Workflow dengan Docker:**
```yaml
name: Advanced ML CI with Docker

on:
  push:
    branches: [ main ]
  workflow_dispatch:

jobs:
  train-and-deploy:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v3
      with:
        python-version: '3.8'
    
    - name: Install MLflow
      run: |
        pip install mlflow
    
    - name: Build Docker image with MLflow
      run: |
        cd MLProject
        mlflow models build-docker -m "models:/your-model/latest" -n "ml-model-image"
    
    - name: Login to Docker Hub
      uses: docker/login-action@v2
      with:
        username: ${{ secrets.DOCKER_USERNAME }}
        password: ${{ secrets.DOCKER_PASSWORD }}
    
    - name: Push to Docker Hub
      run: |
        docker tag ml-model-image ${{ secrets.DOCKER_USERNAME }}/ml-model:latest
        docker push ${{ secrets.DOCKER_USERNAME }}/ml-model:latest
    
    - name: Upload artifacts
      uses: actions/upload-artifact@v3
      with:
        name: docker-artifacts
        path: |
          mlruns/
          MLProject/Dockerfile
```

---

## 📊 KRITERIA 4: Sistem Monitoring dan Logging

### Struktur Folder

```
Monitoring dan Logging/
├── 1.bukti_serving/
├── 2.prometheus.yml
├── 3.prometheus_exporter.py
├── 4.bukti monitoring Prometheus/
├── 5.bukti monitoring Grafana/
├── 6.bukti alerting Grafana/
├── 7.inference.py
```

### Langkah 1: Model Serving (Basic - 2 pts)

**7.inference.py**
```python
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
import pickle
import os

app = Flask(__name__)

# Load model
model_path = "mlruns/experiment_id/run_id/artifacts/model"  # sesuaikan path
model = mlflow.sklearn.load_model(model_path)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        data = request.json
        
        # Convert to DataFrame
        df = pd.DataFrame([data])
        
        # Make prediction
        prediction = model.predict(df)
        probability = model.predict
# Make prediction
        prediction = model.predict(df)
        probability = model.predict_proba(df)
        
        return jsonify({
            'prediction': int(prediction[0]),
            'probability': probability[0].tolist()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    print("Starting ML Model Server...")
    app.run(host='0.0.0.0', port=5000, debug=True)
```

**Jalankan Model Serving:**
```bash
# Jalankan server
python 7.inference.py

# Test di terminal lain
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"feature1": 1.0, "feature2": 2.0, "feature3": 3.0}'
```

### Langkah 2: Setup Prometheus (Skilled - 3 pts)

**2.prometheus.yml**
```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  # - "first_rules.yml"
  # - "second_rules.yml"

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'ml-model'
    static_configs:
      - targets: ['localhost:8000']
    scrape_interval: 5s
    metrics_path: /metrics

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['localhost:9100']
```

**3.prometheus_exporter.py**
```python
import time
import random
import threading
from prometheus_client import start_http_server, Gauge, Counter, Histogram, Info
import psutil
import requests
import json

# Define metrics
model_predictions_total = Counter('ml_model_predictions_total', 'Total number of predictions made')
model_prediction_duration_seconds = Histogram('ml_model_prediction_duration_seconds', 'Time spent on prediction')
model_accuracy_gauge = Gauge('ml_model_accuracy', 'Current model accuracy')
model_cpu_usage = Gauge('ml_model_cpu_usage_percent', 'CPU usage percentage')
model_memory_usage = Gauge('ml_model_memory_usage_bytes', 'Memory usage in bytes')
model_disk_usage = Gauge('ml_model_disk_usage_percent', 'Disk usage percentage')
model_network_bytes_sent = Gauge('ml_model_network_bytes_sent_total', 'Network bytes sent')
model_network_bytes_recv = Gauge('ml_model_network_bytes_recv_total', 'Network bytes received')
model_uptime_seconds = Gauge('ml_model_uptime_seconds', 'Model server uptime in seconds')
model_response_time_seconds = Histogram('ml_model_response_time_seconds', 'Response time for model predictions')
model_error_rate = Gauge('ml_model_error_rate', 'Error rate of predictions')

# Model info
model_info = Info('ml_model_info', 'Information about the ML model')
model_info.info({'version': '1.0', 'algorithm': 'RandomForest', 'framework': 'scikit-learn'})

class MLMetricsCollector:
    def __init__(self):
        self.start_time = time.time()
        self.error_count = 0
        self.total_requests = 0
        
    def collect_system_metrics(self):
        """Collect system metrics"""
        while True:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                model_cpu_usage.set(cpu_percent)
                
                # Memory usage
                memory = psutil.virtual_memory()
                model_memory_usage.set(memory.used)
                
                # Disk usage
                disk = psutil.disk_usage('/')
                model_disk_usage.set(disk.percent)
                
                # Network usage
                net_io = psutil.net_io_counters()
                model_network_bytes_sent.set(net_io.bytes_sent)
                model_network_bytes_recv.set(net_io.bytes_recv)
                
                # Uptime
                uptime = time.time() - self.start_time
                model_uptime_seconds.set(uptime)
                
                # Simulate model accuracy (in real scenario, calculate from validation set)
                accuracy = 0.85 + random.uniform(-0.05, 0.05)
                model_accuracy_gauge.set(accuracy)
                
                # Error rate
                if self.total_requests > 0:
                    error_rate = self.error_count / self.total_requests
                    model_error_rate.set(error_rate)
                
                time.sleep(10)  # Collect every 10 seconds
                
            except Exception as e:
                print(f"Error collecting metrics: {e}")
                time.sleep(10)
    
    def simulate_model_requests(self):
        """Simulate model prediction requests for demo"""
        while True:
            try:
                # Simulate prediction request
                start_time = time.time()
                
                # Simulate prediction processing time
                processing_time = random.uniform(0.1, 2.0)
                time.sleep(processing_time)
                
                # Record metrics
                model_predictions_total.inc()
                model_prediction_duration_seconds.observe(processing_time)
                model_response_time_seconds.observe(processing_time)
                
                self.total_requests += 1
                
                # Simulate occasional errors
                if random.random() < 0.05:  # 5% error rate
                    self.error_count += 1
                
                # Wait before next request
                time.sleep(random.uniform(1, 5))
                
            except Exception as e:
                print(f"Error in simulation: {e}")
                self.error_count += 1
                time.sleep(5)

def main():
    # Start metrics server
    start_http_server(8000)
    print("Prometheus metrics server started on port 8000")
    
    # Initialize collector
    collector = MLMetricsCollector()
    
    # Start background threads
    system_thread = threading.Thread(target=collector.collect_system_metrics)
    system_thread.daemon = True
    system_thread.start()
    
    simulation_thread = threading.Thread(target=collector.simulate_model_requests)
    simulation_thread.daemon = True
    simulation_thread.start()
    
    print("Metrics collection started...")
    print("Visit http://localhost:8000/metrics to see metrics")
    
    # Keep the main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down metrics collector...")

if __name__ == '__main__':
    main()
```

**Setup dan Jalankan Prometheus:**
```bash
# Download Prometheus
wget https://github.com/prometheus/prometheus/releases/download/v2.40.0/prometheus-2.40.0.linux-amd64.tar.gz
tar xvfz prometheus-2.40.0.linux-amd64.tar.gz
cd prometheus-2.40.0.linux-amd64

# Copy config
cp ../2.prometheus.yml ./prometheus.yml

# Jalankan Prometheus
./prometheus --config.file=prometheus.yml --storage.tsdb.path=./data --web.console.templates=consoles --web.console.libraries=console_libraries

# Jalankan metrics exporter di terminal lain
python 3.prometheus_exporter.py
```

### Langkah 3: Setup Grafana (Advanced - 4 pts)

**Install Grafana:**
```bash
# Ubuntu/Debian
sudo apt-get install -y software-properties-common
sudo add-apt-repository "deb https://packages.grafana.com/oss/deb stable main"
wget -q -O - https://packages.grafana.com/gpg.key | sudo apt-key add -
sudo apt-get update
sudo apt-get install grafana

# Start Grafana
sudo systemctl start grafana-server
sudo systemctl enable grafana-server

# Access: http://localhost:3000
# Default login: admin/admin
```

**Grafana Dashboard JSON (untuk import):**

Buat file `grafana_dashboard.json`:
```json
{
  "dashboard": {
    "id": null,
    "title": "ML Model Monitoring - [Username-Dicoding]",
    "tags": ["machine-learning", "monitoring"],
    "timezone": "browser",
    "panels": [
      {
        "id": 1,
        "title": "Total Predictions",
        "type": "stat",
        "targets": [
          {
            "expr": "ml_model_predictions_total",
            "refId": "A"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "color": {
              "mode": "thresholds"
            },
            "thresholds": {
              "steps": [
                {"color": "green", "value": null},
                {"color": "yellow", "value": 1000},
                {"color": "red", "value": 5000}
              ]
            }
          }
        },
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0}
      },
      {
        "id": 2,
        "title": "Model Accuracy",
        "type": "gauge",
        "targets": [
          {
            "expr": "ml_model_accuracy",
            "refId": "A"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "min": 0,
            "max": 1,
            "thresholds": {
              "steps": [
                {"color": "red", "value": null},
                {"color": "yellow", "value": 0.7},
                {"color": "green", "value": 0.85}
              ]
            }
          }
        },
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0}
      },
      {
        "id": 3,
        "title": "CPU Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "ml_model_cpu_usage_percent",
            "refId": "A"
          }
        ],
        "gridPos": {"h": 8, "w": 8, "x": 0, "y": 8}
      },
      {
        "id": 4,
        "title": "Memory Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "ml_model_memory_usage_bytes / 1024 / 1024 / 1024",
            "refId": "A"
          }
        ],
        "gridPos": {"h": 8, "w": 8, "x": 8, "y": 8}
      },
      {
        "id": 5,
        "title": "Disk Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "ml_model_disk_usage_percent",
            "refId": "A"
          }
        ],
        "gridPos": {"h": 8, "w": 8, "x": 16, "y": 8}
      },
      {
        "id": 6,
        "title": "Prediction Duration",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(ml_model_prediction_duration_seconds_sum[5m]) / rate(ml_model_prediction_duration_seconds_count[5m])",
            "refId": "A"
          }
        ],
        "gridPos": {"h": 8, "w": 8, "x": 0, "y": 16}
      },
      {
        "id": 7,
        "title": "Error Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "ml_model_error_rate",
            "refId": "A"
          }
        ],
        "gridPos": {"h": 8, "w": 8, "x": 8, "y": 16}
      },
      {
        "id": 8,
        "title": "Model Uptime",
        "type": "graph",
        "targets": [
          {
            "expr": "ml_model_uptime_seconds / 3600",
            "refId": "A"
          }
        ],
        "gridPos": {"h": 8, "w": 8, "x": 16, "y": 16}
      },
      {
        "id": 9,
        "title": "Network Traffic (Sent)",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(ml_model_network_bytes_sent_total[5m])",
            "refId": "A"
          }
        ],
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 24}
      },
      {
        "id": 10,
        "title": "Network Traffic (Received)",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(ml_model_network_bytes_recv_total[5m])",
            "refId": "A"
          }
        ],
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 24}
      }
    ],
    "time": {
      "from": "now-1h",
      "to": "now"
    },
    "refresh": "5s"
  }
}
```

### Langkah 4: Setup Alerting (Advanced - 4 pts)

**Grafana Alerting Rules:**

1. **CPU Usage Alert:**
```json
{
  "alert": {
    "name": "High CPU Usage",
    "message": "CPU usage is above 80%",
    "frequency": "10s",
    "conditions": [
      {
        "query": {
          "queryType": "",
          "refId": "A",
          "model": {
            "expr": "ml_model_cpu_usage_percent",
            "interval": "",
            "refId": "A"
          }
        },
        "reducer": {
          "type": "last",
          "params": []
        },
        "evaluator": {
          "params": [80],
          "type": "gt"
        }
      }
    ]
  }
}
```

2. **Model Accuracy Alert:**
```json
{
  "alert": {
    "name": "Low Model Accuracy",
    "message": "Model accuracy dropped below 80%",
    "frequency": "30s",
    "conditions": [
      {
        "query": {
          "queryType": "",
          "refId": "A",
          "model": {
            "expr": "ml_model_accuracy",
            "interval": "",
            "refId": "A"
          }
        },
        "reducer": {
          "type": "last",
          "params": []
        },
        "evaluator": {
          "params": [0.8],
          "type": "lt"
        }
      }
    ]
  }
}
```

3. **High Error Rate Alert:**
```json
{
  "alert": {
    "name": "High Error Rate",
    "message": "Model error rate is above 10%",
    "frequency": "30s",
    "conditions": [
      {
        "query": {
          "queryType": "",
          "refId": "A",
          "model": {
            "expr": "ml_model_error_rate",
            "interval": "",
            "refId": "A"
          }
        },
        "reducer": {
          "type": "last",
          "params": []
        },
        "evaluator": {
          "params": [0.1],
          "type": "gt"
        }
      }
    ]
  }
}
```

### Langkah 5: Dokumentasi dan Screenshot

**Struktur folder untuk bukti:**

```bash
mkdir -p "4.bukti monitoring Prometheus"
mkdir -p "5.bukti monitoring Grafana"  
mkdir -p "6.bukti alerting Grafana"
```

**Screenshot yang diperlukan:**

1. **1.bukti_serving/**
   - Screenshot model serving berjalan (http://localhost:5000/health)
   - Screenshot response dari prediction endpoint

2. **4.bukti monitoring Prometheus/**
   - 1.monitoring_predictions_total.png
   - 2.monitoring_cpu_usage.png
   - 3.monitoring_memory_usage.png
   - 4.monitoring_accuracy.png
   - 5.monitoring_response_time.png

3. **5.bukti monitoring Grafana/**
   - 1.monitoring_dashboard_overview.png
   - 2.monitoring_cpu_memory.png
   - 3.monitoring_model_metrics.png
   - 4.monitoring_network_traffic.png
   - 5.monitoring_error_rate.png
   - (untuk advanced: 10 screenshot berbeda)

4. **6.bukti alerting Grafana/**
   - 1.rules_cpu_alert.png
   - 2.notifikasi_cpu_alert.png  
   - 3.rules_accuracy_alert.png
   - 4.notifikasi_accuracy_alert.png
   - 5.rules_error_rate_alert.png
   - 6.notifikasi_error_rate_alert.png

---

## 📝 CHECKLIST SUBMISSION

### Kriteria 1: Eksperimen Dataset
- [ ] Repository GitHub dibuat dengan struktur yang benar
- [ ] Notebook eksperimen lengkap (EDA, preprocessing)
- [ ] File otomatisasi (untuk Skilled/Advanced)
- [ ] GitHub Actions workflow (untuk Advanced)

### Kriteria 2: Model ML
- [ ] MLflow tracking setup
- [ ] Model training dengan logging
- [ ] Screenshot dashboard dan artifacts
- [ ] Hyperparameter tuning (untuk Skilled)
- [ ] DagsHub integration (untuk Advanced)

### Kriteria 3: Workflow CI
- [ ] MLflow Project struktur lengkap
- [ ] GitHub Actions workflow
- [ ] Artifacts tersimpan (untuk Skilled)
- [ ] Docker integration (untuk Advanced)

### Kriteria 4: Monitoring
- [ ] Model serving berjalan
- [ ] Prometheus setup dengan minimal 3 metrics (Basic)
- [ ] Grafana dashboard dengan minimal 5 metrics (Skilled)
- [ ] Grafana alerting dengan 3 rules (Advanced)
- [ ] Screenshot lengkap sesuai level

---

## 🚀 TIPS SUKSES

1. **Mulai dari Basic**: Pastikan level Basic berjalan sempurna sebelum naik ke Skilled/Advanced
2. **Test Secara Berkala**: Jalankan setiap komponen secara terpisah untuk memastikan tidak ada error
3. **Dokumentasi Screenshot**: Pastikan username Dicoding terlihat di semua screenshot Grafana
4. **Repository Terstruktur**: Ikuti struktur folder yang diminta dengan tepat
5. **Dependencies**: Catat semua package yang dibutuhkan di requirements.txt

**Selamat mengerjakan! 🎯**
