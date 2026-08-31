import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class DiseaseModelTrainer:
    def __init__(self, disease_name):
        self.disease_name = disease_name
        self.model = None
        self.scaler = None
        self.imputer = None
        self.label_encoder = None
        self.feature_names = None
        self.class_names = None
        
    def load_and_preprocess(self, filepath, target_col):
        # Load CSV and preprocess data
        print(f"\n{'='*60}")
        print(f"Loading {self.disease_name} dataset...")
        print(f"{'='*60}")
        
        df = pd.read_csv(filepath)
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        print(f"\nMissing values before cleaning:")
        print(df.isnull().sum()[df.isnull().sum() > 0])
        
        # Fill numeric columns with median, categorical with mode
        for col in df.columns:
            if df[col].dtype in ['float64', 'int64']:
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode()[0])
        
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        if y.dtype == 'object':
            self.label_encoder = LabelEncoder()
            y = self.label_encoder.fit_transform(y)
            self.class_names = self.label_encoder.classes_
        else:
            self.class_names = np.unique(y)
        
        self.feature_names = X.columns.tolist()
        
        for col in X.columns:
            if X[col].dtype == 'object':
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
        
        print(f"\nTarget distribution:")
        print(pd.Series(y).value_counts())
        print(f"Class names: {self.class_names}")
        
        return X, y
    
    def handle_imbalance(self, X_train, y_train, strategy='smote'):
        # Class imbalance
        print(f"\nHandling class imbalance using {strategy.upper()}...")
        
        if strategy == 'smote':
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        elif strategy == 'undersample':
            rus = RandomUnderSampler(random_state=42)
            X_resampled, y_resampled = rus.fit_resample(X_train, y_train)
        elif strategy == 'combined':
            over = SMOTE(sampling_strategy=0.5, random_state=42)
            under = RandomUnderSampler(sampling_strategy=0.8, random_state=42)
            X_resampled, y_resampled = over.fit_resample(X_train, y_train)
            X_resampled, y_resampled = under.fit_resample(X_resampled, y_resampled)
        else:
            return X_train, y_train
        
        print(f"Before resampling: {pd.Series(y_train).value_counts().to_dict()}")
        print(f"After resampling: {pd.Series(y_resampled).value_counts().to_dict()}")
        
        return X_resampled, y_resampled
    
    def train_model(self, X, y, balance_strategy='smote', test_size=0.2):
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"\nTrain set size: {X_train.shape[0]}")
        print(f"Test set size: {X_test.shape[0]}")
        
        # Handle class imbalance
        X_train_balanced, y_train_balanced = self.handle_imbalance(
            X_train, y_train, strategy=balance_strategy
        )
        # Impute missing values (fit on training data)
        self.imputer = SimpleImputer(strategy='median')
        X_train_imputed = self.imputer.fit_transform(X_train_balanced)
        X_test_imputed = self.imputer.transform(X_test)

        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train_imputed)
        X_test_scaled = self.scaler.transform(X_test_imputed)
        
        print(f"\nTraining Random Forest model...")
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=4,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train_scaled, y_train_balanced)
        
        cv_scores = cross_val_score(
            self.model, X_train_scaled, y_train_balanced, cv=5, scoring='accuracy'
        )
        print(f"Cross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)
        
        print(f"\n{'='*60}")
        print(f"MODEL EVALUATION - {self.disease_name}")
        print(f"{'='*60}")
        
        print(f"\nTest Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        
        # ROC-AUC for binary classification
        if len(np.unique(y)) == 2:
            roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
            print(f"ROC-AUC Score: {roc_auc:.4f}")
        
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=[str(c) for c in self.class_names]))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\nConfusion Matrix:")
        print(cm)
        
        self.plot_feature_importance(top_n=15)
        
        return X_test_scaled, y_test, y_pred
    
    def plot_feature_importance(self, top_n=15):
        if self.model and self.feature_names:
            importances = self.model.feature_importances_
            indices = np.argsort(importances)[-top_n:]
            
            plt.figure(figsize=(10, 8))
            plt.title(f'Top {top_n} Feature Importances - {self.disease_name}')
            plt.barh(range(len(indices)), importances[indices])
            plt.yticks(range(len(indices)), [self.feature_names[i] for i in indices])
            plt.xlabel('Relative Importance')
            plt.tight_layout()
            plt.savefig(f'{self.disease_name.lower().replace(" ", "_")}_feature_importance.png')
            print(f"\nFeature importance plot saved!")
    
    def save_model(self, model_dir='models'):
        # Save model
        import os
        os.makedirs(model_dir, exist_ok=True)
        
        model_name = self.disease_name.lower().replace(' ', '_')
        
        joblib.dump(self.model, f'{model_dir}/{model_name}_model.pkl')
        joblib.dump(self.scaler, f'{model_dir}/{model_name}_scaler.pkl')
        if self.imputer:
            joblib.dump(self.imputer, f'{model_dir}/{model_name}_imputer.pkl')
        
        if self.label_encoder:
            joblib.dump(self.label_encoder, f'{model_dir}/{model_name}_label_encoder.pkl')
        
        # Save feature names and class names
        metadata = {
            'feature_names': self.feature_names,
            'class_names': self.class_names.tolist() if hasattr(self.class_names, 'tolist') else list(self.class_names),
            'disease_name': self.disease_name
        }
        joblib.dump(metadata, f'{model_dir}/{model_name}_metadata.pkl')
        
        print(f"\n✓ Model saved to {model_dir}/{model_name}_model.pkl")
    
    def predict(self, input_data):
        if isinstance(input_data, dict):
            input_data = pd.DataFrame([input_data])
        
        for feature in self.feature_names:
            if feature not in input_data.columns:
                input_data[feature] = 0
        input_data = input_data[self.feature_names]

        for col in input_data.columns:
            input_data[col] = pd.to_numeric(input_data[col], errors='coerce')

        if self.imputer:
            input_imputed = self.imputer.transform(input_data)
        else:
            input_imputed = input_data.values

        input_scaled = self.scaler.transform(input_imputed)
        
        prediction = self.model.predict(input_scaled)[0]
        probabilities = self.model.predict_proba(input_scaled)[0]
        
        if self.label_encoder:
            prediction = self.label_encoder.inverse_transform([prediction])[0]
        
        return {
            'prediction': prediction,
            'probabilities': dict(zip(self.class_names, probabilities)),
            'confidence': max(probabilities)
        }

if __name__ == "__main__":
    diseases = {
        'Diabetes': {
            'filepath': 'datasets/diabetes/diabetes_012_health_indicators_BRFSS2015.csv',
            'target_col': 'Diabetes_012',
            'balance_strategy': 'smote'
        },
        'Hypertension': {
            'filepath': 'datasets/hipertensi.csv',
            'target_col': 'target',
            'balance_strategy': 'smote'
        },
        'Stroke': {
            'filepath': 'datasets/cervical-cancer/kag_risk_factors_cervical_cancer.csv',
            'target_col': 'Biopsy',
            'balance_strategy': 'combined'
        },
        'Jantung Koroner': {
            'filepath': 'datasets/oral-cancer/oral_cancer_prediction_dataset.csv',
            'target_col': 'Oral Cancer (Diagnosis)',
            'balance_strategy': 'smote'
        }
    }

    trained_models = {}
    
    for disease_name, config in diseases.items():
        try:
            print(f"\n\n{'#'*60}")
            print(f"# TRAINING MODEL FOR: {disease_name}")
            print(f"{'#'*60}")
            
            trainer = DiseaseModelTrainer(disease_name)
            
            X, y = trainer.load_and_preprocess(
                config['filepath'], 
                config['target_col']
            )
            
            trainer.train_model(
                X, y, 
                balance_strategy=config['balance_strategy']
            )
            
            trainer.save_model()
            
            trained_models[disease_name] = trainer
            
            print(f"\n{disease_name} model training completed successfully!")
            
        except FileNotFoundError:
            print(f"\nError: Could not find {config['filepath']}")
            print(f"  Please make sure the CSV file exists in the current directory.")
        except Exception as e:
            print(f"\nError training {disease_name} model: {str(e)}")
    
    print(f"\n\n{'='*60}")
    print(f"TRAINING SUMMARY")
    print(f"{'='*60}")
    print(f"Successfully trained models: {len(trained_models)}/{len(diseases)}")
    print(f"Models saved in 'models/' directory")
    
    # Example: Make a prediction
    if 'Diabetes' in trained_models:
        print(f"\n{'='*60}")
        print(f"EXAMPLE PREDICTION - Diabetes")
        print(f"{'='*60}")
        
        sample_input = {
            'HighBP': 1, 'HighChol': 1, 'CholCheck': 1, 'BMI': 28.5,
            'Smoker': 0, 'Stroke': 0, 'HeartDiseaseorAttack': 0,
            'PhysActivity': 1, 'Fruits': 1, 'Veggies': 1,
            'HvyAlcoholConsump': 0, 'AnyHealthcare': 1, 'NoDocbcCost': 0,
            'GenHlth': 3, 'MentHlth': 0, 'PhysHlth': 0, 'DiffWalk': 0,
            'Sex': 1, 'Age': 9, 'Education': 5, 'Income': 6
        }
        
        result = trained_models['Diabetes'].predict(sample_input)
        print(f"\nPrediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Probabilities: {result['probabilities']}")