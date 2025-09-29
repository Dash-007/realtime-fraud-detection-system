"""
Model training pipeline for fraud detection.
Handles training, validation, and model comparison.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    Train and evaluate fraud detection models.
    """
    
    def __init__(self, random_state=42):
        """
        Initialize trainer with random state fro reproducibility.
        """
        self.random_state = random_state
        self.models = {}
        self.scaler = StandardScaler()
        self.results = {}
        
    def prepare_data(self, df: pd.DataFrame, target_col='Class'):
        """
        Prepare data for training.
        
        Args:
            df: DataFrame with features and target
            target_col: Name of target column
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        
        # Seperate features and target
        X = df.drop([target_col, 'Time'], axis=1, errors='ignore')
        y = df[target_col]
        
        # Split data - stratify to maintain fraud ratio
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=self.random_state,
            stratify=y
        )
        
        logger.info(f"Train set: {len(X_train)} samples, {y_train.mean():.2%} fraud.")
        logger.info(f"Test set: {len(X_test)} samples, {y_test.mean():.2%} fraud")
        
        # Scale features (for logistic regression)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Conver back to dataframe to keep feature names
        X_train_scaled = pd.DataFrame(X_train_scaled,
                                      columns=X.columns,
                                      index=X_train.index)
        X_test_scaled = pd.DataFrame(X_test_scaled,
                                     columns=X.columns,
                                     index=X_test.index)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_baseline_models(self, X_train, X_test, y_train, y_test):
        """
        Train multiple baseline models.
        
        Returns:
            Dictionary of trained models
        """
        # Calculate class weight for imbalance
        n_normal = (y_train == 0).sum()
        n_fraud = (y_train == 1).sum()
        class_weight_ratio = n_normal / n_fraud
        
        logger.info(f"\nClass weight ratio: {class_weight_ratio:.1f}:1")
        
        # Mpdel 1: Logitic Regression (simple, interpretable)
        logger.info("\nTraining Logistic Regression...")
        lr = LogisticRegression(
            class_weight='balanced', # Handles imbalance
            random_state=self.random_state,
            max_iter=1000            
        )
        lr.fit(X_train, y_train)
        self.models['Logistic Regression'] = lr
        
        # Model 2: Random Forest (handles non-linearity)
        logger.info("Training Random Forest...")
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight='balanced', # Handle imbalance
            random_state=self.random_state,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)
        self.models['Random Forest'] = rf
        
        # Evaluate each model
        for model_name, model in self.models.items():
            logger.info(f"\nEvaluating {model_name}...")
            self._evaluate_model(model, model_name, X_test, y_test)
            
        return self.models
        
    def _evaluate_model(self, model, model_name, X_test, y_test):
        """
        Evaluate a single model and store results.
        """
        from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Store results
        self.results[model_name] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': cm,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        # Print results
        print(f"\n{'='*50}")
        print(f"{model_name} Results:")
        print(f"\n{'='*50}")
        print(f"Precision: {precision:.3f} (Of predicted frauds, {precision*100:.1f}% were correct)")
        print(f"Recall: {recall:.3f} (Caught {recall*100:.1f}% of actual frauds)")
        print(f"F1 Score: {f1:.3f}")
        print(f"ROC-AUC: {roc_auc:.3f}")
        print(f"\nConfusion Matrix:")
        print(f"            Predicted")
        print(f"            Normal          Fraud")
        print(f"Normal      {cm[0,0]:6d}    {cm[0,1]:6d}")
        print(f"Fraud       {cm[1,0]:6d}    {cm[1,1]:6d}")
        
        # Business metrics
        false_positives = cm[0,1]
        false_negatives = cm[1,0]
        print(f"\nBusiness Impact:")
        print(f"False alarms: {false_positives} legitimate transactions flagged")
        print(f"Missed fraud: {false_negatives} fraudulent transactions missed")