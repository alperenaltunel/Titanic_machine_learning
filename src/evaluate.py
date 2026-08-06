import time
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

class ModelEvaluator:
    """
    Modelleri 5-Fold Cross-Validation ve çoklu metriklerle adil bir şekilde değerlendiren sınıf.
    """
    def __init__(self, n_splits: int = 5, random_state: int = 42):
        self.n_splits = n_splits
        self.random_state = random_state
        self.skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    def evaluate_model(self, model, X: pd.DataFrame, y: pd.Series, model_name: str) -> dict:
        """
        Tek bir modeli 5-Fold Cross Validation ile değerlendirir ve metrikleri hesaplar.
        """
        scoring = {
            'accuracy': 'accuracy',
            'precision': 'precision',
            'recall': 'recall',
            'f1': 'f1',
            'roc_auc': 'roc_auc'
        }

        start_time = time.time()
        cv_results = cross_validate(model, X, y, cv=self.skf, scoring=scoring, return_train_score=True)
        fit_time = time.time() - start_time

        # Metriklerin 5 fold ortalamasını alıyoruz
        results = {
            'Model': model_name,
            'Train Accuracy': np.mean(cv_results['train_accuracy']),
            'Val Accuracy': np.mean(cv_results['test_accuracy']),
            'Precision': np.mean(cv_results['test_precision']),
            'Recall': np.mean(cv_results['test_recall']),
            'F1-Score': np.mean(cv_results['test_f1']),
            'ROC-AUC': np.mean(cv_results['test_roc_auc']),
            'Fit Time (sec)': round(fit_time, 4)
        }

        # Overfitting (Ezber) Farkı Kontrolü
        results['Overfit Gap'] = round(results['Train Accuracy'] - results['Val Accuracy'], 4)

        return results

    def tune_and_evaluate(self, model, param_grid: dict, X: pd.DataFrame, y: pd.Series, model_name: str):
        """
        GridSearchCV ile en iyi hiperparametreleri bulur ve eğiltmiş en iyi modeli ve sonuçları döner.
        """
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=self.skf,
            scoring='f1',
            n_jobs=-1
        )
        
        start_time = time.time()
        grid_search.fit(X, y)
        fit_time = time.time() - start_time

        best_model = grid_search.best_estimator_
        
        # En iyi modelin değerlendirmesi
        eval_metrics = self.evaluate_model(best_model, X, y, model_name=f"{model_name} (Tuned)")
        eval_metrics['Best Params'] = str(grid_search.best_params_)

        return best_model, eval_metrics