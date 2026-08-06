from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

class ModelFactory:
    """
    Projede kullanılacak 8 farklı ML modelini ve hiperparametre 
    aralıklarını yöneten fabrika sınıfı.
    """
    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def get_models(self) -> dict:
        """
        Varsayılan (baseline) modelleri bir sözlük halinde döndürür.
        """
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=self.random_state),
            "Decision Tree": DecisionTreeClassifier(random_state=self.random_state),
            "Random Forest": RandomForestClassifier(random_state=self.random_state),
            "Gradient Boosting": GradientBoostingClassifier(random_state=self.random_state),
            "KNN": KNeighborsClassifier(),
            "SVM": SVC(probability=True, random_state=self.random_state),
            "Naive Bayes": GaussianNB(),
            "MLP (Neural Net)": MLPClassifier(max_iter=1000, random_state=self.random_state)
        }
        return models

    def get_param_grids(self) -> dict:
        """
        GridSearchCV / Hyperparameter Tuning için parametre uzaylarını döndürür.
        """
        param_grids = {
            "Logistic Regression": {
                'C': [0.01, 0.1, 1, 10, 100],
                'solver': ['liblinear', 'lbfgs'],
                'max_iter': [1000]
            },
            "Decision Tree": {
                'max_depth': [3, 5, 7, 10, None],
                'min_samples_split': [2, 5, 10],
                'criterion': ['gini', 'entropy']
            },
            "Random Forest": {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7, None],
                'min_samples_split': [2, 5, 10]
            },
            "Gradient Boosting": {
                'n_estimators': [50, 100, 150],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5]
            },
            "KNN": {
                'n_neighbors': [3, 5, 7, 9, 11],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            },
            "SVM": {
                'C': [0.1, 1, 10, 100],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto']
            },
            "Naive Bayes": {
                'var_smoothing': [1e-9, 1e-8, 1e-7]
            },
            "MLP (Neural Net)": {
                'hidden_layer_sizes': [(50,), (100,), (50, 50)],
                'activation': ['relu', 'tanh'],
                'alpha': [0.0001, 0.001, 0.01]
            }
        }
        return param_grids