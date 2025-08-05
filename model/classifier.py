from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score
)

# ================== Model Training & Evaluation ==================
def train_and_evaluate_models(X_train, y_train, X_test, y_test, prefix="", classifier_methods=None):
    if classifier_methods is None:
        classifier_methods = ["KNN", "DT", "RF", "SVM", "NB"]
    models = {
        'KNN': KNeighborsClassifier(metric='euclidean', n_neighbors=3, weights='distance'),
        'DT': DecisionTreeClassifier(criterion='entropy', max_depth=10, min_samples_leaf=2, min_samples_split=2,random_state=42),
        'RF': RandomForestClassifier(criterion='entropy', max_depth=None, min_samples_leaf=1, min_samples_split=2, n_estimators=200, random_state=42),
        'SVM': SVC(C=10, kernel='rbf', gamma='scale', random_state=42),
        'NB': GaussianNB(var_smoothing=1e-09)
    }

    print(f"[INFO] Train and Evaluate model with {classifier_methods} methods...")
    results = []

    for classifier_method in classifier_methods:
        model = models.get(classifier_method)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results.append({
            'classifier': classifier_method,
            'model': prefix,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='micro'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1': f1_score(y_test, y_pred, average='weighted')
        })

    print(results)
    return results