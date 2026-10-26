"""Shared estimators and search space for evaluation and deployment training."""
SEED = 42
TASKS = {
    'eyes': {'target': 'eye_state', 'features': ['ear_l', 'ear_r', 'mar'], 'labels': ['OPEN', 'CLOSED']},
    'mouth': {'target': 'mouth_state', 'features': ['ear_l', 'ear_r', 'mar'], 'labels': ['NO_YAWN', 'YAWN']},
}


def build_models():
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    models = {
        'svm': Pipeline([('scale', StandardScaler()), ('model', SVC())]),
        'decision_tree': DecisionTreeClassifier(random_state=SEED),
        'random_forest': RandomForestClassifier(random_state=SEED, n_jobs=1),
    }
    # Proposed grids include Table II settings; full original grids were not given.
    grids = {
        'svm': {
            'model__kernel': ['linear', 'rbf'], 'model__C': [0.1, 1, 10],
            'model__gamma': ['scale'], 'model__class_weight': [None, 'balanced'],
        },
        'decision_tree': {
            'max_depth': [None, 4, 10], 'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 5], 'class_weight': [None, 'balanced'],
        },
        'random_forest': {
            'n_estimators': [50, 100], 'max_depth': [None, 10],
            'min_samples_leaf': [1, 3], 'class_weight': [None, 'balanced'],
        },
    }
    return models, grids
