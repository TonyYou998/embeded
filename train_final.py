"""Tune on development subjects, refit on all eligible rows, save six models."""
import argparse
import hashlib
import json
import platform
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from model_config import TASKS, SEED, build_models
from train_evaluate import load_rows, select_rows, validate_subjects, dump_csv

MODEL_NAMES = {'svm': 'svm', 'decision_tree': 'dt', 'random_forest': 'rf'}


def train_all(features_path, output_dir, jobs=1):
    import joblib
    import numpy as np
    import sklearn
    from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut, ParameterGrid

    models, grids = build_models()
    rows = load_rows(features_path)
    datasets = {}
    # Fail before writing models if either task is invalid.
    for name, task in TASKS.items():
        selected, excluded = select_rows(rows, task)
        subjects = validate_subjects(selected, task)
        datasets[name] = selected, excluded, subjects
    expected_files = [f'{short}_{task}.joblib' for task in TASKS for short in MODEL_NAMES.values()]
    # Use a new directory for each new training run so old model sets stay intact.
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f'Output directory is not empty: {output_dir}. Choose a new --output-dir.')
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        'schema_version': 1,
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'source': str(features_path.resolve()),
        'source_sha256': hashlib.sha256(features_path.read_bytes()).hexdigest(),
        'versions': {'python': platform.python_version(), 'sklearn': sklearn.__version__,
                     'numpy': np.__version__, 'joblib': joblib.__version__},
        'seed': SEED,
        'protocol': 'Leave-one-subject-out GridSearchCV on development data, then refit on all eligible rows.',
        'score_note': 'CV scores select parameters; they are not independent test estimates. Use the earlier nested evaluation for performance reporting.',
        'search_grids': grids, 'expected_model_files': expected_files, 'tasks': {},
    }
    summary = []
    total = len(TASKS) * len(models)
    done = 0
    for task_name, task in TASKS.items():
        selected, excluded, subjects = datasets[task_name]
        X = np.array([[float(row[k]) for k in task['features']] for row in selected])
        y = np.array([row[task['target']] for row in selected])
        groups = np.array([row['subject_id'] for row in selected])
        splits = list(LeaveOneGroupOut().split(X, y, groups=groups))
        task_info = {
            'target': task['target'], 'features': task['features'], 'labels': task['labels'],
            'samples': len(selected), 'subjects': subjects,
            'label_counts': dict(Counter(row[task['target']] for row in selected)),
            'excluded_counts': dict(Counter(row['reason'] for row in excluded)),
            'validation_subjects': [sorted(set(groups[valid].tolist())) for _, valid in splits],
            'models': {},
        }
        dump_csv(output_dir / f'{task_name}_excluded.csv', excluded, ['filename', 'reason'])
        for model_name, prototype in models.items():
            print(f'[{done + 1}/{total}] {task_name}/{model_name}: {len(selected)} rows, '
                  f'{len(ParameterGrid(grids[model_name]))} candidates, {len(splits)} subject folds', flush=True)
            search = GridSearchCV(prototype, grids[model_name], scoring='f1_macro',
                                  cv=splits, refit=True, n_jobs=jobs, error_score='raise', verbose=1)
            search.fit(X, y)
            estimator = search.best_estimator_
            filename = f'{MODEL_NAMES[model_name]}_{task_name}.joblib'
            # Save the estimator directly; an SVM file includes its fitted scaler.
            joblib.dump(estimator, output_dir / filename, compress=3)
            restored = joblib.load(output_dir / filename)
            if not np.array_equal(estimator.predict(X), restored.predict(X)):
                raise RuntimeError(f'Save/load predictions differ: {filename}')
            if restored.n_features_in_ != len(task['features']):
                raise RuntimeError(f'Incorrect saved feature count: {filename}')
            candidate_rows = [
                {'params': json.dumps(params, sort_keys=True),
                 'mean_selection_macro_f1': float(search.cv_results_['mean_test_score'][i]),
                 'std_selection_macro_f1': float(search.cv_results_['std_test_score'][i]),
                 'rank': int(search.cv_results_['rank_test_score'][i])}
                for i, params in enumerate(search.cv_results_['params'])
            ]
            dump_csv(output_dir / f'{MODEL_NAMES[model_name]}_{task_name}_search.csv', candidate_rows, list(candidate_rows[0]))
            info = {'file': filename, 'best_params': search.best_params_,
                    'selection_cv_macro_f1': float(search.best_score_),
                    'sha256': hashlib.sha256((output_dir / filename).read_bytes()).hexdigest(),
                    'classes': restored.classes_.tolist(), 'roundtrip_verified': True}
            task_info['models'][model_name] = info
            summary.append({'task': task_name, 'model': model_name, 'samples': len(selected),
                            'file': filename, 'selection_cv_macro_f1': info['selection_cv_macro_f1'],
                            'best_params': json.dumps(search.best_params_, sort_keys=True)})
            done += 1
            print(f'Saved and reloaded {filename}: {search.best_params_}', flush=True)
        metadata['tasks'][task_name] = task_info
    dump_csv(output_dir / 'training_summary.csv', summary, list(summary[0]))
    # Written last: its presence indicates all six artifacts were completed.
    (output_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'Completed: {done} models saved in {output_dir.resolve()}', flush=True)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features', type=Path, default=Path('data/features/features.csv'))
    parser.add_argument('--output-dir', type=Path, default=Path('models/classifiers'))
    parser.add_argument('--jobs', type=int, default=1)
    args = parser.parse_args()
    if args.jobs == 0:
        parser.error('--jobs must not be zero')
    try:
        train_all(args.features, args.output_dir, args.jobs)
    except ImportError as exc:
        parser.error(f'Missing dependency: {exc}. Run python -m pip install -r requirements-train.txt')


if __name__ == '__main__':
    main()
