"""Three-feature classifiers with nested subject-based GridSearchCV."""
import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from model_config import TASKS, SEED, build_models


def load_rows(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        required = {'filename', 'subject_id', 'status', 'eye_state', 'mouth_state', 'ear_l', 'ear_r', 'mar'}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f'Missing columns: {sorted(missing)}')
        rows = list(reader)
    names = [r['filename'] for r in rows]
    if any(not name.strip() for name in names) or len(set(names)) != len(names):
        raise ValueError('Empty or duplicate filenames in features CSV')
    return rows


def select_rows(rows, task):
    selected, excluded = [], []
    for row in rows:
        reason = None
        if row['status'] != 'OK':
            reason = 'status_not_OK'
        elif not row['subject_id'].strip() or row['subject_id'].strip().upper() in {'UNKNOWN', 'NONE'}:
            reason = 'missing_subject'
        elif row[task['target']] not in task['labels']:
            reason = 'unknown_or_invalid_label'
        else:
            try:
                values = [float(row[k]) for k in task['features']]
                if any(not math.isfinite(v) or v < 0 for v in values):
                    reason = 'invalid_feature'
            except (ValueError, TypeError):
                reason = 'invalid_feature'
        if reason:
            excluded.append({'filename': row['filename'], 'reason': reason})
        else:
            selected.append(row)
    return selected, excluded


def dump_csv(path, rows, columns):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def validate_subjects(rows, task):
    subjects = sorted({r['subject_id'] for r in rows})
    if len(subjects) < 3:
        raise ValueError(f"{task['target']}: need at least 3 subjects, found {subjects}")
    for subject in subjects:
        observed = {r[task['target']] for r in rows if r['subject_id'] == subject}
        if observed != set(task['labels']):
            raise ValueError(
                f"{task['target']}: subject {subject} lacks a class after filtering; "
                'each inner/outer validation subject must have both classes'
            )
    return subjects


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features', type=Path, default=Path('data/features/features.csv'))
    parser.add_argument('--output-dir', type=Path, default=Path('reports/paper_all_features'))
    parser.add_argument('--jobs', type=int, default=1, help='Grid search workers; -1 uses all CPUs')
    args = parser.parse_args()
    try:
        import numpy as np
        import sklearn
        from sklearn.base import clone
        from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut, ParameterGrid
        from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score
    except ImportError:
        parser.error('Install training dependencies: python -m pip install -r requirements-train.txt')

    models, param_grids = build_models()
    rows = load_rows(args.features)
    # Validate both tasks before producing any results.
    datasets = {}
    for name, task in TASKS.items():
        selected, excluded = select_rows(rows, task)
        subjects = validate_subjects(selected, task)
        datasets[name] = selected, excluded, subjects

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics, predictions, details = [], [], {}
    positive_metrics = ['precision', 'recall', 'f1-score']
    for task_name, task in TASKS.items():
        selected, excluded, subjects = datasets[task_name]
        print(f"{task_name}: features={task['features']}, samples={len(selected)}", flush=True)
        details[task_name] = {'features': task['features'], 'labels': task['labels'],
            'positive_label': task['labels'][1], 'included': len(selected),
            'excluded_reasons': dict(Counter(r['reason'] for r in excluded)), 'models': {}}
        dump_csv(args.output_dir / f'{task_name}_excluded.csv', excluded, ['filename', 'reason'])
        X = np.array([[float(r[k]) for k in task['features']] for r in selected])
        y = np.array([r[task['target']] for r in selected])
        groups = np.array([r['subject_id'] for r in selected])
        for model_name, prototype in models.items():
            fold_details = []
            pooled = []
            for subject in subjects:
                train, test = groups != subject, groups == subject
                inner_cv = list(LeaveOneGroupOut().split(
                    X[train], y[train], groups=groups[train]))
                print(f'{task_name} / {model_name}: outer test={subject}, '
                      f'train={train.sum()}, test={test.sum()}, '
                      f'candidates={len(ParameterGrid(param_grids[model_name]))}, '
                      f'inner folds={len(inner_cv)}', flush=True)
                search = GridSearchCV(
                    clone(prototype), param_grids[model_name], scoring='f1_macro',
                    cv=inner_cv, refit=True, n_jobs=args.jobs,
                    error_score='raise', verbose=1,
                )
                search.fit(X[train], y[train])
                estimator = search.best_estimator_
                print(f'Best parameters: {search.best_params_}; '
                      f'inner macro F1={search.best_score_:.4f}', flush=True)
                candidates = [
                    {'params': json.dumps(params, sort_keys=True),
                     'mean_validation_macro_f1': float(search.cv_results_['mean_test_score'][i]),
                     'std_validation_macro_f1': float(search.cv_results_['std_test_score'][i]),
                     'rank': int(search.cv_results_['rank_test_score'][i])}
                    for i, params in enumerate(search.cv_results_['params'])
                ]
                dump_csv(args.output_dir / f'{task_name}_{model_name}_fold{subjects.index(subject) + 1}_search.csv',
                         candidates, list(candidates[0]))
                predicted = estimator.predict(X[test])
                report = classification_report(y[test], predicted, labels=task['labels'], output_dict=True, zero_division=0)
                cm = confusion_matrix(y[test], predicted, labels=task['labels']).tolist()
                metric = {'task': task_name, 'model': model_name, 'test_subject': subject,
                    'n_train': int(train.sum()), 'n_test': int(test.sum()),
                    'best_params': json.dumps(search.best_params_, sort_keys=True),
                    'inner_best_macro_f1': float(search.best_score_),
                    'accuracy': accuracy_score(y[test], predicted),
                    'balanced_accuracy': balanced_accuracy_score(y[test], predicted),
                    'macro_f1': f1_score(y[test], predicted, labels=task['labels'], average='macro', zero_division=0)}
                for key in positive_metrics:
                    metric['positive_' + key.replace('-', '_')] = report[task['labels'][1]][key]
                metrics.append(metric)
                fold_details.append({'test_subject': subject, 'train_subjects': [s for s in subjects if s != subject],
                    'best_params': search.best_params_, 'inner_best_macro_f1': float(search.best_score_),
                    'inner_validation_subjects': [sorted(set(groups[train][valid])) for _, valid in inner_cv],
                    'train_distribution': dict(Counter(y[train])), 'test_distribution': dict(Counter(y[test])),
                    'classification_report': report, 'confusion_matrix': cm})
                for index, prediction in zip(np.flatnonzero(test), predicted):
                    row = selected[index]
                    item = {'task': task_name, 'model': model_name, 'subject_id': subject,
                        'filename': row['filename'], 'true_label': row[task['target']],
                        'predicted_label': str(prediction), 'correct': row[task['target']] == prediction}
                    predictions.append(item)
                    pooled.append(item)
            actual = [r['true_label'] for r in pooled]
            predicted = [r['predicted_label'] for r in pooled]
            details[task_name]['models'][model_name] = {'folds': fold_details,
                'pooled_classification_report': classification_report(actual, predicted, labels=task['labels'], output_dict=True, zero_division=0),
                'pooled_confusion_matrix': confusion_matrix(actual, predicted, labels=task['labels']).tolist()}

    summary = []
    for task_name in TASKS:
        for model_name in models:
            folds = [r for r in metrics if r['task'] == task_name and r['model'] == model_name]
            item = {'task': task_name, 'model': model_name, 'n_folds': len(folds)}
            for key in ['accuracy', 'balanced_accuracy', 'macro_f1', 'positive_precision', 'positive_recall', 'positive_f1_score']:
                values = [r[key] for r in folds]
                item[key + '_mean'] = float(np.mean(values))
                item[key + '_std'] = float(np.std(values))
            summary.append(item)
    dump_csv(args.output_dir / 'fold_metrics.csv', metrics, list(metrics[0]))
    dump_csv(args.output_dir / 'summary.csv', summary, list(summary[0]))
    columns = ['task', 'model', 'subject_id', 'filename', 'true_label', 'predicted_label', 'correct']
    dump_csv(args.output_dir / 'predictions.csv', predictions, columns)
    dump_csv(args.output_dir / 'errors.csv', [r for r in predictions if not r['correct']], columns)
    metadata = {'source': str(args.features.resolve()), 'sklearn_version': sklearn.__version__, 'seed': SEED,
        'protocol': 'Nested leave-one-subject-out; inner subject-based GridSearchCV; scoring=f1_macro.',
        'paper_adaptation': 'All classifiers use ear_l, ear_r, mar in that order. Two independent tasks on this dataset, not the paper NORMAL/YAWN/SLEEP multiclass task. Proposed search grids include Table II settings; original full grids were not specified.',
        'confusion_matrix_axes': 'Rows=true, columns=predicted; order given by each task labels.',
        'search_grids': param_grids,
        'parameters': {name: str(model) for name, model in models.items()}, 'tasks': details}
    (args.output_dir / 'evaluation.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding='utf-8')
    lines = ['# Subject-independent baseline', '',
        'All SVM, DT and RF classifiers use three inputs in order: ear_l, ear_r, mar, for both eyes and mouth tasks.',
        'Nested subject-based GridSearchCV; outer test subjects never participate in parameter selection.',
        'SVM scaling is fitted inside each inner training fold and refitted on outer train only.',
        'Paper-inspired adaptation: independent eyes/mouth tasks, not NORMAL/YAWN/SLEEP reproduction.',
        'Proposed grids include Table II configurations; the paper does not specify its complete grids.',
        'No final deployment model is trained or saved.', '',
        '| Task | Model | Macro F1 mean ± std | Balanced accuracy mean | Positive recall mean |',
        '|---|---|---:|---:|---:|']
    for row in summary:
        lines.append(f"| {row['task']} | {row['model']} | {row['macro_f1_mean']:.3f} ± {row['macro_f1_std']:.3f} | {row['balanced_accuracy_mean']:.3f} | {row['positive_recall_mean']:.3f} |")
    lines.extend(['', 'Positive class: CLOSED for eyes; YAWN for mouth. Standard deviation uses ddof=0.',
        'summary.csv averages subjects equally; pooled reports weight each image equally.',
        'See evaluation.json for per-class metrics and confusion matrices, errors.csv for mistakes.',
        'Only three subjects: results do not establish generalization to a wider population.',
        'Reviewing these results to choose a model makes this a development evaluation; independent data is needed for a final estimate.'])
    (args.output_dir / 'README.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines), flush=True)
    print(f'\nSaved reports: {args.output_dir.resolve()}', flush=True)


if __name__ == '__main__':
    main()
