"""Parse Edge Impulse info.labels without inferring drowsiness from eye state."""
import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

EYES = {'mata_terbuka': 'OPEN', 'mata_terpejam': 'CLOSED'}
MOUTH = {'menguap': 'YAWN', 'tidak_menguap': 'NO_YAWN'}
COLUMNS = ['filename', 'subject_id', 'frame_id', 'eye_state', 'mouth_state']


def identity(name):
    match = re.match(r'^frame_(\d+)_(.+?)\.(?:jpg|jpeg|png)(?:\.|$)', name, re.I)
    return (match[2], int(match[1])) if match else (None, None)


def state(boxes, mapping, expected):
    values = [mapping[b['label']] for b in boxes if b.get('label') in mapping]
    # One missing eye must not silently stand in for two annotated eyes.
    return values[0] if len(values) == expected and len(set(values)) == 1 else 'UNKNOWN'


def parse(info, boxes_index=None):
    if not isinstance(info, dict) or not isinstance(info.get('files'), list):
        raise ValueError('Expected info.labels object with files array')
    rows, issues, manifest = [], [], []
    seen = set()
    for record in info['files']:
        filename = record.get('path')
        if not isinstance(filename, str) or not filename:
            raise ValueError('Record missing path; cannot preserve image identity')
        filename = filename.replace('\\', '/')
        if filename in seen:
            raise ValueError(f'Duplicate image path: {filename}')
        seen.add(filename)
        subject, frame = identity(record.get('name') or PurePosixPath(filename).name)
        boxes = record.get('boundingBoxes') or []
        if not isinstance(boxes, list) or any(not isinstance(b, dict) for b in boxes):
            raise ValueError(f'Invalid boundingBoxes: {filename}')
        eye, mouth = state(boxes, EYES, 2), state(boxes, MOUTH, 1)
        reasons = []
        if subject is None:
            reasons.append('unparsed_identity')
        if eye == 'UNKNOWN':
            reasons.append('missing_conflicting_or_wrong_count_eyes')
        if mouth == 'UNKNOWN':
            reasons.append('missing_conflicting_or_wrong_count_mouth')
        unknown = sorted({str(b.get('label')) for b in boxes if b.get('label') not in EYES | MOUTH})
        if unknown:
            reasons.append('unrecognized_labels:' + ','.join(unknown))
        if boxes_index is not None:
            other = boxes_index.get(PurePosixPath(filename).name)
            canonical = lambda bs: sorted(json.dumps({k: b.get(k) for k in ('label', 'x', 'y', 'width', 'height')}, sort_keys=True) for b in bs)
            if other is None or canonical(other) != canonical(boxes):
                reasons.append('bounding_boxes_mismatch_or_missing')
        rows.append(dict(zip(COLUMNS, (filename, subject, frame, eye, mouth))))
        manifest.append({'filename': filename, 'original_name': record.get('name'), 'category': record.get('category')})
        if reasons:
            issues.append({'filename': filename, 'reasons': reasons})
    rows.sort(key=lambda r: (r['subject_id'] or '', r['frame_id'] if r['frame_id'] is not None else -1, r['filename']))
    return rows, issues, manifest


def write_csv(path, rows, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def audit(rows, manifest, expected_step=5):
    groups = defaultdict(list)
    categories = {r['filename']: r['category'] for r in manifest}
    result = {'total_frames': len(rows), 'eye_distribution': dict(Counter(r['eye_state'] for r in rows)),
              'mouth_distribution': dict(Counter(r['mouth_state'] for r in rows)), 'subjects': {}}
    sequences = []
    for row in rows:
        groups[row['subject_id'] or 'UNKNOWN'].append(row)
    for subject, items in sorted(groups.items()):
        ids = [r['frame_id'] for r in items if r['frame_id'] is not None]
        deltas = Counter(b-a for a, b in zip(ids, ids[1:]))
        cats = sorted({categories[r['filename']] or 'UNKNOWN' for r in items})
        result['subjects'][subject] = {'frames': len(items), 'first_frame': min(ids) if ids else None,
            'last_frame': max(ids) if ids else None, 'duplicate_frame_ids': [k for k,v in Counter(ids).items() if v > 1],
            'frame_step_distribution': dict(deltas), 'categories': cats,
            'eye_distribution': dict(Counter(r['eye_state'] for r in items)),
            'mouth_distribution': dict(Counter(r['mouth_state'] for r in items))}
        previous = None
        for row in items:
            delta = row['frame_id'] - previous['frame_id'] if previous and row['frame_id'] is not None and previous['frame_id'] is not None else None
            sequences.append({**row, 'category': categories[row['filename']], 'delta_frame': delta,
                              'step_anomaly': delta is not None and delta != expected_step})
            previous = row
    result['subjects_in_multiple_categories'] = [s for s,v in result['subjects'].items() if len(v['categories']) > 1]
    result['sequence_note'] = 'Frame order only; no FPS, duration, or continuity across recordings is assumed.'
    return result, sequences


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--info', type=Path, required=True)
    parser.add_argument('--boxes', type=Path, nargs='*', default=[])
    parser.add_argument('--output', type=Path, default=Path('data/processed/labels.csv'))
    parser.add_argument('--report-dir', type=Path, default=Path('reports'))
    parser.add_argument('--expected-step', type=int, default=5)
    args = parser.parse_args()
    index = {} if args.boxes else None
    for path in args.boxes:
        loaded = json.loads(path.read_text(encoding='utf-8-sig'))['boundingBoxes']
        for key, value in loaded.items():
            if key in index:
                raise ValueError(f'Duplicate key across box files: {key}')
            index[key] = value
    rows, issues, manifest = parse(json.loads(args.info.read_text(encoding='utf-8-sig')), index)
    for item in manifest:
        if not (args.info.parent / item['filename']).is_file():
            issues.append({'filename': item['filename'], 'reasons': ['image_missing_relative_to_info']})
    summary, sequences = audit(rows, manifest, args.expected_step)
    summary.update({'info_source': str(args.info.resolve()), 'boxes_sources': [str(p.resolve()) for p in args.boxes],
                    'issue_records': len(issues), 'issues': issues})
    write_csv(args.output, rows, COLUMNS)
    write_csv(args.report_dir / 'manifest.csv', manifest, ['filename', 'original_name', 'category'])
    write_csv(args.report_dir / 'sequences.csv', sequences, COLUMNS + ['category', 'delta_frame', 'step_anomaly'])
    args.report_dir.mkdir(parents=True, exist_ok=True)
    (args.report_dir / 'label_audit.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('subjects', 'issues')}, indent=2))


if __name__ == '__main__':
    main()
