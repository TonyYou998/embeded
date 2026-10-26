"""Optional batch scaffold; run only after manually reviewing labels."""
import argparse
import csv
from collections import Counter
from pathlib import Path, PurePosixPath
from time import perf_counter
from features import ratios
from prepare_labels import COLUMNS, write_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labels', type=Path, default=Path('data/processed/labels.csv'))
    parser.add_argument('--image-root', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('data/features/features.csv'))
    parser.add_argument('--labels-reviewed', action='store_true')
    args = parser.parse_args()
    if not args.labels_reviewed:
        parser.error('Review labels.csv and reports first, then pass --labels-reviewed')
    if not args.model.is_file():
        parser.error('Provide a local MediaPipe Face Landmarker .task model')
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    options = vision.FaceLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=str(args.model)),
        running_mode=vision.RunningMode.IMAGE, num_faces=2)
    with args.labels.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    root = args.image_root.resolve()
    total = len(rows)
    counts = Counter()
    started = perf_counter()

    def report_progress(index, row):
        counts[row['status']] += 1
        elapsed = perf_counter() - started
        rate = index / elapsed if elapsed > 0 else 0
        remaining = (total - index) / rate if rate > 0 else 0
        print(
            f"[{index}/{total} | {index / total:.1%}] "
            f"{row['status']} | {rate:.1f} images/s | "
            f"ETA {remaining:.0f}s | {row['filename']}",
            flush=True,
        )

    print(f'Loaded {total} images. Initializing MediaPipe...', flush=True)
    with vision.FaceLandmarker.create_from_options(options) as detector:
        print('MediaPipe ready. Starting extraction...', flush=True)
        for index, row in enumerate(rows, start=1):
            row.update(ear_l=None, ear_r=None, mar=None, status='OK', error=None)
            path = (root / PurePosixPath(row['filename'])).resolve()
            if not path.is_relative_to(root):
                raise ValueError(f'Image path outside image root: {path}')
            if not path.is_file():
                row['status'] = 'IMAGE_MISSING'
                report_progress(index, row)
                continue
            try:
                image = mp.Image.create_from_file(str(path))
                faces = detector.detect(image).face_landmarks
                if len(faces) != 1:
                    row['status'] = 'NO_FACE' if not faces else 'MULTIPLE_FACES'
                else:
                    row.update(ratios(faces[0], image.width, image.height))
                    if any(row[k] is None for k in ('ear_l', 'ear_r', 'mar')):
                        row['status'] = 'DEGENERATE_GEOMETRY'
            except (ValueError, RuntimeError, OSError) as exc:
                row.update(status='EXTRACTION_ERROR', error=str(exc))
            report_progress(index, row)
    print(f'Saving CSV to {args.output.resolve()}...', flush=True)
    write_csv(args.output, rows, COLUMNS + ['ear_l', 'ear_r', 'mar', 'status', 'error'])
    print(f'Completed {total} images in {perf_counter() - started:.1f}s.', flush=True)
    for status, count in sorted(counts.items()):
        print(f'  {status}: {count}', flush=True)


if __name__ == '__main__':
    main()
