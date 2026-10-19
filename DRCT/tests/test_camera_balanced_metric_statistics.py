import csv

import pytest


np = pytest.importorskip('numpy')

from scripts.camera_balanced_metric_statistics import evaluate  # noqa: E402


def write_metrics(path, rows):
    with path.open('w', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def camera_files(root, name, images, psnr_delta, dists_delta):
    baseline = root / f'{name}_baseline.csv'
    candidate = root / f'{name}_candidate.csv'
    stems = [f'{name}_{index}' for index in range(images)]
    write_metrics(baseline, [
        {'image': stem, 'psnr': 30.0, 'dists': 0.2} for stem in stems
    ])
    write_metrics(candidate, [
        {
            'image': stem,
            'psnr': 30.0 + psnr_delta,
            'dists': 0.2 + dists_delta,
        }
        for stem in stems
    ])
    return [name, str(baseline), str(candidate)]


def test_equal_camera_weighting_with_unequal_subset_sizes(tmp_path):
    cameras = [
        camera_files(tmp_path, 'Canon', 1, 0.2, -0.02),
        camera_files(tmp_path, 'Nikon', 3, 0.0, 0.00),
    ]
    rows = evaluate(
        cameras, ['psnr', 'dists'], bootstrap_samples=100,
        permutation_samples=100, seed=0,
    )
    by_metric = {row['metric']: row for row in rows}
    assert by_metric['psnr']['candidate_minus_baseline'] == pytest.approx(0.1)
    assert by_metric['dists']['candidate_minus_baseline'] == pytest.approx(-0.01)
    assert by_metric['dists']['favorable_effect'] == pytest.approx(0.01)
    assert by_metric['psnr']['camera_counts'] == '{"Canon": 1, "Nikon": 3}'


def test_rejects_mismatched_stems_within_camera(tmp_path):
    camera = camera_files(tmp_path, 'Canon', 2, 0.1, -0.01)
    write_metrics(tmp_path / 'Canon_candidate.csv', [
        {'image': 'wrong', 'psnr': 30.1, 'dists': 0.19}
    ])
    nikon = camera_files(tmp_path, 'Nikon', 2, 0.1, -0.01)
    with pytest.raises(ValueError, match='mismatched image stems'):
        evaluate([camera, nikon], ['psnr'], bootstrap_samples=10,
                 permutation_samples=10)


def test_requires_distinct_camera_names(tmp_path):
    first = camera_files(tmp_path, 'Canon', 2, 0.1, -0.01)
    second = ['Canon', first[1], first[2]]
    with pytest.raises(ValueError, match='Camera names must be unique'):
        evaluate([first, second], ['psnr'], bootstrap_samples=10,
                 permutation_samples=10)
