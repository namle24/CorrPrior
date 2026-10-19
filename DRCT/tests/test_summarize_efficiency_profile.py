import json
from pathlib import Path

import pytest

from scripts.summarize_efficiency_profile import summarize


def _record(model, size, trial):
    is_candidate = model == 'C2'
    return {
        'total_parameters': 120 if is_candidate else 100,
        'trainable_parameters': 110 if is_candidate else 100,
        'checkpoint_size_mib': 12.0 if is_candidate else 10.0,
        'input_shape': [1, 3, size, size],
        'output_shape': [1, 3, size * 4, size * 4],
        'cuda_device': 'Test GPU',
        'torch_version': 'test',
        'torch_cuda_version': 'test',
        'cudnn_version': 1,
        'warmup': 20,
        'repeats': 100,
        'seed': 0,
        'latency_ms_mean_cuda': (20.0 if is_candidate else 10.0) + trial,
        'latency_ms_mean_wall': (21.0 if is_candidate else 11.0) + trial,
        'peak_memory_mib': 60.0 if is_candidate else 50.0,
    }


def _write_complete_profile(root):
    for model in ('B0', 'C2'):
        for size in (64, 128):
            for trial in range(1, 6):
                path = root / f'{model}_{size}x{size}_trial{trial}.json'
                path.write_text(
                    json.dumps(_record(model, size, trial)), encoding='utf-8'
                )


def test_summary_validates_trials_and_reports_candidate_overhead(tmp_path):
    _write_complete_profile(tmp_path)

    report = summarize(tmp_path)

    assert report['trial_count'] == 5
    size = report['sizes']['64x64']
    assert size['models']['B0']['latency_ms_mean_cuda']['median'] == 13.0
    assert size['models']['C2']['latency_ms_mean_cuda']['sample_std'] == pytest.approx(
        1.5811388301
    )
    assert size['C2_minus_B0']['total_parameters'] == 20
    assert size['C2_minus_B0']['checkpoint_size_mib'] == 2.0
    assert size['C2_minus_B0']['latency_ratio_median'] == pytest.approx(23.0 / 13.0)
    assert size['C2_minus_B0']['peak_memory_ratio_median'] == 1.2


def test_summary_fails_closed_when_a_trial_is_missing(tmp_path):
    _write_complete_profile(tmp_path)
    (tmp_path / 'C2_128x128_trial5.json').unlink()

    with pytest.raises(FileNotFoundError, match='Missing profile trial'):
        summarize(tmp_path)


def test_summary_fails_when_stable_metadata_changes(tmp_path):
    _write_complete_profile(tmp_path)
    path = tmp_path / 'B0_64x64_trial3.json'
    record = json.loads(path.read_text(encoding='utf-8'))
    record['total_parameters'] += 1
    path.write_text(json.dumps(record), encoding='utf-8')

    with pytest.raises(ValueError, match='total_parameters'):
        summarize(tmp_path)
