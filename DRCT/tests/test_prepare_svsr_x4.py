import json
from pathlib import Path

from PIL import Image
import pytest

from scripts.prepare_svsr_x4 import build_pairing, prepare


def write_png(path, size):
    Image.new('RGB', size, color=(10, 20, 30)).save(path)


def make_dataset(tmp_path):
    lr_root = tmp_path / 'LR'
    hr_root = tmp_path / 'HR' / 'X4'
    lr_root.mkdir(parents=True)
    hr_root.mkdir(parents=True)
    write_png(lr_root / '00001_Camera_Lens_f8_LR_ISO100.png', (4, 3))
    write_png(lr_root / '00001_Camera_Lens_f8_LR_ISO800.png', (4, 3))
    write_png(hr_root / '00001_Camera_Lens_f8_X4-HR_ISO100.png', (16, 12))
    return lr_root, hr_root


def test_build_pairing_maps_all_iso_captures_to_one_hr(tmp_path):
    lr_root, hr_root = make_dataset(tmp_path)
    pairs = build_pairing(lr_root, hr_root)
    assert len(pairs) == 2
    assert {pair['lr_iso'] for pair in pairs} == {100, 800}
    assert len({pair['hr_path'] for pair in pairs}) == 1


def test_prepare_creates_exact_stem_gt_view_and_manifest(tmp_path):
    lr_root, hr_root = make_dataset(tmp_path)
    output = tmp_path / 'paired_gt'
    manifest = tmp_path / 'manifest.json'
    report = prepare(lr_root, hr_root, output, manifest, link_mode='hardlink')
    assert report['pair_count'] == 2
    assert report['unique_scene_count'] == 1
    assert sorted(path.stem for path in output.glob('*.png')) == sorted(
        path.stem for path in lr_root.glob('*.png')
    )
    assert json.loads(manifest.read_text())['pairing_sha256'] == report['pairing_sha256']


def test_prepare_rejects_non_x4_pair(tmp_path):
    lr_root, hr_root = make_dataset(tmp_path)
    write_png(hr_root / '00001_Camera_Lens_f8_X4-HR_ISO100.png', (15, 12))
    with pytest.raises(RuntimeError, match='Non-x4 pair'):
        prepare(lr_root, hr_root, tmp_path / 'paired_gt', tmp_path / 'manifest.json')


def test_build_pairing_rejects_unmatched_lr(tmp_path):
    lr_root, hr_root = make_dataset(tmp_path)
    write_png(lr_root / '99999_Camera_Lens_f8_LR_ISO100.png', (4, 3))
    with pytest.raises(RuntimeError, match='No x4 HR image'):
        build_pairing(lr_root, hr_root)


def test_prepare_refuses_existing_output(tmp_path):
    lr_root, hr_root = make_dataset(tmp_path)
    output = tmp_path / 'paired_gt'
    output.mkdir()
    with pytest.raises(RuntimeError, match='Refusing to overwrite'):
        prepare(lr_root, hr_root, output, tmp_path / 'manifest.json')
