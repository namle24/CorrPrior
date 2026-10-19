"""Build and audit an exact-stem SVSR x4 benchmark view.

SVSR contains several noisy LR captures for each noise-free HR scene.  Their
filenames are intentionally different, which is inconvenient for evaluators
that require one-to-one stem alignment.  This script creates a non-destructive
GT view with one link per LR capture and records the complete pairing and image
hashes.  The original LR and HR files are never modified.
"""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import shutil

from PIL import Image


LR_PATTERN = re.compile(r'^(?P<scene>.+)_LR_ISO(?P<iso>[0-9]+)$')
HR_PATTERN = re.compile(r'^(?P<scene>.+)_X4-HR_ISO(?P<iso>[0-9]+)$')


def file_sha256(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open('rb') as file:
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def png_index(root, pattern, role):
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise RuntimeError(f'{role} directory does not exist: {root}')
    entries = []
    for path in sorted(root.glob('*.png')):
        match = pattern.fullmatch(path.stem)
        if match is None:
            raise RuntimeError(f'Unexpected {role} filename: {path.name}')
        entries.append((path, match.groupdict()))
    if not entries:
        raise RuntimeError(f'No PNG images found in {root}.')
    return entries


def build_pairing(lr_root, hr_root):
    lr_entries = png_index(lr_root, LR_PATTERN, 'LR')
    hr_entries = png_index(hr_root, HR_PATTERN, 'HR')
    hr_by_scene = {}
    for path, metadata in hr_entries:
        scene = metadata['scene']
        if scene in hr_by_scene:
            raise RuntimeError(
                f'More than one x4 HR image for scene {scene!r}: '
                f'{hr_by_scene[scene]} and {path}'
            )
        hr_by_scene[scene] = path

    pairs = []
    used_scenes = set()
    for lr_path, metadata in lr_entries:
        scene = metadata['scene']
        if scene not in hr_by_scene:
            raise RuntimeError(f'No x4 HR image for LR scene {scene!r}.')
        used_scenes.add(scene)
        pairs.append({
            'stem': lr_path.stem,
            'scene': scene,
            'lr_iso': int(metadata['iso']),
            'lr_path': lr_path,
            'hr_path': hr_by_scene[scene],
        })

    unused = sorted(set(hr_by_scene) - used_scenes)
    if unused:
        raise RuntimeError(
            f'{len(unused)} x4 HR scenes have no LR capture; examples={unused[:5]}'
        )
    return pairs


def image_hw(path):
    with Image.open(path) as image:
        width, height = image.size
    return height, width


def create_link(source, destination, mode):
    if mode == 'hardlink':
        os.link(source, destination)
    elif mode == 'symlink':
        destination.symlink_to(Path(source).resolve())
    elif mode == 'copy':
        shutil.copy2(source, destination)
    else:  # protected by argparse, retained for direct callers
        raise ValueError(f'Unsupported link mode: {mode}')


def aggregate_digest(records):
    digest = hashlib.sha256()
    for record in records:
        for key in ('stem', 'scene', 'lr_sha256', 'hr_sha256'):
            digest.update(str(record[key]).encode('utf-8'))
            digest.update(b'\0')
        digest.update(b'\n')
    return digest.hexdigest()


def prepare(lr_root, hr_root, gt_output_root, manifest_path, link_mode='hardlink'):
    lr_root = Path(lr_root).expanduser().resolve()
    hr_root = Path(hr_root).expanduser().resolve()
    gt_output_root = Path(gt_output_root).expanduser().resolve()
    manifest_path = Path(manifest_path).expanduser().resolve()
    if gt_output_root.exists():
        raise RuntimeError(f'Refusing to overwrite existing GT view: {gt_output_root}')
    if manifest_path.exists():
        raise RuntimeError(f'Refusing to overwrite existing manifest: {manifest_path}')

    pairs = build_pairing(lr_root, hr_root)
    gt_output_root.mkdir(parents=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    hash_cache = {}
    records = []
    for pair in pairs:
        lr_path = pair['lr_path']
        hr_path = pair['hr_path']
        lr_hw = image_hw(lr_path)
        hr_hw = image_hw(hr_path)
        if hr_hw != (4 * lr_hw[0], 4 * lr_hw[1]):
            raise RuntimeError(
                f'Non-x4 pair {pair["stem"]}: LR={lr_hw}, HR={hr_hw}.'
            )
        destination = gt_output_root / f'{pair["stem"]}.png'
        create_link(hr_path, destination, link_mode)
        for path in (lr_path, hr_path):
            if path not in hash_cache:
                hash_cache[path] = file_sha256(path)
        records.append({
            'stem': pair['stem'],
            'scene': pair['scene'],
            'lr_iso': pair['lr_iso'],
            'lr_height': lr_hw[0],
            'lr_width': lr_hw[1],
            'hr_height': hr_hw[0],
            'hr_width': hr_hw[1],
            'lr_file': lr_path.name,
            'hr_file': hr_path.name,
            'lr_sha256': hash_cache[lr_path],
            'hr_sha256': hash_cache[hr_path],
        })

    report = {
        'schema_version': 1,
        'dataset': 'SVSR',
        'scale': 4,
        'lr_root': str(lr_root),
        'original_hr_root': str(hr_root),
        'paired_gt_root': str(gt_output_root),
        'link_mode': link_mode,
        'pair_count': len(records),
        'unique_scene_count': len({record['scene'] for record in records}),
        'lr_iso_counts': dict(sorted(Counter(
            str(record['lr_iso']) for record in records
        ).items())),
        'pairing_sha256': aggregate_digest(records),
        'pairs': records,
    }
    with manifest_path.open('w', encoding='utf-8') as file:
        json.dump(report, file, indent=2)
        file.write('\n')
    return report


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lr_root', required=True)
    parser.add_argument('--hr_root', required=True)
    parser.add_argument('--gt_output_root', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument(
        '--link_mode', choices=('hardlink', 'symlink', 'copy'), default='hardlink'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    report = prepare(
        args.lr_root,
        args.hr_root,
        args.gt_output_root,
        args.manifest,
        args.link_mode,
    )
    print(json.dumps({key: report[key] for key in (
        'dataset', 'scale', 'pair_count', 'unique_scene_count',
        'lr_iso_counts', 'pairing_sha256', 'paired_gt_root'
    )}, indent=2))


if __name__ == '__main__':
    main()
