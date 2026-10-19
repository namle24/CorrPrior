import hashlib
import importlib.util
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from PIL import Image
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RENDER = load_script("render_result_figure")
CHECK = load_script("check_release")


@pytest.mark.parametrize("name", ["architecture.png", "qualitative_comparison.jpg"])
def test_documented_images_decode_with_recorded_dimensions(name):
    manifest = json.loads((ROOT / "docs/assets/manifest.json").read_text())
    record = next(row for row in manifest["files"] if row["path"].endswith("/" + name))
    with Image.open(ROOT / record["path"]) as image:
        assert image.size == (record["width"], record["height"])
        image.verify()


def test_editable_architecture_is_uncompressed_drawio():
    tree = ET.parse(ROOT / "docs/assets/architecture.drawio")
    assert tree.find(".//mxGraphModel") is not None
    assert len(tree.findall(".//mxCell")) > 50


def test_plot_matches_csv_and_has_accessible_svg_labels():
    actual = (ROOT / "docs/assets/psnr_effects.svg").read_text()
    assert actual == RENDER.render(RENDER.read_effects())
    root = ET.fromstring(actual)
    namespace = {"svg": "http://www.w3.org/2000/svg"}
    assert root.find("svg:title", namespace) is not None
    assert "not trained-baseline gains" in root.find("svg:desc", namespace).text
    assert len(root.findall(".//svg:circle", namespace)) == 5


def test_effects_retain_reported_values_and_groups():
    rows = RENDER.read_effects()
    assert [(r["group"], r["dataset"], r["delta_psnr_db"], r["ci_low_db"], r["ci_high_db"])
            for r in rows] == [
        ("matched", "RealSR", 0.1321, 0.0903, 0.1827),
        ("matched", "RealSR", -0.0017, -0.0065, 0.0031),
        ("matched", "SVSR", 0.0030, -0.0122, 0.0184),
        ("intervention", "RealSR", 0.3194, 0.2604, 0.3789),
        ("intervention", "RealSR", 0.3678, 0.2793, 0.4546),
    ]


@pytest.mark.parametrize("problem", ["modified", "wrong_signature", "forbidden_path", "missing", "symlink"])
def test_asset_allowlist_rejects_unverified_or_invalid_assets(tmp_path, problem):
    asset = tmp_path / "docs/assets/example.png"
    asset.parent.mkdir(parents=True)
    payload = (ROOT / "docs/assets/architecture.png").read_bytes()
    asset.write_bytes(payload)
    record = {"path": "docs/assets/example.png", "sha256": hashlib.sha256(payload).hexdigest()}
    if problem == "modified":
        asset.write_bytes(payload + b"changed")
    elif problem == "wrong_signature":
        asset.write_bytes(b"not an image")
        record["sha256"] = hashlib.sha256(asset.read_bytes()).hexdigest()
    elif problem == "forbidden_path":
        record["path"] = "checkpoints/model.pth"
    elif problem == "missing":
        record["path"] = "docs/assets/missing.png"
    else:
        alias = asset.parent / "alias.png"
        alias.symlink_to(asset)
        record["path"] = "docs/assets/alias.png"
    (asset.parent / "manifest.json").write_text(json.dumps({"files": [record]}))
    errors, binary_assets = CHECK.check_assets(tmp_path)
    assert errors
    assert not binary_assets
