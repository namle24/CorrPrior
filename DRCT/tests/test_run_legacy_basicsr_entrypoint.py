import sys

from scripts.run_legacy_basicsr_entrypoint import (
    install_generated_version_module,
)


def test_restores_setup_generated_version_without_writing_source(tmp_path):
    package = tmp_path / 'restoration_package'
    package.mkdir()
    (tmp_path / 'VERSION').write_text('0.3.0\n', encoding='utf-8')

    module_name = 'restoration_package.version'
    previous = sys.modules.pop(module_name, None)
    try:
        install_generated_version_module(tmp_path, 'restoration_package')
        restored = sys.modules[module_name]
        assert restored.__version__ == '0.3.0'
        assert restored.version_info == (0, 3, 0)
        assert restored.__gitsha__ == 'frozen-source'
        assert not (package / 'version.py').exists()
    finally:
        sys.modules.pop(module_name, None)
        if previous is not None:
            sys.modules[module_name] = previous


def test_does_not_invent_metadata_for_absent_package(tmp_path):
    install_generated_version_module(tmp_path, 'absent_package')
    assert 'absent_package.version' not in sys.modules
