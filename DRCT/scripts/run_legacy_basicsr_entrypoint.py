"""Launch a frozen legacy restoration entrypoint on a current PyTorch stack.

Some frozen source archives omit version modules generated only during package
installation, and legacy BasicSR code imports a torchvision module removed by
newer releases. This bootstrap restores only those import-time compatibility
symbols, then executes the authors' unchanged entrypoint with the remaining
command-line arguments.
"""

import argparse
import importlib
from pathlib import Path
import runpy
import sys
import types


def install_generated_version_module(source_root, package_name):
    """Restore package metadata that the upstream setup script would create."""
    source_root = Path(source_root).expanduser().resolve()
    package_root = source_root / package_name
    version_path = package_root / 'version.py'
    module_name = f'{package_name}.version'
    if not package_root.is_dir() or version_path.is_file() or module_name in sys.modules:
        return

    root_version = source_root / 'VERSION'
    if not root_version.is_file():
        raise RuntimeError(
            f'Cannot restore missing {module_name}: {root_version} is absent.'
        )
    version = root_version.read_text(encoding='utf-8').strip()
    version_module = types.ModuleType(module_name)
    version_module.__version__ = version
    version_module.__gitsha__ = 'frozen-source'
    version_module.version_info = tuple(
        int(part) if part.isdigit() else part for part in version.split('.')
    )
    sys.modules[module_name] = version_module


def install_compatibility_modules(source_root, version_packages=None):
    """Install metadata and torchvision aliases without changing model code."""
    source_root = Path(source_root).expanduser().resolve()

    try:
        importlib.import_module('torchvision.transforms.functional_tensor')
    except ModuleNotFoundError:
        from torchvision.transforms.functional import rgb_to_grayscale

        functional_tensor = types.ModuleType(
            'torchvision.transforms.functional_tensor'
        )
        functional_tensor.rgb_to_grayscale = rgb_to_grayscale
        sys.modules['torchvision.transforms.functional_tensor'] = (
            functional_tensor
        )

    packages = ['basicsr', *(version_packages or [])]
    for package_name in dict.fromkeys(packages):
        install_generated_version_module(source_root, package_name)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source_root', required=True)
    parser.add_argument('--entrypoint', required=True)
    parser.add_argument(
        '--version_package', action='append', default=[],
        help='Package whose setup-generated version.py should be restored.',
    )
    args, remaining = parser.parse_known_args(argv)

    source_root = Path(args.source_root).expanduser().resolve()
    entrypoint = Path(args.entrypoint).expanduser().resolve()
    if source_root not in entrypoint.parents:
        raise RuntimeError('Entrypoint must be inside the frozen source root.')
    if not entrypoint.is_file():
        raise RuntimeError(f'Entrypoint does not exist: {entrypoint}')

    sys.path.insert(0, str(source_root))
    install_compatibility_modules(source_root, args.version_package)
    sys.argv = [str(entrypoint), *remaining]
    runpy.run_path(str(entrypoint), run_name='__main__')


if __name__ == '__main__':
    main()
