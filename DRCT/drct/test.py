# flake8: noqa
import os.path as osp

import drct.archs
import drct.data
import drct.models
import basicsr.test as basicsr_test
from basicsr.test import test_pipeline
from basicsr.utils import set_random_seed

from drct.runtime_options import reseed_after_option_overrides


_basicsr_parse_options = basicsr_test.parse_options


def _parse_options_with_output_root(root_path, is_train=False):
    """Allow test artifacts to be written outside a nearly full source disk."""
    opt, args = _basicsr_parse_options(root_path, is_train=is_train)
    reseed_after_option_overrides(opt, set_random_seed)
    output_root = opt.get('output_root')
    if not is_train and output_root:
        results_root = osp.join(osp.expanduser(output_root), opt['name'])
        opt['path']['results_root'] = results_root
        opt['path']['log'] = results_root
        opt['path']['visualization'] = osp.join(results_root, 'visualization')
    return opt, args


basicsr_test.parse_options = _parse_options_with_output_root

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    test_pipeline(root_path)
