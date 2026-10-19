# flake8: noqa
import os.path as osp

#import hat.archs
#import hat.data
#import hat.models

from drct.archs import *
from drct.data import *
from drct.models import *

import basicsr.train as basicsr_train
from basicsr.utils import get_root_logger
from basicsr.utils import set_random_seed

from drct.runtime_options import (
    build_exact_resume_cpu_prefetcher,
    configure_exact_resume_skip_batches,
    reseed_after_option_overrides,
)


_basicsr_parse_options = basicsr_train.parse_options
_basicsr_cpu_prefetcher = basicsr_train.CPUPrefetcher


def _parse_options_with_output_root(root_path, is_train=True):
    """Allow large experiment artifacts to live outside the source disk."""
    opt, args = _basicsr_parse_options(root_path, is_train=is_train)
    reseed_after_option_overrides(opt, set_random_seed)
    exact_resume_skip = configure_exact_resume_skip_batches(opt)
    if is_train and exact_resume_skip:
        basicsr_train.CPUPrefetcher = build_exact_resume_cpu_prefetcher(
            _basicsr_cpu_prefetcher,
            exact_resume_skip,
            log=lambda message: get_root_logger(
                logger_name='basicsr'
            ).info(message),
        )
    else:
        basicsr_train.CPUPrefetcher = _basicsr_cpu_prefetcher
    output_root = opt.get('output_root')
    if is_train and output_root:
        experiments_root = osp.join(osp.expanduser(output_root), opt['name'])
        opt['path']['experiments_root'] = experiments_root
        opt['path']['models'] = osp.join(experiments_root, 'models')
        opt['path']['training_states'] = osp.join(experiments_root, 'training_states')
        opt['path']['log'] = experiments_root
        opt['path']['visualization'] = osp.join(experiments_root, 'visualization')
    return opt, args


basicsr_train.parse_options = _parse_options_with_output_root

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    basicsr_train.train_pipeline(root_path)
