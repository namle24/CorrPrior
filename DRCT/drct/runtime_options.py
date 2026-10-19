"""Runtime option helpers shared by the DRCT train and test entrypoints."""

import os
from pathlib import Path


EXACT_RESUME_SKIP_ENV = 'RN_PD_MAE_EXACT_RESUME_SKIP_BATCHES'


def reseed_after_option_overrides(opt, seed_function):
    """Reapply the final parsed seed after BasicSR processes ``--force_yml``.

    Upstream BasicSR seeds Python, NumPy, and Torch before applying force-YAML
    overrides. Calling this helper immediately after parsing makes a forced
    ``manual_seed`` effective for the main process as well as the later data
    workers.
    """
    if 'manual_seed' not in opt:
        raise ValueError('Parsed options do not contain manual_seed.')
    seed = int(opt['manual_seed']) + int(opt.get('rank', 0))
    seed_function(seed)
    return seed


def configure_exact_resume_skip_batches(opt, environ=None):
    """Validate an explicitly requested deterministic within-epoch resume.

    BasicSR restores the optimizer iteration but restarts the dataloader at the
    beginning of the saved epoch.  The journal SwinIR transfer was interrupted
    within epoch zero, so an exact continuation must replay the already
    consumed batches before the next optimizer step.  This opt-in environment
    setting is intentionally narrow and leaves all ordinary runs unchanged.
    """
    environ = os.environ if environ is None else environ
    raw_value = environ.get(EXACT_RESUME_SKIP_ENV)
    if raw_value is None:
        opt['exact_resume_skip_batches'] = 0
        return 0
    try:
        skip_batches = int(raw_value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f'{EXACT_RESUME_SKIP_ENV} must be a non-negative integer.'
        ) from error
    if skip_batches < 0:
        raise ValueError(
            f'{EXACT_RESUME_SKIP_ENV} must be a non-negative integer.'
        )
    if skip_batches == 0:
        opt['exact_resume_skip_batches'] = 0
        return 0

    resume_state = opt.get('path', {}).get('resume_state')
    if not resume_state:
        raise ValueError(
            f'{EXACT_RESUME_SKIP_ENV} requires path.resume_state.'
        )
    state_stem = Path(str(resume_state)).stem
    if not state_stem.isdigit() or int(state_stem) != skip_batches:
        raise ValueError(
            'The resume-state filename iteration must equal the exact-resume '
            f'batch count ({skip_batches}); got {resume_state!r}.'
        )
    if int(opt.get('num_gpu', 0)) != 1:
        raise ValueError('Exact within-epoch resume currently requires num_gpu=1.')
    prefetch_mode = opt.get('datasets', {}).get('train', {}).get(
        'prefetch_mode'
    )
    if prefetch_mode not in (None, 'cpu'):
        raise ValueError(
            'Exact within-epoch resume requires CPU prefetching.'
        )
    opt['exact_resume_skip_batches'] = skip_batches
    return skip_batches


def build_exact_resume_cpu_prefetcher(base_class, skip_batches, log=None):
    """Wrap a BasicSR CPU prefetcher to replay a consumed batch prefix once."""
    if not isinstance(skip_batches, int) or skip_batches < 0:
        raise ValueError('skip_batches must be a non-negative integer.')

    class ExactResumeCPUPrefetcher(base_class):
        def __init__(self, loader):
            super().__init__(loader)
            self._exact_resume_skip_batches = skip_batches
            self._exact_resume_skip_applied = False

        def reset(self):
            super().reset()
            if (
                self._exact_resume_skip_applied
                or self._exact_resume_skip_batches == 0
            ):
                return
            for consumed in range(self._exact_resume_skip_batches):
                try:
                    next(self.loader)
                except StopIteration as error:
                    raise RuntimeError(
                        'Exact-resume prefix exceeds the number of batches in '
                        f'the saved epoch: requested '
                        f'{self._exact_resume_skip_batches}, consumed {consumed}.'
                    ) from error
            self._exact_resume_skip_applied = True
            if log is not None:
                log(
                    'Exact-resume dataloader replay consumed '
                    f'{self._exact_resume_skip_batches} batches.'
                )

    return ExactResumeCPUPrefetcher
