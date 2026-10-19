"""Paired SR data with explicit, balanced degradation-level coverage."""

import os.path as osp

from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, imfrombytes, img2tensor, scandir
from basicsr.utils.color_util import bgr2ycbcr
from basicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class MultiLevelPairedImageDataset(data.Dataset):
    """Pair every GT image with every configured offline degradation level.

    The logical dataset is the Cartesian product of GT filenames and LQ roots.
    Consequently, each level has exactly the same number of samples and the
    selected level is reproducible from the sampled dataset index. This avoids
    relying on a pre-merged LQ directory whose provenance and level balance are
    otherwise difficult to verify.
    """

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend'].copy()
        if self.io_backend_opt.get('type') != 'disk':
            raise ValueError(
                'MultiLevelPairedImageDataset currently requires the disk '
                f'backend, got {self.io_backend_opt.get("type")!r}.'
            )

        self.mean = opt.get('mean')
        self.std = opt.get('std')
        self.gt_folder = osp.expanduser(opt['dataroot_gt'])
        raw_lq_folders = opt.get('dataroot_lq_levels')
        if raw_lq_folders is None:
            raw_lq_folders = opt['dataroot_lq']
        if isinstance(raw_lq_folders, str):
            raw_lq_folders = [raw_lq_folders]
        self.lq_folders = [osp.expanduser(path) for path in raw_lq_folders]
        if not self.lq_folders:
            raise ValueError('At least one LQ root is required.')

        gt_names = sorted(
            name for name in scandir(self.gt_folder, full_path=False)
            if osp.splitext(name)[1].lower()
            in {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
        )
        if not gt_names:
            raise RuntimeError(f'No GT images found in {self.gt_folder}.')
        gt_name_set = set(gt_names)
        for lq_folder in self.lq_folders:
            if not osp.isdir(lq_folder):
                raise RuntimeError(f'LQ root does not exist: {lq_folder}')
            lq_names = {
                name for name in scandir(lq_folder, full_path=False)
                if osp.splitext(name)[1].lower()
                in {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
            }
            missing = gt_name_set - lq_names
            unexpected = lq_names - gt_name_set
            if missing or unexpected:
                examples = sorted(missing)[:3]
                raise RuntimeError(
                    f'Unaligned root {lq_folder}: {len(missing)} missing and '
                    f'{len(unexpected)} unexpected filenames; examples={examples}'
                )

        self.gt_names = gt_names
        self.level_names = [osp.basename(osp.normpath(path)) for path in self.lq_folders]

    def __getitem__(self, index):
        if self.file_client is None:
            backend_type = self.io_backend_opt.pop('type')
            self.file_client = FileClient(backend_type, **self.io_backend_opt)

        level_index = index % len(self.lq_folders)
        gt_index = index // len(self.lq_folders)
        filename = self.gt_names[gt_index]
        gt_path = osp.join(self.gt_folder, filename)
        lq_path = osp.join(self.lq_folders[level_index], filename)

        img_gt = imfrombytes(self.file_client.get(gt_path, 'gt'), float32=True)
        img_lq = imfrombytes(self.file_client.get(lq_path, 'lq'), float32=True)
        scale = self.opt['scale']

        if self.opt['phase'] == 'train':
            img_gt, img_lq = paired_random_crop(
                img_gt, img_lq, self.opt['gt_size'], scale, gt_path
            )
            img_gt, img_lq = augment(
                [img_gt, img_lq],
                self.opt['use_hflip'],
                self.opt['use_rot'],
            )

        if self.opt.get('color') == 'y':
            img_gt = bgr2ycbcr(img_gt, y_only=True)[..., None]
            img_lq = bgr2ycbcr(img_lq, y_only=True)[..., None]

        if self.opt['phase'] != 'train':
            img_gt = img_gt[
                :img_lq.shape[0] * scale,
                :img_lq.shape[1] * scale,
                :,
            ]

        img_gt, img_lq = img2tensor(
            [img_gt, img_lq], bgr2rgb=True, float32=True
        )
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': lq_path,
            'gt_path': gt_path,
            'degradation_level': self.level_names[level_index],
            'degradation_level_index': level_index,
        }

    def __len__(self):
        return len(self.gt_names) * len(self.lq_folders)
