import torch
from torch.nn import functional as F
from collections import OrderedDict

from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.models.sr_model import SRModel
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img

import math
from tqdm import tqdm
import os
import csv
import re
from os import path as osp

@MODEL_REGISTRY.register()
class DRCTModel(SRModel):
    @staticmethod
    def _synchronize_auxiliary_modules(net_g, net_g_ema):
        """Copy journal-only branches and return the synchronized names."""
        synchronized = []
        for module_name in ('mae_wrapper', 'sft_layers'):
            source = getattr(net_g, module_name, None)
            target = getattr(net_g_ema, module_name, None)
            if source is None and target is None:
                continue
            if source is None or target is None:
                raise RuntimeError(
                    f'EMA auxiliary-module mismatch for {module_name}: '
                    f'train={source is not None}, ema={target is not None}'
                )
            target.load_state_dict(source.state_dict(), strict=True)
            synchronized.append(module_name)
        return synchronized

    def init_training_settings(self):
        """Initialize training and align auxiliary branches in the EMA model.

        BasicSR constructs ``net_g`` and ``net_g_ema`` independently before it
        loads the shared SR checkpoint.  Parameters absent from that checkpoint
        therefore keep independent initializations.  This is especially
        problematic for the frozen-random-prior control: training and EMA would
        otherwise use two different random encoders, and the usual EMA update
        would slowly mix them.  Copy only the newly added auxiliary modules
        here.  Backbone parameters retain the normal checkpoint/EMA behavior.
        """
        super().init_training_settings()
        if self.ema_decay <= 0 or not hasattr(self, 'net_g_ema'):
            return

        net_g = self.get_bare_model(self.net_g)
        net_g_ema = self.get_bare_model(self.net_g_ema)
        synchronized = self._synchronize_auxiliary_modules(net_g, net_g_ema)

        if synchronized:
            get_root_logger().info(
                'Synchronized EMA auxiliary modules from net_g: %s',
                ', '.join(synchronized),
            )

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)

        l_total = 0
        loss_dict = OrderedDict()

        # Pixel loss (standard DRCT objective)
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix

        # Optional perceptual/style losses if configured
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style

        # PD-MAE consistency loss for SFT mode
        net_g_bare = self.net_g.module if hasattr(self.net_g, 'module') else self.net_g
        weight_mae = float(self.opt.get('mae_weight', 0.05))
        if (weight_mae > 0 and getattr(net_g_bare, 'use_sft', False)
                and hasattr(net_g_bare, 'mae_wrapper')):
            with torch.no_grad():
                mae_feat_hr = net_g_bare.mae_wrapper(self.gt, input_upscale=1)
            mae_feat_sr = net_g_bare.mae_wrapper(self.output, input_upscale=1)
            l_g_mae = F.mse_loss(mae_feat_sr, mae_feat_hr)
            l_total += weight_mae * l_g_mae
            loss_dict['l_g_mae'] = l_g_mae

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def pre_process(self):
        # pad to multiplication of window_size
        window_size = self.opt['network_g']['window_size']
        self.scale = self.opt.get('scale', 1)
        self.mod_pad_h, self.mod_pad_w = 0, 0
        _, _, h, w = self.lq.size()
        if h % window_size != 0:
            self.mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            self.mod_pad_w = window_size - w % window_size
        self.img = F.pad(self.lq, (0, self.mod_pad_w, 0, self.mod_pad_h), 'reflect')

    def process(self):
        # model inference
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(self.img)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(self.img)
            # self.net_g.train()

    def tile_process(self):
        """It will first crop input images to tiles, and then process each tile.
        Finally, all the processed tiles are merged into one images.
        Modified from: https://github.com/ata4/esrgan-launcher
        """
        batch, channel, height, width = self.img.shape
        output_height = height * self.scale
        output_width = width * self.scale
        output_shape = (batch, channel, output_height, output_width)

        # start with black image
        self.output = self.img.new_zeros(output_shape)
        tiles_x = math.ceil(width / self.opt['tile']['tile_size'])
        tiles_y = math.ceil(height / self.opt['tile']['tile_size'])

        # loop over all tiles
        for y in range(tiles_y):
            for x in range(tiles_x):
                # extract tile from input image
                ofs_x = x * self.opt['tile']['tile_size']
                ofs_y = y * self.opt['tile']['tile_size']
                # input tile area on total image
                input_start_x = ofs_x
                input_end_x = min(ofs_x + self.opt['tile']['tile_size'], width)
                input_start_y = ofs_y
                input_end_y = min(ofs_y + self.opt['tile']['tile_size'], height)

                # input tile area on total image with padding
                input_start_x_pad = max(input_start_x - self.opt['tile']['tile_pad'], 0)
                input_end_x_pad = min(input_end_x + self.opt['tile']['tile_pad'], width)
                input_start_y_pad = max(input_start_y - self.opt['tile']['tile_pad'], 0)
                input_end_y_pad = min(input_end_y + self.opt['tile']['tile_pad'], height)

                # input tile dimensions
                input_tile_width = input_end_x - input_start_x
                input_tile_height = input_end_y - input_start_y
                tile_idx = y * tiles_x + x + 1
                input_tile = self.img[:, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]

                # upscale tile
                try:
                    if hasattr(self, 'net_g_ema'):
                        self.net_g_ema.eval()
                        with torch.no_grad():
                            output_tile = self.net_g_ema(input_tile)
                    else:
                        self.net_g.eval()
                        with torch.no_grad():
                            output_tile = self.net_g(input_tile)
                except RuntimeError as error:
                    print('Error', error)
                print(f'\tTile {tile_idx}/{tiles_x * tiles_y}')

                # output tile area on total image
                output_start_x = input_start_x * self.opt['scale']
                output_end_x = input_end_x * self.opt['scale']
                output_start_y = input_start_y * self.opt['scale']
                output_end_y = input_end_y * self.opt['scale']

                # output tile area without padding
                output_start_x_tile = (input_start_x - input_start_x_pad) * self.opt['scale']
                output_end_x_tile = output_start_x_tile + input_tile_width * self.opt['scale']
                output_start_y_tile = (input_start_y - input_start_y_pad) * self.opt['scale']
                output_end_y_tile = output_start_y_tile + input_tile_height * self.opt['scale']

                # put tile into output image
                self.output[:, :, output_start_y:output_end_y,
                            output_start_x:output_end_x] = output_tile[:, :, output_start_y_tile:output_end_y_tile,
                                                                       output_start_x_tile:output_end_x_tile]

    def post_process(self):
        _, _, h, w = self.output.size()
        self.output = self.output[:, :, 0:h - self.mod_pad_h * self.scale, 0:w - self.mod_pad_w * self.scale]

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        metric_rows = []
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)

            self.pre_process()
            if 'tile' in self.opt:
                self.tile_process()
            else:
                self.process()
            self.post_process()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']])
            metric_data['img'] = sr_img
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                metric_data['img2'] = gt_img
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["name"]}.png')
                os.makedirs(osp.dirname(save_img_path), exist_ok=True)
                imwrite(sr_img, save_img_path)

            if with_metrics:
                # calculate metrics
                row = {'image': img_name}
                for name, opt_ in self.opt['val']['metrics'].items():
                    value = calculate_metric(metric_data, opt_)
                    self.metric_results[name] += value
                    row[name] = float(value)
                metric_rows.append(row)
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)
            if self.opt['val'].get('save_metrics_csv', False):
                safe_dataset_name = re.sub(r'[^A-Za-z0-9_.-]+', '_', dataset_name)
                metrics_dir = osp.join(self.opt['path']['results_root'], 'metrics')
                os.makedirs(metrics_dir, exist_ok=True)
                metrics_path = osp.join(metrics_dir, f'{safe_dataset_name}.csv')
                with open(metrics_path, 'w', newline='', encoding='utf-8') as csv_file:
                    writer = csv.DictWriter(
                        csv_file,
                        fieldnames=['image', *self.opt['val']['metrics'].keys()],
                    )
                    writer.writeheader()
                    writer.writerows(metric_rows)
                get_root_logger().info('Saved per-image metrics to %s', metrics_path)
