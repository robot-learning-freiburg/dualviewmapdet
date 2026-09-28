# Copyright (c) 2026 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

# This source code is derived from SparseDrive
# Copyright (c) 2024 Horizon Robotics, licensed under the MIT license,
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.
import torch
import torch.nn as nn
import numpy as np
from PIL import Image


class Grid(object):
    def __init__(
        self, use_h, use_w, rotate=1, offset=False, ratio=0.5, mode=0, prob=1.0
    ):
        self.use_h = use_h
        self.use_w = use_w
        self.rotate = rotate
        self.offset = offset
        self.ratio = ratio
        self.mode = mode
        self.st_prob = prob
        self.prob = prob

    def set_prob(self, epoch, max_epoch):
        self.prob = self.st_prob * epoch / max_epoch

    def __call__(self, img, label):
        if np.random.rand() > self.prob:
            return img, label
        h = img.size(1)
        w = img.size(2)
        self.d1 = 2
        self.d2 = min(h, w)
        hh = int(1.5 * h)
        ww = int(1.5 * w)
        d = np.random.randint(self.d1, self.d2)
        if self.ratio == 1:
            self.l = np.random.randint(1, d)
        else:
            self.l = min(max(int(d * self.ratio + 0.5), 1), d - 1)
        mask = np.ones((hh, ww), np.float32)
        st_h = np.random.randint(d)
        st_w = np.random.randint(d)
        if self.use_h:
            for i in range(hh // d):
                s = d * i + st_h
                t = min(s + self.l, hh)
                mask[s:t, :] *= 0
        if self.use_w:
            for i in range(ww // d):
                s = d * i + st_w
                t = min(s + self.l, ww)
                mask[:, s:t] *= 0

        r = np.random.randint(self.rotate)
        mask = Image.fromarray(np.uint8(mask))
        mask = mask.rotate(r)
        mask = np.asarray(mask)
        mask = mask[
            (hh - h) // 2 : (hh - h) // 2 + h,
            (ww - w) // 2 : (ww - w) // 2 + w,
        ]

        mask = torch.from_numpy(mask).float()
        if self.mode == 1:
            mask = 1 - mask

        mask = mask.expand_as(img)
        if self.offset:
            offset = torch.from_numpy(2 * (np.random.rand(h, w) - 0.5)).float()
            offset = (1 - mask) * offset
            img = img * mask + offset
        else:
            img = img * mask

        return img, label


class GridMask(nn.Module):
    def __init__(
        self,
        use_h,
        use_w,
        rotate=1,
        offset=False,
        ratio=0.5,
        mode=0,
        prob=1.0,
        mask_channel_index: int | None = None,  # e.g. -1 if last channel is mask; None => old behavior
    ):
        super(GridMask, self).__init__()
        self.use_h = use_h
        self.use_w = use_w
        self.rotate = rotate
        self.offset = offset
        self.ratio = ratio
        self.mode = mode
        self.st_prob = prob
        self.prob = prob
        self.mask_channel_index = mask_channel_index

    def set_prob(self, epoch, max_epoch):
        self.prob = self.st_prob * epoch / max_epoch  # + 1.#0.5

    def _build_grid_mask(self, h: int, w: int, device, dtype):
        """Returns grid keep-mask of shape (1, h, w) with values in {0,1}."""
        hh = int(1.5 * h)
        ww = int(1.5 * w)

        d = np.random.randint(2, h)  # original behavior
        l = min(max(int(d * self.ratio + 0.5), 1), d - 1)

        mask = np.ones((hh, ww), np.float32)
        st_h = np.random.randint(d)
        st_w = np.random.randint(d)
        if self.use_h:
            for i in range(hh // d):
                s = d * i + st_h
                t = min(s + l, hh)
                mask[s:t, :] *= 0
        if self.use_w:
            for i in range(ww // d):
                s = d * i + st_w
                t = min(s + l, ww)
                mask[:, s:t] *= 0

        r = np.random.randint(self.rotate)
        mask = Image.fromarray(np.uint8(mask))
        mask = mask.rotate(r)
        mask = np.asarray(mask)
        mask = mask[
            (hh - h) // 2 : (hh - h) // 2 + h,
            (ww - w) // 2 : (ww - w) // 2 + w,
        ]

        grid = torch.from_numpy(mask.copy()).to(device=device, dtype=dtype)  # (h,w)
        if self.mode == 1:
            grid = 1 - grid  # invert keep/drop

        return grid.unsqueeze(0)  # (1,h,w)

    def forward(self, x):
        if np.random.rand() > self.prob or not self.training:
            return x

        n, c, h, w = x.size()
        x_ = x.view(-1, c, h, w)  # keep channels explicit

        device = x_.device
        dtype = x_.dtype

        grid = self._build_grid_mask(h, w, device=device, dtype=dtype)  # (1,h,w)
        grid_hw = grid.unsqueeze(1)  # (1,1,h,w) for broadcasting

        # Default: old behavior (mask applied to all channels)
        if self.mask_channel_index is None:
            if self.offset:
                offset = torch.from_numpy(2 * (np.random.rand(h, w) - 0.5)).to(device=device, dtype=dtype)
                offset = offset.view(1, 1, h, w)
                x_ = x_ * grid_hw + offset * (1 - grid_hw)
            else:
                x_ = x_ * grid_hw
            return x_.view(n, c, h, w)

        # NEW behavior: treat one channel as "mask channel"
        mi = self.mask_channel_index
        if mi < 0:
            mi = c + mi
        if not (0 <= mi < c):
            raise ValueError(f"mask_channel_index out of range: got {self.mask_channel_index} for C={c}")

        # Split
        old_mask = x_[:, mi : mi + 1]  # (B,1,H,W)
        # Everything except mask channel
        if mi == 0:
            data = x_[:, 1:]
            left, right = None, data
        elif mi == c - 1:
            data = x_[:, :c - 1]
            left, right = data, None
        else:
            left = x_[:, :mi]
            right = x_[:, mi + 1:]
            data = torch.cat([left, right], dim=1)

        # Apply grid to data channels
        if self.offset:
            offset = torch.from_numpy(2 * (np.random.rand(h, w) - 0.5)).to(device=device, dtype=dtype)
            offset = offset.view(1, 1, h, w)
            data_out = data * grid_hw + offset * (1 - grid_hw)
        else:
            data_out = data * grid_hw

        # Combine masks: original lidar-validity mask AND grid keep mask
        # (Assumes old_mask is already 0/1-ish; works fine for floats too.)
        new_mask = old_mask * grid_hw

        # Reassemble in original channel order
        if mi == 0:
            out = torch.cat([new_mask, data_out], dim=1)
        elif mi == c - 1:
            out = torch.cat([data_out, new_mask], dim=1)
        else:
            out = torch.cat([data_out[:, :mi], new_mask, data_out[:, mi:]], dim=1)

        return out.view(n, c, h, w)

