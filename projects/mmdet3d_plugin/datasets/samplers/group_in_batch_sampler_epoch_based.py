# https://github.com/Divadi/SOLOFusion/blob/main/mmdet3d/datasets/samplers/infinite_group_each_sample_in_batch_sampler.py
import itertools
import copy

import numpy as np
import torch
import torch.distributed as dist
from mmcv.runner import get_dist_info
from torch.utils.data.sampler import Sampler


# https://github.com/open-mmlab/mmdetection/blob/3b72b12fe9b14de906d1363982b9fba05e7d47c1/mmdet/core/utils/dist_utils.py#L157
def sync_random_seed(seed=None, device="cuda"):
    """Make sure different ranks share the same seed.
    All workers must call this function, otherwise it will deadlock.
    This method is generally used in `DistributedSampler`,
    because the seed should be identical across all processes
    in the distributed group.
    In distributed sampling, different ranks should sample non-overlapped
    data in the dataset. Therefore, this function is used to make sure that
    each rank shuffles the data indices in the same order based
    on the same seed. Then different ranks could use different indices
    to select non-overlapped data from the same data list.
    Args:
        seed (int, Optional): The seed. Default to None.
        device (str): The device where the seed will be put on.
            Default to 'cuda'.
    Returns:
        int: Seed to be used.
    """
    if seed is None:
        seed = np.random.randint(2**31)
    assert isinstance(seed, int)

    rank, world_size = get_dist_info()

    if world_size == 1:
        return seed

    if rank == 0:
        random_num = torch.tensor(seed, dtype=torch.int32, device=device)
    else:
        random_num = torch.tensor(0, dtype=torch.int32, device=device)
    dist.broadcast(random_num, src=0)
    return random_num.item()


class GroupInBatchSamplerEpochBased(Sampler):
    """
    Pardon this horrendous name. Basically, we want every sample to be from its own group.
    If batch size is 4 and # of GPUs is 8, each sample of these 32 should be operating on
    its own group.

    Shuffling is only done for group order, not done within groups.
    """

    def __init__(
        self,
        dataset,
        batch_size=1,
        world_size=None,
        rank=None,
        seed=0,
        sequence_flip_prob=0.,
        drop_last=True,
        shuffle=True,
    ):
        _rank, _world_size = get_dist_info()
        if world_size is None:
            world_size = _world_size
        if rank is None:
            rank = _rank

        self.dataset = dataset
        self.batch_size = batch_size
        self.world_size = world_size
        self.rank = rank
        self.seed = sync_random_seed(seed)

        self.size = len(self.dataset)

        assert hasattr(self.dataset, "flag")
        self.flag = self.dataset.flag
        self.group_sizes = np.bincount(self.flag)
        self.groups_num = len(self.group_sizes)
        self.global_batch_size = batch_size * world_size
        assert self.groups_num >= self.global_batch_size

        # Now, for efficiency, make a dict group_idx: List[dataset sample_idxs]
        self.group_idx_to_sample_idxs = {
            group_idx: np.where(self.flag == group_idx)[0].tolist()
            for group_idx in range(self.groups_num)
        }

        self.epoch = 0
        self.shuffle = shuffle
        self.sequence_flip_prob = sequence_flip_prob
        self.drop_last = drop_last

        self.sampler = self
        self._init_epoch()

    def _init_epoch(self):
        # Get a list of group indices per sample idx. Considering samples over all
        # GPUs, each sample position has its own list

        group_indices = self._generate_group_indices()
        group_indices_per_global_sample_idx = []
        group_indices_len_per_global_sample_idx = []
        for global_sample_idx in range(self.global_batch_size):
            group_indices_per_global_sample_idx_i = [group_indices[i]
                                                     for i in range(global_sample_idx, self.groups_num, self.global_batch_size)]
            group_indices_per_global_sample_idx.append(group_indices_per_global_sample_idx_i)
            sum = 0
            for group_idx in group_indices_per_global_sample_idx_i:
                sum += len(self.group_idx_to_sample_idxs[group_idx])
            group_indices_len_per_global_sample_idx.append(sum)

        # local GPU
        self.group_indices_per_local_sample_idx = []
        for local_sample_idx in range(self.batch_size):
                self.group_indices_per_local_sample_idx.append(
                    group_indices_per_global_sample_idx[self.rank * self.batch_size + local_sample_idx]
                )

        if self.drop_last:
            self.counter = min(group_indices_len_per_global_sample_idx)

        # Keep track of a buffer of dataset sample idxs for each local sample idx
        self.buffer_per_local_sample = [[] for _ in range(self.batch_size)]
        self.aug_per_local_sample = [None for _ in range(self.batch_size)]

    def _generate_group_indices(self):
        g = torch.Generator()
        if self.shuffle:
            g.manual_seed(self.seed + self.epoch)
        else:
            g.manual_seed(self.seed)  # order stay the same in different epochs, e.g. for val
        return torch.randperm(self.groups_num, generator=g).tolist()
        # return torch.arange(self.groups_num).tolist() # Note: activate to not shuffle

    def __iter__(self):
        while True:
            if self.drop_last:
                if self.counter == 0:
                    return
                self.counter -= 1

            curr_batch = []
            for local_sample_idx in range(self.batch_size):
                if len(self.buffer_per_local_sample[local_sample_idx]) == 0:
                    # Finished current group, refill with next group
                    try:
                        new_group_idx = self.group_indices_per_local_sample_idx[local_sample_idx].pop(0)
                    except (StopIteration, IndexError):
                        assert not self.drop_last
                        continue
                    self.buffer_per_local_sample[
                        local_sample_idx
                    ] = copy.deepcopy(
                        self.group_idx_to_sample_idxs[new_group_idx]
                    )
                    if np.random.uniform() < self.sequence_flip_prob:
                        self.buffer_per_local_sample[
                            local_sample_idx
                        ] = self.buffer_per_local_sample[local_sample_idx][
                            ::-1
                        ]
                    if self.dataset.keep_consistent_seq_aug:
                        self.aug_per_local_sample[
                            local_sample_idx
                        ] = self.dataset.get_augmentation()

                if not self.dataset.keep_consistent_seq_aug:
                    self.aug_per_local_sample[
                        local_sample_idx
                    ] = self.dataset.get_augmentation()

                curr_batch.append(
                    dict(
                        idx=self.buffer_per_local_sample[local_sample_idx].pop(0),
                        aug_config=self.aug_per_local_sample[local_sample_idx],
                    )
                )
            if len(curr_batch) == 0:
                assert not self.drop_last
                return
            else:
                yield curr_batch

    def __len__(self):
        """Length of base dataset."""
        if self.drop_last:
            return self.size // self.global_batch_size
        else:
            return (self.size + self.global_batch_size - 1) // self.global_batch_size

    def set_epoch(self, epoch):
        self.epoch = epoch
        self._init_epoch()

