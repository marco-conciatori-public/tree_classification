import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms.functional as tf


class Dataset_from_obs_targets(Dataset):
    def __init__(self, obs_list: list, target_list: list, name: str = None):
        assert len(obs_list) == len(target_list), 'ERROR: obs_list and' \
                   ' target_list must have the same number of elements.'

        self.obs_list = obs_list
        self.target_list = target_list
        if name is not None:
            self.name = name
        else:
            self.name = self.__class__.__name__

    def __len__(self):
        return len(self.obs_list)

    def __getitem__(self, idx):
        return tf.to_tensor(self.obs_list[idx]), self.target_list[idx]

    # def get_subset(self, idx_min: int = None, idx_max: int = None):
    #     if idx_min is None and idx_max is None:
    #         raise ValueError('ERROR: "idx_min" and "idx_max" cannot both be None.')
    #     if idx_min is None:
    #         idx_min = 0
    #     if idx_max is None:
    #         idx_max = len(self)
    #     return Dataset_from_obs_targets(
    #         obs_list=self.obs_list[idx_min:idx_max],
    #         target_list=self.target_list[idx_min:idx_max],
    #         name=self.name + '_subset',
    #     )
    #
    def random_split(self, lengths: list, generator=None) -> list:
        assert sum(lengths) == len(self), f'ERROR: sum(lengths) ({sum(lengths)}) must equal len(self) ({len(self)}).'
        ds_list = []
        if generator is not None:
            # TODO: implement generator
            raise NotImplementedError('ERROR: "generator" argument of "random_split" method is not yet implemented.')
        else:
            sequence = np.arange(len(self))
            np.random.shuffle(sequence)
            base = 0
            for length in lengths:
                assert length > 0, 'ERROR: all elements of "lengths" must be greater than 0.'
                temp_obs_list = []
                temp_target_list = []
                for i in range(base, base + length):
                    temp_obs_list.append(self.obs_list[sequence[i]])
                    temp_target_list.append(self.target_list[sequence[i]])

                temp_ds = Dataset_from_obs_targets(
                    obs_list=temp_obs_list,
                    target_list=temp_target_list,
                    name=self.name + f'_random_split_{length}',
                )
                ds_list.append(temp_ds)
                base += length

        return ds_list
