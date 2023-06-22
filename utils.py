import torch
from pathlib import Path


def get_available_id(partial_name: str, folder_path: str) -> int:
    pure_path = Path(folder_path)
    if pure_path.exists():
        matching_paths = pure_path.glob(f'{partial_name}*')
        current_ids = set()
        for path in matching_paths:
            # also remove the separator character between the model name and model id
            path_name_removed = path.name[len(partial_name) + 1:]
            last_id = ''
            counter = 0
            while path_name_removed[counter].isdigit():
                counter += 1
            last_id += path_name_removed[:counter]
            last_id = int(last_id)
            current_ids.add(last_id)

        if len(current_ids) == 0:
            return 0
        max_id = max(current_ids)
        # if there are no holes in the serial number we want to have in the complete_set the next
        # biggest free number. This requires +2 instead of +1 because range() exclude the right boundary.
        complete_set = set(range(0, max_id + 2))
        difference = complete_set - current_ids
        min_free_id = min(difference)
        return min_free_id

    return 0


def get_available_device(verbose: int = 0) -> torch.device:
    if not torch.cuda.is_available():
        print('WARNING: GPU not found, using CPU.')
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0')

    if verbose >= 1:
        print(f'Device: {device}.')

    return device


def check_split_proportions(train_val_test_proportions: list, tolerance: float):
    # check that proportions adds up to 1, except for rounding errors
    assert 1 - tolerance < sum(train_val_test_proportions) < 1 + tolerance, \
        f'The values of train_val_test_proportions must add up to 1 +/- {tolerance}.' \
        f' They add up to {sum(train_val_test_proportions)}.'
