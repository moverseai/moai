from moai.utils.arguments import ensure_string_list

import torch
import glob
import os
import typing
import logging
import toolz
import trimesh

__all__ = ["StructuredGeometry"]

log = logging.getLogger(__name__)

class StructuredGeometry(torch.utils.data.Dataset):
    def __init__(self,
        root:           str='',
        **kwargs:       typing.Mapping[str, typing.Mapping[str, typing.Any]],
    ):
        self.key_to_list = {}
        self.key_to_params = {}
        self.key_to_loader = {}
        for k, m in kwargs.items():
            glob_list = ensure_string_list(m['glob'])
            files = []
            for g in glob_list:
                files += sorted(glob.glob(os.path.join(root, g)))
            self.key_to_list[k] = list(map(lambda f: os.path.join(root, f), files))
            self.key_to_params[k] = toolz.dissoc(m, 'type', 'glob')
        log.info(f"Loaded {','.join(map(lambda kv: str(len(kv[1])) + ' ' + kv[0], self.key_to_list.items()))} items.")

    def __len__(self) -> int:
        return len(next(iter(self.key_to_list.values())))

    def __getitem__(self, index: int) -> typing.Dict[str, torch.Tensor]:
        ret = { }
        for k, l in self.key_to_list.items():
            geometry = trimesh.load(l[index], **self.key_to_params[k], process=False)
            ret[k] = {
                'vertices': torch.from_numpy(geometry.vertices).float(),
            }
            if hasattr(geometry, 'faces'):
                ret[k]['faces'] = torch.from_numpy(geometry.faces).long()
        return ret