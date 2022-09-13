import torchvision
import torch
import hydra.utils as hyu
import omegaconf.omegaconf
import typing
import logging

log = logging.getLogger(__name__)

__all__ = ["Combine"]

class Combine(torch.utils.data.Dataset):
    def __init__(self,
        dataset:        torch.utils.data.Dataset,        
        inputs:         typing.Sequence[str],
        outputs:        typing.Sequence[str],
        transforms:     omegaconf.DictConfig={},
    ):
        super(Combine, self).__init__()
        self.inner, self.inputs, self.outputs = dataset, inputs, outputs
        self.transforms = []
        log.warning("Using an ordered composition of the following torchvision transforms:")
        for i, (k, v) in enumerate(transforms.items()):
            xform = hyu.instantiate(v)
            log.info(f"\t\t\t{i + 1}. A `{k}` transform.")
            self.transforms.append(xform)
        self.composition = torchvision.transforms.Compose(self.transforms)

    def __len__(self) -> int:
        return len(self.inner)

    def __getitem__(self, index: int) -> typing.Dict[str, torch.Tensor]:
        #NOTE: tensor/PIL image conversions are expected to be handled via the transforms order
        item = self.inner[index]
        for i, o in zip(self.inputs, self.outputs):            
            item[o] = self.composition(item[i])
        return item