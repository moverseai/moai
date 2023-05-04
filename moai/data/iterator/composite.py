import torch
import typing
from moai.data.iterator import Indexed
from moai.data.iterator import Zipped
from moai.data.iterator import Concatenated
from moai.data.iterator import Interleaved
from moai.data.iterator import Repeated
from moai.data.iterator import Windowed
from collections import defaultdict
from moai.data.iterator.zip import Zipper


class Composited(torch.utils.data.Dataset):
    # create a map between the iterator value and a python class
    # that will be used to create the iterator
    _iterators = {
        'indexed': Indexed,
        'window': Windowed,
        'zipped': Zipped,
        'repeat': Repeated,
        'interleave': Interleaved,
        'concat': Concatenated,
    }

    r"""
    A composite iterator that zips multiple iterators & datasets together.
    """

    def __init__(
        self,
        iterators: typing.Sequence[torch.utils.data.Dataset],
        datasets: typing.Sequence[torch.utils.data.Dataset],
    ):
        r"""
        Initializes a composite iterator.

        Args:
            iterators (list): a list of iterators to zip together.
        """
        super(Composited, self).__init__()
        # w = Windowed({'vrs': datasets['vrs']}, **iterators['window'])
        d = defaultdict()
        for iterator, dataset in zip(iterators,datasets):
            # select the iterator class based on the iterator name
            # and create the iterator
            iter_ = self._iterators[iterator]({dataset: datasets[dataset]}, **iterators[iterator])
            # add the iterator to the list of iterators
            d[iterator] = iter_
            
        # self.dataset = Zipped(iterators)
        self.dataset = Zipper([d_ for d_ in d.values()])

    
    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        r"""
        Gets the next batch of samples from the composite iterator.

        Args:
            index (int): the index of the batch.

        Returns:
            dict: a dictionary of samples from the composite iterator.
        """
        item = self.dataset[index]

        return item