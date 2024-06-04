import logging

# from ts.torch_handler.image_classifier import ImageClassifier
from collections.abc import Callable

import torch
from torch.profiler import ProfilerActivity
from torchvision import transforms

log = logging.getLogger(__name__)


class MNISTDigitClassifier(Callable):
    """
    MNISTDigitClassifier handler class. This handler extends class ImageClassifier from image_classifier.py, a
    default handler. This handler takes an image and returns the number in that image.

    Here method postprocess() has been overridden while others are reused from parent class.
    """

    image_processing = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    def __init__(self, input_key: str = "predicted_label"):
        super(MNISTDigitClassifier, self).__init__()
        self.profiler_args = {
            "activities": [ProfilerActivity.CPU],
            "record_shapes": True,
        }
        self.input_key = input_key

    # def postprocess(self, data):
    #     """The post process of MNIST converts the predicted output response to a label.

    #     Args:
    #         data (list): The predicted output from the Inference with probabilities is passed
    #         to the post-process function
    #     Returns:
    #         list : A list of dictionaries with predictions and explanations is returned
    #     """
    #     return data.argmax(1).tolist()

    def __call__(self, data, device):
        """
        Transform input data into a tensor

        Args:
            data (list): List of raw input data
            device (torch.device): Device where the data is loaded

        Returns:
            torch.Tensor: Tensor of preprocessed data
        """
        return [{"class": int(data[self.input_key])}]
