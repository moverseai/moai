import torch
import functools



class Pck3D(torch.nn.Module):
    """
    Detected joint is considered correct if the distance between 
    the predicted and the ground truth joint is within a certain threshold (threshold varies)
    """
    def __init__(self,
        threshold: float=0.05
    ) -> None:

        super().__init__()
        self.threshold = threshold
    
    def forward(self,
        distance: torch.Tensor #distance between gt and predicted kpts
    ) -> torch.Tensor:

        pck =  100 * (distance < self.threshold).sum() / distance.shape[1]

        return pck


Pck3D_1 = functools.partial(Pck3D, threshold=0.01) #1 cm
Pck3D_3 = functools.partial(Pck3D, threshold=0.03) # 3 cm
Pck3D_7 = functools.partial(Pck3D, threshold=0.07) # 7 cm