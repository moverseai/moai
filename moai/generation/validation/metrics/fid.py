import torch

class FID(torch.nn.Module):
    def __init__(self):
        super(FID, self).__init__()

    def forward(self,
        gt:         torch.Tensor,
        pred:       torch.Tensor,
        weights:    torch.Tensor=None,
    ) -> torch.Tensor:
        num_samples = pred.shape[0] if pred.shape[0] <= gt.shape[0] else gt.shape[0] 
        mu1, mu2= torch.mean(gt[:num_samples], dim=0), torch.mean(pred[:num_samples], dim=0)
        c1, c2 = torch.cov(gt[:num_samples]), torch.cov(pred[:num_samples])
        diff_sq = torch.norm(mu1 - mu2, p=2, dim=-1, keepdim=False)
        fid = diff_sq + torch.trace(c1 + c2 - 2*torch.sqrt(c1*c2))
        if weights is not None:
            fid = fid * weights
        return fid