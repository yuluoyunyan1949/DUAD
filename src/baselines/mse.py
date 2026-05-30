import torch
import numpy as np

class MSE:
    def __init__(self):
        self.threshold = 0.0

    def fit(self, model, loader):
        mses = []
        model.eval()
        with torch.no_grad():
            for x, _ in loader:
                x = x.cuda()
                rec, _, _ = model(x)
                err = torch.mean((rec - x)**2, dim=(1,2,3))
                mses.append(err.cpu())
        mses = torch.cat(mses).numpy()
        self.threshold = np.percentile(mses, 95)