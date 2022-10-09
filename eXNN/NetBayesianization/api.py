import torch
from .wrap import create_bayesian_wrapper

def BasicBayesianPrediction(data: torch.Tensor, 
                            model: torch.nn.Module,
                            n_iter: int,
                            mode: str,
                            p: float,
                            a: float,
                            b: float):
    return BasicBayesianWrapper(model, mode, p, a, b).predict(data, n_iter)

class BasicBayesianWrapper:
    def __init__(self,
                 model: torch.nn.Module,
                 mode: str,
                 p: float,
                 a: float,
                 b: float):
        self.model = create_bayesian_wrapper(model, mode, p, a, b)

    def predict(self, data, n_iter):
        all_probs = []
        with torch.no_grad():
            for i in range(n_iter):
                prob = self.model(data).softmax(dim=1)
                all_probs.append(prob)
        all_probs = torch.stack(all_probs, dim=0)
        all_preds = all_probs.argmax(dim=-1)
        mean_preds = all_preds.mode(dim=0)[0]
        return mean_preds