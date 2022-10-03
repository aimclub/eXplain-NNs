import torch

def BasicBayesianPrediction(data: torch.Tensor, 
                            model: torch.nn.Module,
                            n_iter: int,
                            mode: str,
                            p: float,
                            a: float,
                            b: float):
    raise NotImplementedError()

class BasisBayesianWrapper:
    def __init__(self,
                 model: torch.nn.Module,
                 mode: str,
                 p: float,
                 a: float,
                 b: float):
        self.model = model
        self.mode = mode
        self.p = p
        self.a = a
        self.b = b

    def predict(self, data, n_iter):
        return BasicBayesianPrediction(data,
                                       self.model,
                                       n_iter,
                                       self.mode,
                                       self.p,
                                       self.a,
                                       self.b)

def create_bayesian_wrapper(model: torch.nn.Module) -> torch.nn.Module:
    raise NotImplementedError()