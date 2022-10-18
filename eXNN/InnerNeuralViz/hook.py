import torch.nn as nn

class Hook:
    def __init__(self, m:nn.Module):
        self.module = m
        self.fwd = None
        self.bwd = None
        def fwd_hook(m,i,o): self.fwd = o
        def bwd_hook(m,i,o): self.bwd = o[0]
        self.module.register_forward_hook(fwd_hook)
        self.module.register_backward_hook(bwd_hook)
    def show(self):
        print('> fwd'); print_shapes(self.fwd)
        print('> bwd'); print_shapes(self.bwd)
    def clear(self):
        self.fwd = None
        self.bwd = None

def _get_module_by_name(model, name):
    for n,m in model.named_modules(): 
        if n == name: return m
    raise Exception(f'Model does not contain submodule {name}')

def get_hook(model, layer_name) -> Hook:
    layer = _get_module_by_name(model, layer_name)
    return Hook(layer)