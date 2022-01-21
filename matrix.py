import torch.distributed as dist
import torch.utils.data.distributed
import torch
import torch.nn.functional as F
from IPython import embed

# class F(torch.autograd.Function):
    
#     @staticmethod
#     def forward(ctx, input):
#         return input

#     @staticmethod
#     def backward(ctx, grad_output):
#         dist.all_reduce(grad_output)

class f(torch.autograd.Function): 
    @staticmethod
    def forward(ctx, x): 
        return x 
    @staticmethod
    def backward(ctx, gradient): 
        dist.all_reduce(gradient) 
        return gradient

class g(torch.autograd.Function): 
    @staticmethod
    def forward(ctx, x): 
        dist.all_reduce(x) 
        return x 
    @staticmethod
    def backward(ctx, gradient): 
        return gradient

class ParalleledMatrixMulply(torch.nn.Module):
    def __init__(self, size1, size2, device):
        super().__init__()
        self.size1 = size1
        self.size2 = size2
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.bias = torch.nn.Parameter(torch.zeros(size1, size2))
        self.weight = torch.nn.Parameter(torch.rand(size1 // self.world_size, size2))
        self.f = f.apply
        self.g = g.apply

    def forward(self, X):
        # col-wize parallel
        # print(A.size(1) // self.world_size * self.rank, A.size(1) // self.world_size * (self.rank + 1))
        weight = self.f(self.weight)
        # B = self.f(B)
        X = X[:, X.size(1) // self.world_size * self.rank : X.size(1) // self.world_size * (self.rank + 1)]
        Y = torch.matmul(X, weight)
        Y = self.g(Y).clone()
        Y += self.bias
        return Y
