import torch.distributed as dist
import torch.utils.data.distributed
import torch
import torch.nn.functional as F

# class F(torch.autograd.Function):
    
#     @staticmethod
#     def forward(ctx, input):
#         return input

#     @staticmethod
#     def backward(ctx, grad_output):
#         dist.all_reduce(grad_output)


class ParalleledMatrixMulply(torch.nn.Module):

    def __init__(self, size1, size2):
        super().__init__()
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.bias = torch.nn.Parameter(torch.zeros(size1, size2))
        
    def forward(self, A, B):
        # col-wize parallel
        print(A.size(1) // self.world_size * self.rank, A.size(1) // self.world_size * (self.rank + 1))
        A = A[:, A.size(1) // self.world_size * self.rank : A.size(1) // self.world_size * (self.rank + 1)]
        # row-wize parallel
        B = B[B.size(0) // self.world_size * self.rank : B.size(0) // self.world_size * (self.rank + 1), :]
        Y = torch.matmul(A, B)
        dist.all_reduce(Y)
        Y += self.bias
        return Y

    # def backward(ctx, grad_output):
    #     """
    #     In the backward pass we receive a Tensor containing the gradient of the loss
    #     with respect to the output, and we need to compute the gradient of the loss
    #     with respect to the input.
    #     """
    #     input, = ctx.saved_tensors
    #     grad_input = grad_output.clone()
    #     grad_input[input < 0] = 0
    #     return grad_input
