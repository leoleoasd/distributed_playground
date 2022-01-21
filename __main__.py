import argparse
from .matrix import ParalleledMatrixMulply
from IPython import embed
import torch
import torch.distributed as dist
import pytorch_lightning as pl

pl.seed_everything(1234)

parser = argparse.ArgumentParser(description='PyTorch distributed training on cifar-10')
parser.add_argument('--rank', default=0, type=int,
                    help='rank of current process')
parser.add_argument('--word_size', default=2, type=int,
                    help="word size")
parser.add_argument('--init_method', default='tcp://127.0.0.1:23456',
                    help="init-method")
args = parser.parse_args()

dist.init_process_group(backend='nccl', init_method=args.init_method, rank=args.rank, world_size=args.word_size)

net = ParalleledMatrixMulply(4,4)
# net = net.cuda()
device = torch.device('cuda', args.rank)
net = net.to(device)
net.train()
# net = torch.nn.parallel.DistributedDataParallel(net)

A = torch.rand(4, 4).to(device)
B = torch.rand(4, 4).to(device)
C = net(A, B)
print(C - torch.matmul(A, B))

embed()