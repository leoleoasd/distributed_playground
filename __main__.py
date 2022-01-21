import argparse
from matrix import ParalleledMatrixMulply
from IPython import embed
import torch
import torch.distributed as dist
import pytorch_lightning as pl
from tqdm import tqdm

pl.seed_everything(1234)

parser = argparse.ArgumentParser(description='PyTorch distributed training on cifar-10')
parser.add_argument('--rank', default=0, type=int,
                    help='rank of current process')
parser.add_argument('--world_size', default=2, type=int,
                    help="word size")
parser.add_argument('--init_method', default='tcp://127.0.0.1:23456',
                    help="init-method")
args = parser.parse_args()

dist.init_process_group(backend='nccl', init_method=args.init_method, rank=args.rank, world_size=args.world_size)

device = torch.device('cuda', args.rank)
size = 15000
net = ParalleledMatrixMulply(size, size, device)
# net = net.cuda()
net = net.to(device)


optim = torch.optim.Adam(net.parameters(), 1)
for i in tqdm(range(10)):
    optim.zero_grad()
    A = torch.rand(size, size).to(device)
    C = net(A)
    C = torch.sum(C, 1)
    # C = torch.softmax(C, 0)
    loss = C[0]
    loss.backward()
    optim.step()
    print(f"{loss=}")
# print(C - torch.matmul(A, B))

embed()


# world_size=2: 9203MiB
# world_size=1: 13559MiB
