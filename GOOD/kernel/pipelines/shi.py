
from collections import defaultdict
import torch
x=torch.tensor([[1,4,4],[2,2,8]])
print(x[1][x[0]==4])
t=x[1][x[0]==4]
for a in t:
    print(a)