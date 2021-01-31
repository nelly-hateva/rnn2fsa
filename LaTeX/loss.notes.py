import math

import torch
from torch import nn

output = torch.tensor([[0.2784, 0.0618],
                       [0.3401, -0.0615],
                       [0.3401, -0.0615]])
target = torch.tensor([0, 1, 1])
print(output)
print(target)
print(nn.CrossEntropyLoss()(output, target))
print(nn.CrossEntropyLoss(reduction='sum')(output, target))
print(nn.CrossEntropyLoss(reduction='none')(output, target))

print("LogSoftMax", nn.LogSoftmax(dim=1)(output))

lsfm = math.log(math.exp(0.2784) / (math.exp(0.2784) + math.exp(0.0618)))
print("log(e^0.2784 / (e^0.2784 + e^0.0618))", lsfm)
print("log(e^0.0618 / (e^0.2784 + e^0.0618))", math.log(math.exp(0.0618) / (math.exp(0.2784) + math.exp(0.0618))))

lsfm2 = math.log(math.exp(-0.0615) / (math.exp(0.3401) + math.exp(-0.0615)))
print("log(e^0.3401 / (e^0.3401 + e^-0.0615))", math.log(math.exp(0.3401) / (math.exp(0.3401) + math.exp(-0.0615))))
print("log(e^-0.0615 / (e^0.3401 + e^-0.0615))", lsfm2)

print("LLLOOS ", -(lsfm + lsfm2 + lsfm2) / 3.)
print("LLLOOS SUM", -(lsfm + lsfm2 + lsfm2))
print("LLLOOS NONE", -(lsfm + lsfm2 + lsfm2))
