import torch.nn.functional as F

softmax_helper = lambda x: F.softmax(x, 1) ## 0是对列做归一化，1是对行做归一化