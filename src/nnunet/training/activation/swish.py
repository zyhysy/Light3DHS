
from torch import nn, Tensor
from typing import Optional, Tuple

class Swish(nn.SiLU):

    def __init__(self, inplace: Optional[bool] = False) -> None:
        super().__init__(inplace=inplace)

    def profile_module(self, input: Tensor) -> Tuple[Tensor, float, float]:
        return input, 0.0, 0.0
