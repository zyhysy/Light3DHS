#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

from torch import nn, Tensor
from typing import Optional, Tuple


class PReLU(nn.PReLU):

    def __init__(
        self, num_parameters: Optional[int] = 1, init: Optional[float] = 0.25
    ) -> None:
        super().__init__(num_parameters=num_parameters, init=init)

    def profile_module(self, input: Tensor) -> Tuple[Tensor, float, float]:
        return input, 0.0, 0.0
