#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2016-2022                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

from __future__ import annotations # MG20220819: Necessary list[float] type hints in python < 3.10

import random

################################################################################

class Noise():
    def __init__(self,
            params: dict = {}) -> None:

        noise_type = params.get("type", "no")
        if (noise_type == "no"):
            self.add_noise = self.add_noise_no
        elif (noise_type == "normal"):
            self.add_noise = self.add_noise_normal
            self.avg = params.get("avg", 0.)
            self.std = params.get("stdev", 0.)
        else:
            assert (0), "noise type must be \"no\" or \"normal\". Aborting."

    def add_noise_no(self,
            I: list[float]) -> None:
        pass

    def add_noise_normal(self,
            I: list[float]) -> None:
        I[0] += random.normalvariate(self.avg, self.std)
