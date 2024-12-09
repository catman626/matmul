#!/usr/bin/python3
# coding=utf-8
#
# Copyright (C) 2023-2024. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================

import numpy as np
import os


def gen_golden_data():
    np.random.seed(626)
    M = 512
    N = 1024
    K = 512

    x1_gm = np.random.randint(1, 10, [M, K]).astype(np.float16)
    x2_gm = np.random.randint(1, 10, [K, N]).astype(np.float16)
    bias_gm = np.random.randint(1, 10, [1, N]).astype(np.float32)

    golden = np.matmul(x1_gm.astype(np.float32), x2_gm.astype(np.float32)).astype(np.float32)
    golden = np.add(golden , bias_gm)

    os.system("mkdir -p input")
    os.system("mkdir -p output")

    x1_gm.tofile("./input/x1_gm.bin")
    x2_gm.tofile("./input/x2_gm.bin")
    bias_gm.tofile("./input/bias_gm.bin")

    golden.tofile("./output/golden.bin")


if __name__ == "__main__":
    gen_golden_data()
