# -*- coding: utf-8 -*-
import bisect

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def obj_fun(
        real: NDArray,
        detect: NDArray,
        tol: float,
        beat_type: list,
        bisected: bool = True
):
    """目标函数，输出混淆矩阵和匹配成功的特征点与实际位置的偏移值。
    Match the predicted R-peak positions based on the actual R-peak positions.
    :param reals: The predicted R-peak positions.
    :param detect: The actual R-peak positions.
    :param beat_type: The beats type.
    return: Matching the predicted R-peak positions match_beat_type, conf_matrix and np.array(diff).
    """
    if bisected:
        _a, _b = int(real[0]), int(real[-1])
        left_index = bisect.bisect_left(list(detect), _a - tol)
        right_index = bisect.bisect_right(list(detect), _b + tol)
        detect = detect[left_index:right_index]
    dist_matrix = np.abs(real[:, np.newaxis] - detect, dtype=float)  # row is real point index
    # print(len(real))
    # print(len(detect))
    # print(len(beat_type))
    real_len = len(real)
    detect_len = len(detect)
    if detect_len == 0:
        return np.array([
            [0, real_len],
            [detect_len, np.nan],
        ]), []
    flag = np.zeros_like(real, dtype=bool)
    diff = []
    match_beat_type = [None] * len(detect)
    for i in range(real_len):
        idx = np.argmin(dist_matrix[i, :])
        dist = dist_matrix[i, idx]
        if dist <= tol:
            dist_matrix[:, idx] = np.full_like(dist_matrix[:, idx], np.inf)
            flag[i] = True
            diff.append(detect[idx] - real[i])
            match_beat_type[idx] = beat_type[i]
    vaild = np.sum(flag)
    conf_matrix = np.array([
        [vaild, real_len - vaild],
        [detect_len - vaild, np.nan],
    ])
    return match_beat_type, conf_matrix, np.array(diff)


def summary(matrix, diff, fs, file=None):
    """

    :param matrix:
    :param diff:
    :param fs:
    :param file:
    :return: (f1, p, r, m, s) 分别表示F1分数，Precision, Recall , maen and standard error for real point and vail
    """
    p = matrix[0, 0] / (matrix[0, 0] + matrix[1, 0])
    r = matrix[0, 0] / (matrix[0, 0] + matrix[0, 1])
    f1 = (2 * (p * r) / (p + r))
    m = np.mean(diff / fs * 1000)
    s = np.std(diff / fs * 1000)
    print('TP: %0.2f, FP: %0.2f, FN: %0.2f' % (matrix[0, 0], matrix[1, 0], matrix[0, 1]), file=file)
    print("Precision: %0.2f" % (p * 100), file=file)
    print("Recall: %0.2f" % (r * 100), file=file)
    print("F1: %0.2f" % (f1 * 100), file=file)
    print("Diff(mean, ms): %0.2f" % np.mean(diff / fs * 1000), file=file)
    print("Diff(std, ms): %0.2f" % np.std(diff / fs * 1000), file=file)
    return f1, p, r, m, s

