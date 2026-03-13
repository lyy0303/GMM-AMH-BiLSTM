# -*- coding: utf-8 -*-
"""
分别调优POT算法对rho序列和delta序列的检测参数
"""
import itertools
from dataclasses import dataclass
from typing import List, TextIO

import neurokit2 as nk
import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from GaussECG.peak_over_threshold import alarm
from GaussECG.two_stage_density_peak_clustering import TwoStageDPC
from paper.ModelOpt.opt_utils import obj_fun, summary
from Toolbox.DBtool import mitArr, luecgdb


@dataclass(kw_only=True)
class Objs:
    rho: NDArray
    delta: NDArray
    peaks: NDArray


def opt_func(
        objs: List[Objs],
        q1s: NDArray,
        q2s: NDArray,
        delta_quantile: float = 0.98,
        rho_quantile: float = 0.98,
        tol: float = 5,
        fs: float = 360,
        bisected: bool = True,
        file: TextIO = None,

):
    params = itertools.product(q1s, q2s)
    F1, P, R, M, S = 0, 0, 0, 0, 0
    opt_param = None

    for param in params:
        delta_params: dict[str, int | float] = {'q': param[0], 'd': int(fs), 'quantile': delta_quantile}
        rho_params: dict[str, int | float] = {'q': param[1], 'd': int(fs), 'quantile': rho_quantile}

        CM = np.zeros((2, 2))
        DIF = []
        for item in tqdm(objs, desc="Params(delta_q, rho_q): %0.3f,%0.2f" % (param[0], param[1])):
            rho_peaks = alarm(item.rho, method='MOM', **rho_params)
            delta_peaks = alarm(item.delta, method='MOM', **delta_params)
            peaks = np.intersect1d(rho_peaks, delta_peaks)
            cm, dif = obj_fun(item.peaks, peaks, tol, bisected=bisected)
            CM += cm
            DIF = np.append(DIF, dif)
        print("\n=================\n"
              ">> Params(delta_q, rho_q): %0.3f,%0.2f" % (param[0], param[1]), file=file)
        f1, p, r, m, s = summary(CM, DIF, fs, file=file)
        if f1 > F1:
            opt_param = param
            F1, P, R, M, S = f1, p, r, m, s
    print("\n=====SUMMARY=====", file=file)
    print("Opt Params(delta_q, rho_q): %0.3f,%0.2f" % (opt_param[0], opt_param[1]), file=file)
    print("Precision: %0.2f" % (P * 100), file=file)
    print("Recall: %0.2f" % (R * 100), file=file)
    print("F1: %0.2f" % (F1 * 100), file=file)

    return F1, opt_param


def main():
    delta_quantile = 0.97
    rho_quantile = 0.90
    q1s = np.linspace(0.001, 0.009, 9)
    q2s = np.linspace(0.01, 0.05, 5)
    tol = 5

    C = []
    db = mitArr()
    for item in db.records():
        signal = np.array(nk.ecg_clean(item.signal, sampling_rate=item.fs, method="vg"))
        detector = TwoStageDPC(fs=item.fs)
        detector.clc_params(signal)
        cluster_param = Objs(rho=detector.params.rho,
                             delta=detector.params.delta,
                             peaks=item.r_loc)
        C.append(cluster_param)
    f = open(
        f'opt q tol={tol} mitarr ({int(delta_quantile * 100)},{int(rho_quantile * 100)}).text',
        'w')
    opt_func(
        C,
        q1s=q1s,
        q2s=q2s,
        delta_quantile=delta_quantile,
        rho_quantile=rho_quantile,
        tol=tol,
        fs=360,
        bisected=True,
        file=f,
    )
    f.close()

    C = []
    db = luecgdb()
    for item in db.records():
        signal = np.array(nk.ecg_clean(item.signal, sampling_rate=item.fs, method="vg"))
        detector = TwoStageDPC(fs=item.fs, win_len=30)
        detector.clc_params(signal)
        cluster_param = Objs(rho=detector.params.rho,
                             delta=detector.params.delta,
                             peaks=item.r_loc)
        C.append(cluster_param)

    f = open(
        f'opt q tol={tol} luecgdb ({int(delta_quantile * 100)},{int(rho_quantile * 100)}).text',
        'w')
    opt_func(
        C,
        q1s=q1s,
        q2s=q2s,
        delta_quantile=delta_quantile,
        rho_quantile=rho_quantile,
        tol=tol,
        fs=500,
        bisected=True,
        file=f,
    )
    f.close()


if __name__ == '__main__':
    main()
