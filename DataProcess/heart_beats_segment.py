# -*- coding: utf-8 -*-
from typing import Literal, Tuple, Dict, Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pandas import DataFrame
from prominence_delineator import ProminenceDelineator
import os
from scipy.signal import find_peaks
from matplotlib import pyplot as plt
# from plot_seg import viz_ecg

PEAK_SYMBOL = ['N', 'L', 'R', 'e', 'j',        # N
               'A', 'a', 'J', 'S',             # S
               'V', 'E',                       # V
               'F',                            # F
               '/', 'f', 'Q'                   # Q
               ]
"""Picture format"""
FontSize: int = 16  # label font size
plt.rcParams['font.size'] = FontSize - 2
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams["savefig.transparent"] = True
plt.rcParams["savefig.dpi"] = 900
plt.rcParams["savefig.bbox"] = 'tight'



def _log(message: str):
    """将错误信息追加到日志文件"""
    with open('log.txt', 'a', encoding='utf-8') as f:
        f.write(message + "\n")


def segment(
        # name, symbol,
        signal: NDArray,
        fs: float,
        sample: np.ndarray,
        method: Literal['Prominence', 'DPC'] = 'Prominence',
):
    """
    Segment the ECG signal.
    :param signal: The original ECG signal.
    :param fs: The sampling frequency.
    :param method: Segmentation method.
    :return:
        Dict{``key``: ``int``, ``value``: ``DataFrame``}, the DataFrame contains ECG segment information: \n
        'signal', 'P', 'R', 'T', 'P_on', 'P_off', 'R_on', 'R_off', 'T_on', 'T_off', and the laster 9 columns
        is bool array which 1 indicates the peak location.
    """
    if method == 'Prominence':
        # return _segment_prominence(name, symbol, signal, fs, sample)
        return _segment_prominence(signal, fs, sample)
    elif method == 'DPC':
        return _segment_dpc(signal, fs)
    else:
        raise ValueError('Invalid segmentation method.')

def _segment_prominence(
        # name, symbol,
        signal: NDArray,
        fs: float,
        sample: np.ndarray,
) -> tuple[dict[int, DataFrame], NDArray]:
    """
    Using the VG Algorithm and Peak Prominence to search the feature points in the ECG signal, and segment from P-onset
    to T-offset in each heart beat.
    :param signal: The original ECG signal.
    :param fs: The sampling frequency.
    :return:
        Dict{``key``: ``int``, ``value``: ``DataFrame``}, the DataFrame contains ECG segment information: \n
        'signal', 'P', 'R', 'T', 'P_on', 'P_off', 'R_on', 'R_off', 'T_on', 'T_off', and the laster 9 columns
        is bool array which 1 indicates the peak location.
    """
    PromDelineator = ProminenceDelineator(sampling_frequency=fs)
    ecg = PromDelineator.clean_ecg(signal)
    waves = PromDelineator.find_waves(ecg, rpeaks=sample, include_nodetections=True)    # waves是一个字典{'': []}，键为PRT及各波的起始位置和终止位置
    waves_new = find_positions(fs, ecg, waves)
    blocks = {}
    # viz_ecg(waves_new, ecg[: 500], fs, f'view_ecg/{name}.png')
    # waves_new['symbol'] = symbol
    # print(pd.DataFrame(waves_new))
    # (pd.DataFrame(waves_new)).to_csv(f'Peaks/{name}.csv', index=False)
    # Ensure that the waveform being segmented is complete
    none_idx = [i for i in range(len(waves_new['P_on']))
                if waves_new['P_on'][i] is None or waves_new['T_off'][i] is None]
    for i, (onset, offset) in enumerate(zip(waves_new['P_on'], waves_new['T_off'])):
        if i in none_idx:
            continue
        seg_sig = signal[onset:offset + 1]
        block = pd.DataFrame({'signal': seg_sig})
        for key in waves_new.keys():
            peak_loc = np.zeros_like(seg_sig, dtype=bool)
            # print(i, key, waves[key][i], onset, waves[key][i] - onset)
            peak_loc[waves_new[key][i] - onset] = True
            block[key] = peak_loc.astype(int)
        blocks[i] = block

    return blocks


def find_valley(start, end, filter_ecg):
    segment = filter_ecg[start:end + 1]
    valleys, _ = find_peaks(-segment)
    if len(valleys) == 0:
        return start + np.argmin(segment)
    return start + valleys[np.argmin(segment[valleys])]

def find_positions(fs, filter_ecg, waves):
    ms120 = int(120 * fs / 1000)
    ms180 = int(180 * fs / 1000)
    R = waves['R']
    Q = []
    S = []
    Q_on = []
    Q_off = []
    S_on = []
    S_off = []
    threshold = 0.05

    for i in range(len(R)):
        r_peak = R[i]
        # ====================== search Q wave ======================
        # R peak 120ms ahead
        Q_search_start = max(0, r_peak - ms120)
        Q_search_end = r_peak
        search_sig = filter_ecg[Q_search_start: Q_search_end + 1]
        if len(search_sig) == 0:
            Q.append(np.nan)
            Q_on.append(np.nan)
            Q_off.append(np.nan)
            S.append(np.nan)
            S_on.append(np.nan)
            S_off.append(np.nan)
            continue

        q_valley_idx = find_valley(Q_search_start, Q_search_end, filter_ecg)
        Q.append(q_valley_idx)

        # ====================== search P_on ======================
        q_baseline_start = max(0, q_valley_idx - 50)
        q_baseline_end = max(0, q_valley_idx - 10)
        if q_baseline_end <= q_baseline_start:
            baseline_q = np.mean(filter_ecg)
        else:
            baseline_q = np.mean(filter_ecg[q_baseline_start: q_baseline_end])
        Q_start = Q_search_start
        for j in range(q_valley_idx, Q_search_start - 1, -1):
            if abs(filter_ecg[j] - baseline_q) < threshold:
                Q_start = j
                break
        Q_on.append(Q_start)
        # ====================== search P_off ======================
        Q_end = q_valley_idx
        for m in range(q_valley_idx, r_peak):
            if abs(filter_ecg[m] - baseline_q) < threshold:
                Q_end = m
                break
        Q_off.append(Q_end)

        # ====================== search S wave ======================
        s_search_start = Q_end
        s_search_end = min(len(filter_ecg) - 1, s_search_start + ms180)
        s_search_sig = filter_ecg[s_search_start: s_search_end + 1]

        if len(s_search_sig) == 0:
            S.append(np.nan)
            S_on.append(np.nan)
            S_off.append(np.nan)
            continue
        # 计算S波谷（转全局索引）
        s_valley_idx = find_valley(s_search_start, s_search_end, filter_ecg)
        S.append(s_valley_idx)

        # ====================== search S_on ======================
        baseline_window = 20
        baseline_start = min(len(filter_ecg) - 1, s_valley_idx + int(0.1 * fs))
        baseline_end = min(len(filter_ecg) - 1, baseline_start + baseline_window)
        if baseline_end <= baseline_start:
            baseline_s = np.mean(filter_ecg)
        else:
            baseline_s = np.mean(filter_ecg[baseline_start: baseline_end])
        s_start = s_valley_idx  # 默认值
        limit_min = min(r_peak, s_valley_idx)
        limit_max = max(r_peak, s_valley_idx)
        for k in range(limit_min, limit_max + 1):
            if abs(filter_ecg[k] - baseline_s) < threshold:
                s_start = k
                break
        S_on.append(s_start)

        # ====================== search S_off ======================
        s_end = s_valley_idx
        search_limit = min(len(filter_ecg) - 1, s_valley_idx + ms180)
        for idx in range(s_valley_idx, search_limit + 1):
            if abs(filter_ecg[idx] - baseline_s) < threshold:
                s_end = idx
                break
        S_off.append(s_end)

    # update waves
    waves['Q'] = Q
    waves['Q_on'] = Q_on
    waves['Q_off'] = Q_off
    waves['S'] = S
    waves['S_on'] = S_on
    waves['S_off'] = S_off
    # print(len(R))
    # print(len(Q))
    # print(len(Q_on))
    # print(len(Q_off))
    # print(len(S))
    # print(len(S_on))
    # print(len(S_off))
    # pd.DataFrame(waves).to_csv(f'Peaks/peaks.csv', index=False)
    return waves


def _segment_dpc(signal, fs):
    ...


if __name__ == '__main__':     # for testing
    from Toolbox.DBtool import mitArr

    file_name = ['100', '101', '102', '103', '104', '105', '106', '107',
                 '108', '109', '111', '112', '113', '114', '115', '116',
                 '117', '118', '119', '121', '122', '123', '124', '200',
                 '201', '202', '203', '205', '207', '208', '209', '210',
                 '212', '213', '214', '215', '217', '219', '220', '221',
                 '222', '223', '228', '230', '231', '232', '233', '234']
    db = mitArr()
    for idx in ['102']:
        data = db.record(idx)
        # print(data.symbol[258])
        # print(type(data.sample))
        blocks = segment(data.signal, data.fs, data.sample, method='Prominence')
        # blocks = segment(data.name, data.symbol, data.signal, data.fs, data.sample, method='Prominence')
        # print(blocks)
        # for save
        # df_list = list(blocks.values())
        # combiend_df = pd.concat(df_list)
        # combiend_df.to_csv(f"segments_VG_1/ecg_segment_example_{data.name}.csv")


