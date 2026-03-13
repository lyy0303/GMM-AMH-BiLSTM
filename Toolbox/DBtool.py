# -*- coding: utf-8 -*-
"""
Database Toolbox


"""
import os
from dataclasses import dataclass
from typing import List, Generator
import pandas as pd
import wfdb
from matplotlib import pyplot as plt
from prominence_delineator import ProminenceDelineator
from typing import List, Generator, Union
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch
from scipy import signal
import neurokit2 as nk2
from paper.ModelOpt.opt_utils import obj_fun, summary


PEAK_SYMBOL = ['N', 'L', 'R', 'e', 'j',        # N
               'A', 'a', 'J', 'S',             # S
               'V', 'E',                       # V
               'F',                            # F
               '/', 'f', 'Q'                   # Q
               ]

SUFFIX = ['i', 'ii', 'iii', 'avr', 'avl', 'avf', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6']


@dataclass
class Data:
    name: str
    signal: np.ndarray
    original_signal: np.ndarray
    fs: int
    sample: np.ndarray      # r-peaks
    symbol: List[str] = None      # symbol
    q_loc: np.ndarray = None
    s_loc: np.ndarray = None
    p_loc: np.ndarray = None
    p_on: np.ndarray = None
    p_off: np.ndarray = None
    t_loc: np.ndarray = None
    t_on: np.ndarray = None
    t_off: np.ndarray = None


class mitArr:
    def __init__(self,
                path: str = None,
                 ) -> None:
        self.path = path or os.path.dirname(__file__) + r'./Data/mit-bih-arrhythmia-database-1.0.0'
    def record(self,
               record_name: str,
               channel: int | str = 'MLII',
               ) -> Data:
        signal, fields = wfdb.rdsamp(f'{self.path}/{record_name}')
        annotation = wfdb.rdann(record_name=f'{self.path}/{record_name}', extension='atr')
        origin_sample = annotation.sample
        origin_symbol = annotation.symbol

        # print(len(r_peaks))
        # print(type(symbol))
        # print(f'len(origin_sample) = {len(origin_sample)}')
        # print(f'len(origin_symbol) = {len(origin_symbol)}')

        if isinstance(channel, int):
            if channel != 0 or channel != 1:
                raise ValueError('channel number only using 0 or 1')
        elif isinstance(channel, str):
            available_channels = fields['sig_name']
            if channel in fields['sig_name']:
                channel = fields['sig_name'].index(channel)
            else:
                if channel == 'MLII' and 'V5' in available_channels:
                    # print(f"Record {record_name}: MLII not available, using V5 instead")
                    channel = available_channels.index('V5')
                elif channel == 'V5' and 'MLII' in available_channels:
                    # print(f"Record {record_name}: V5 not available, using MLII instead")
                    channel = available_channels.index('MLII')
                else:
                    raise ValueError(f"channel name must be in {available_channels}")
        else:
            raise ValueError('channel must be an integer or string')

        filter_sig = np.array(nk2.ecg_clean(signal[ :, channel], sampling_rate=fields['fs'], method="vg"))
        match_beat_type, conf_matrix, diff, r_pre  = self.match_beat_type(sig=filter_sig, r_real=origin_sample, fs=fields['fs'],
                                               symbol=origin_symbol)
        sample = r_pre[np.isin(match_beat_type, PEAK_SYMBOL)]
        symbol = [s for s in match_beat_type if s in PEAK_SYMBOL]

        df = pd.DataFrame(origin_sample, columns=['origin_sample'])

        df['origin_symbol'] = pd.DataFrame(origin_symbol, columns=['origin_symbol'])
        df['sample'] = pd.DataFrame(sample, columns=['sample'])
        df['symbol'] = pd.DataFrame(symbol, columns=['symbol'])
        df.to_csv(f'D:/PycharmProjects/EcgGmm/Toolbox/Match_symbol_VG/symbol_{record_name}.csv', index=False)

        return Data(
            name=record_name,
            original_signal = signal[ :, channel],
            signal=filter_sig,
            fs=fields['fs'],
            sample=sample,
            symbol=symbol
        )

    def match_beat_type(self,
                        sig: np.ndarray,
                        r_real: np.ndarray,
                        fs:int,
                        symbol: List[str] = None
                        ):
        PromDelineator = ProminenceDelineator(sampling_frequency=fs)
        r_pre = PromDelineator.find_rpeaks(sig)
        match_beat_type, conf_matrix, diff = obj_fun(r_real, r_pre, 5, symbol, bisected=False)  # 开始匹配
        # print(f'r_real:{len(r_real)}')
        # print(f'r_pre:{len(r_pre)}')
        # print(f'match_beat_type:{len(match_beat_type)}')
        return match_beat_type, conf_matrix, diff, r_pre

    def records(self,
                record_names: List[str] = None,
                channel: int | str = 'MLII'
                # channel: Union[int, str] = None,
                ) -> Generator[Data, None, None]:
        if record_names is None:
            with open(f'{self.path}/RECORDS', 'r', encoding='utf-8') as f:
                record_names = f.read().splitlines()

        for record_name in record_names:
            try:
                yield self.record(record_name, channel)
            except ValueError as e:
                print(f"Skipping record {record_name}: {e}")
                continue


class luecgdb:
    def __init__(self,
                 path: str = None
                 ) -> None:
        self.path = path or os.path.dirname(
            __file__) + r'./Data/lobachevsky-university-electrocardiography-database-1.0.1'

    def record(self,
               record_name: str,
               channel: int | str = 'MILL',
               ) -> Data:
        signal, fields = wfdb.rdsamp(f'{self.path}/{record_name}')

        if isinstance(channel, int):
            if channel < 0 or channel > 11:
                raise ValueError('channel number only using integer in [0, 11]')
            channel_idx: int = channel
            channel_suffix: str = SUFFIX[channel]
        elif isinstance(channel, str):
            if channel not in fields['sig_name']:
                raise ValueError(f"channel name must be in {fields['sig_name']}")
            channel_idx: int = SUFFIX.index(channel)
            channel_suffix: str = channel
        else:
            raise ValueError('channel must be an integer or string')

        annotation = wfdb.rdann(record_name=f'{self.path}/{record_name}', extension=channel_suffix)

        P_on, P, P_off = self._get_peak_on_off(annotation, symbols=["p"])
        R_on, R, R_off = self._get_peak_on_off(annotation, symbols=["N"])
        T_on, T, T_off = self._get_peak_on_off(annotation, symbols=["t"])

        return Data(
            name=record_name,
            signal=signal[:, channel_idx],
            fs=fields['fs'],
            q_loc=R_on,
            s_loc=R_off,
            p_loc=P,
            p_on=P_on,
            p_off=P_off,
            t_loc=T,
            t_on=T_on,
            t_off=T_off,
        )

    def records(self,
                record_names: List[str] = None,
                channel: int | str = 'i'
                ) -> Generator[Data, None, None]:
        if record_names is None:
            with open(f'{self.path}/RECORDS', 'r', encoding='utf-8') as f:
                record_names = f.read().splitlines()

        for record_name in record_names:
            try:
                yield self.record(record_name, channel)
            except ValueError:
                continue

    @staticmethod
    def _get_peak_on_off(a, symbols=None, left=0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if symbols is None:
            symbols = ["N"]
        a_sym = np.array(a.symbol)

        # find peak and its on/offset
        peak_ix = np.where(np.isin(a_sym, symbols))[0]
        on_ix = peak_ix - 1
        off_ix = peak_ix + 1

        # check bounds
        off_ix = off_ix[off_ix < len(a_sym)]
        on_ix = on_ix[on_ix >= 0]

        # check if it is really on/offset == "(" or ")"
        on = a.sample[on_ix[a_sym[on_ix] == "("]] - left
        off = a.sample[off_ix[a_sym[off_ix] == ")"]] - left
        peaks = a.sample[peak_ix] - left

        return on, peaks, off


if __name__ == '__main__':     # For testing
    # def test1():
    #     db = luecgdb()
    #     print(db.record('data/1').fs)
    #     print(next(db.records()).fs)
    # test1()
    def test0(name):
        db = mitArr()
        # print(type(db.record('104').beat_type))
        # print(len(db.record(name).sample))
        #
        # print(len(db.record(name).symbol))
        filter_sig = db.record(name).signal
        orignal_sig = db.record(name).original_signal
        print(len(filter_sig))
        print(len(orignal_sig))

        print(db.record(name).signal)
        print(db.record(name).name)

        # print(db.record('100').p_on)

    file_name = ['100', '101', '102', '103', '104', '105', '106', '107',
                 '108', '109', '111', '112', '113', '114', '115', '116',
                 '117', '118', '119', '121', '122', '123', '124', '200',
                 '201', '202', '203', '205', '207', '208', '209', '210',
                 '212', '213', '214', '215', '217', '219', '220', '221',
                 '222', '223', '228', '230', '231', '232', '233', '234']
    # for name in file_name:
    #     db = mitArr()
    #     db.record(name)

    # for name in file_name:
    #     test0(name)

    test0('100')