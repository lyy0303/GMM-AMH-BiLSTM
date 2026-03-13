# -*- coding: utf-8 -*-
"""
Parameterisation of Heartbeat Shape Using Gaussian Multi-peak Fitting
"""
from typing import Tuple
from tqdm import tqdm
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from scipy.optimize import minimize, OptimizeResult
import os
from DataProcess.heart_beats_segment import segment

"""Picture format"""
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams["savefig.transparent"] = True
plt.rcParams["savefig.dpi"] = 900
plt.rcParams["savefig.bbox"] = 'tight'


def multiGauss(
        x: NDArray[float],
        *params
) -> NDArray[float]:
    """
    1d Multi-Gaussian Function
    :param x: The interval sequence for signal.
    :param params: Gaussian parameters conatins the n parameters (A, mu, sigma) and baseline B.
    :return: The multi-Gaussian values.
    """
    if len(params) % 3 != 1:
        raise ValueError(
            "The number of parameters is incorrect. `params` should like \n"
            "[A_0, mu_0, sigma_0, \n"
            "...\n"
            "A_n, mu_n, sigma_n, B] \n"
        )
    return sum(amp * np.exp(-(x - mean) ** 2 / (2 * sigma ** 2)) for amp, mean, sigma in
               zip(params[::3], params[1::3], params[2::3])) + params[-1]


def error_function(
        params: NDArray[float],
        x: NDArray[float],
        y: NDArray[float],
        weight_power: float = 1
) -> float:
    """
    Weighted least squares error function to be minimized
    :param params: Gaussian parameters
    :param x: The x-axis values
    :param y: The observed y values
    :param weight_power: The power weighting
    :return: Weighted error value
    """
    y_pred = multiGauss(x, *params)
    # Generate weights: gives better fit at peak positions
    _y = y - np.min(y)
    weights = _y ** weight_power / (_y ** weight_power).sum()
    return ((y - y_pred) ** 2 * weights).sum()


def _log_error(message: str):
    with open('log_1.txt', 'a', encoding='utf-8') as f:
        f.write(message + "\n")


def get_params0(
        x: NDArray,
        y: NDArray,
        c: NDArray[int],
) -> NDArray[float]:
    clusters = np.unique(c[c > 0])
    b = np.median(y)

    p0 = []
    expected_clusters = 5  # P, Q, R, S, T five components
    for cluster_idx in range(expected_clusters):
        cluster = cluster_idx + 1

        if cluster in clusters:
            mask = (c == cluster)
            cluster_x = x[mask]
            cluster_y = y[mask]

            cluster_mean = np.mean(cluster_y) - b
            if cluster_mean > 0:
                # Positive wave
                peak_idx = np.argmax(cluster_y)
                A = np.max(cluster_y) - b
                polarity = 1
            else:
                # Negative wave
                peak_idx = np.argmin(cluster_y)
                A = np.min(cluster_y) - b
                polarity = -1
            mu = cluster_x[peak_idx]

            half_value = b + A / 2
            if polarity > 0:
                above_half = cluster_y > half_value
            else:
                above_half = cluster_y < half_value

            if np.sum(above_half) > 1:
                fwhm = cluster_x[above_half][-1] - cluster_x[above_half][0]
                sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
            else:
                # When the half-width at half-maximum is insufficient, use one-fourth of the cluster width as the default value
                sigma = (cluster_x[-1] - cluster_x[0]) / 4

            p0.extend([A, mu, sigma])

    p0.append(b)
    return np.array(p0)

def get_bounds(
        p0: NDArray[float],
        delta_peak: float = 0.5,
        delta_mu: float = 10.0,
        delta_sigma: float = 50.0,
) -> Tuple[NDArray[float], NDArray[float]]:
    """
    :param p0: The initial parameters.
    :param delta_peak: Range of variation of the peak value (fraction of amplitude).
    :param delta_mu: Range of variation of the mu (absolute value).
    :param delta_sigma: Range of variation of the sigma (absolute value).
    :return: Lower and upper bounds for parameters
    """
    lower = []
    upper = []

    # Process each Gaussian component (A, mu, sigma)
    for i in range(0, len(p0) - 1, 3):
        A, mu, sigma = p0[i], p0[i + 1], p0[i + 2]

        # 振幅边界：根据极性调整
        if A > 0:  # 正向波
            lower.append(np.max((0, A * (1 - delta_peak))))
            upper.append(A * (1 + delta_peak))
        else:  # 负向波
            lower.append(A * (1 + delta_peak))
            upper.append(np.min((0, A * (1 - delta_peak))))

        # Mean bounds
        lower.append(mu - delta_mu)
        upper.append(mu + delta_mu)

        # Sigma bounds
        lower.append(np.max((0.1, sigma - delta_sigma)))  # sigma must be > 0
        upper.append(sigma + delta_sigma)

    # Baseline bounds
    lower.append(p0[-1] - 1)  # assuming baseline is non-negative
    upper.append(p0[-1] + 1)  # allow baseline to double

    return np.array(lower), np.array(upper)


def fit_multi_Gauss(
        x: NDArray,
        y: NDArray,
        p0: NDArray,
        weight_power: float = 1,
        delta_peak: float = 0.5,
        delta_mu: float = 10.0,
        delta_sigma: float = 50.0,
) -> Tuple[NDArray[float], OptimizeResult]:

    """
    Fit multi-Gaussian model to data using weighted least squares

    :param x: The x-axis values
    :param y: The observed y values
    :param p0: Initial parameters
    :param weight_power: Power for weighting function (higher emphasizes peaks)
    :param delta_peak: Fractional range for amplitude variation
    :param delta_mu: Absolute range for mean variation
    :param delta_sigma: Absolute range for sigma variation
    :return: Optimized parameters and optimization result info
    """
    # Get bounds
    lower_bounds, upper_bounds = get_bounds(p0, delta_peak, delta_mu, delta_sigma)
    bounds = list(zip(lower_bounds, upper_bounds))

    result = minimize(
        fun=error_function,
        x0=p0,
        args=(x, y, weight_power),
        method="L-BFGS-B",
        bounds=bounds,
        tol=1e-20
    )

    return result.x, result


class MGF:
    def __init__(
            self,
            ecg: NDArray,
            fs: int,
            samples: np.ndarray,
            record_name: str = None,
            symbols: list[str] = None,
    ):
        self.ecg = ecg
        self.fs = fs
        self.samples = samples
        self.segments_data = []   # save segments
        self.symbols = symbols or []
        self.record_name = record_name
        self.fitting_results = []   # save fitting seg
        self.log_file = f"fitting_errors.txt"  # log
        self.params = pd.DataFrame(data=[],
                                   columns=['A_1', 'mu_1', 'sigma_1',
                                            'A_2', 'mu_2', 'sigma_2',
                                            'A_3', 'mu_3', 'sigma_3',
                                            'A_4', 'mu_4', 'sigma_4',
                                            'A_5', 'mu_5', 'sigma_5',
                                            'B']
                                   )
    def fit(
            self,
            weight_power: float = 1,
            delta_peak: float = 0.5,
            delta_mu: float = 10.0,
            delta_sigma: float = 50.0,
            show: bool = False,
            _show: bool = False,
            save_segments: bool = False,
            save_dir: str = None,
    ):
        """
        Using Gaussian multi-peak fitting to extract the morphological parameters of
        ECG signals after segmentation by the VG algorithm and Peak Prominence method.
        """
        # segment:
        segments = segment(self.ecg, self.fs, self.samples, method='Prominence')

        # fitting:
        params = pd.DataFrame(data=[],
                              columns=['A_1', 'mu_1', 'sigma_1',
                                       'A_2', 'mu_2', 'sigma_2',
                                       'A_3', 'mu_3', 'sigma_3',
                                       'A_4', 'mu_4', 'sigma_4',
                                       'A_5', 'mu_5', 'sigma_5',
                                       'B'])

        skipped_beats = 0
        for beat_idx, s in tqdm(segments.items(), desc='fitting process', total=len(segments)):
            try:
                symbol = self._get_beat_symbol(beat_idx)

                c = np.zeros(s.shape[0], dtype=int)
                for i, (start, end) in enumerate(
                        zip(['P_on', 'Q_on', 'R_on', 'S_on', 'T_on'], ['P_off', 'Q_off', 'R_off', 'S_off', 'T_off'])):
                    on_list = np.where(s[start] == 1)[0]
                    off_list = np.where(s[end] == 1)[0]
                    if len(on_list) == 0 or len(off_list) == 0:
                        continue
                    onset = on_list[0]
                    offset = off_list[0]
                    c[onset:offset] = i + 1

                p0 = get_params0(s.index.to_numpy(), s.signal.to_numpy(), c)

                popt, pcov = fit_multi_Gauss(s.index.to_numpy(), s.signal.to_numpy(), p0,
                                             weight_power=weight_power,
                                             delta_peak=delta_peak,
                                             delta_mu=delta_mu,
                                             delta_sigma=delta_sigma)

                if len(popt) != len(params.columns):
                    error_msg = f"{self.record_name}： The length of the fitted parameters {len(popt)} does not match the expected {len(params.columns)}, skipping"
                    print(error_msg)
                    _log_error(error_msg)
                    skipped_beats += 1
                    continue
                params.loc[len(params)] = popt
                # 存储拟合结果
                fitting_result = self._extract_fitting_result(beat_idx=beat_idx,
                                                              segment_df=s,
                                                              popt=popt,
                                                              result=pcov,
                                                              symbol=symbol)
                self.fitting_results.append(fitting_result)
            except Exception as e:
                error_msg = f"{self.record_name} an exception occurred during heartbeat {beat_idx} processing: {e}"
                print(error_msg)
                _log_error(error_msg)
                skipped_beats += 1
                continue
        self.params = params
        total_beats = len(segments)
        success_beats = total_beats - skipped_beats
        msg = f"Fitting completed: a total of {total_beats} heartbeats, {success_beats} successful, {skipped_beats} skipped"
        print(msg)
        _log_error(msg)

        if save_segments:
            self.save_segments_to_csv(save_dir)
        if show:
            self._show()

    def _get_beat_symbol(self, beat_idx: int) -> str:
        """获取心拍标签"""
        if self.symbols and beat_idx < len(self.symbols):
            return self.symbols[beat_idx]
        return ""

    def _extract_fitting_result(self, beat_idx: int, segment_df: pd.DataFrame,
                                popt: NDArray, result: OptimizeResult, symbol: str) -> dict:
        """提取拟合结果数据"""
        x = segment_df.index.to_numpy()
        y_original = segment_df['signal'].to_numpy()
        y_fitted = multiGauss(x, *popt)

        data = {
            'beat_index': beat_idx,
            'record_name': self.record_name or 'unknown',
            'symbol': symbol,
            'fitting_success': result.success,
            'fitting_error': result.fun,
            'original_signal': y_original.tolist(),
            'fitted_signal': y_fitted.tolist(),
            'signal_length': len(segment_df),
            'time_points': x.tolist(),
            'parameters': popt.tolist()
        }
        wave_columns = {
            'P_peaks': 'P',
            'Q_peaks': 'Q',
            'R_peaks': 'R',
            'S_peaks': 'S',
            'T_peaks': 'T',
            'P_on': 'P_on',
            'P_off': 'P_off',
            'Q_on': 'Q_on',
            'Q_off': 'Q_off',
            'R_on': 'R_on',
            'R_off': 'R_off',
            'S_on': 'S_on',
            'S_off': 'S_off',
            'T_on': 'T_on',
            'T_off': 'T_off'
        }

        for data_key, df_key in wave_columns.items():
            if df_key in segment_df.columns:
                data[data_key] = segment_df[df_key].tolist()
            else:
                data[data_key] = [0] * len(segment_df)

        return data

    def save_segments_to_csv(self, save_dir: str = None):
        """保存分割信号和拟合结果到CSV文件"""
        if save_dir is None:
            save_dir = os.getcwd()


        os.makedirs(save_dir, exist_ok=True)

        fitting_df = pd.DataFrame(self.fitting_results)
        fitting_path = os.path.join(save_dir, 'fitting_results.csv')
        fitting_df.to_csv(fitting_path, index=False)
        print(f"The fitting results have been saved to: {fitting_path}")

        params_path = os.path.join(save_dir, 'gaussian_params.csv')
        self.params.to_csv(params_path, index=False)
        print(f"The Gaussian parameters have been saved to: {params_path}")

    def _show(self):
        if self.params is None:
            raise ValueError("Attribute 'params' is None")

        fig, ax = plt.subplots(16, 1, sharex=True, figsize=(10, 12))
        for i, item in enumerate(self.params.columns):
            ax[i].plot(self.params[item])
            ax[i].set_ylabel(str(item))
        plt.tight_layout()
        return fig





if __name__ == '__main__':
    from Toolbox.DBtool import mitArr, luecgdb

    file_name = [
                 '100', '101', '102', '103', '104', '105', '106', '107',
                 '108', '109', '111', '112', '113', '114', '115', '116',
                 '117', '118',   '121', '122', '123', '124', '200',
                 '201', '202', '203', '205', '207', '208', '209', '210',
                 '212', '213', '214', '215', '217', '219', '220', '221',
                 '222', '223', '228', '230', '231', '232', '233', '234']
    db = mitArr()
    # for index in file_name:
    #     out_dir = f'signal_and_params_of_5_gaussian_VG_1_skip/{index}/view'
    #     os.makedirs(out_dir, exist_ok=True)

    for data in db.records(file_name):
        # print(type(data))
        out_dir = f'signal_and_params_of_5_gaussian_VG_1_skip/{data.name}'
        ecg = data.signal
        fs = data.fs
        # print("fs",fs)
        # print(ecg)
        mgf = MGF(ecg=ecg,
                  fs=fs,
                  samples=data.sample,
                  record_name=data.name,
                  symbols=data.symbol
                  )
        # print(data.beat_type)
        mgf.fit(
            save_segments=True,
            save_dir=out_dir
        )


