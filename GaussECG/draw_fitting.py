import ast
import os
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from numpy.typing import NDArray

"""Picture format"""
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams["savefig.transparent"] = True
plt.rcParams["savefig.dpi"] = 900
plt.rcParams["savefig.bbox"] = 'tight'




def read_csv(path, filename):
    """读取CSV文件并处理特殊格式的列"""
    df = pd.read_csv(f'{path}/{filename}')
    # 处理可能存储为字符串的列表/数组
    for col in df.columns:
        if df[col].dtype == object and df[col].str.startswith('[').any():
            df[col] = df[col].apply(ast.literal_eval)
    return df


def plot_fitting_comparison(data,
                            fitted_signal,
                            original_signal,
                            beat_idx,
                            time_points,
                            symbol,
                            parameters,  # 新增参数：高斯参数
                            target_symbols: list = None,  # 新增：目标symbol类型
                            beat_indices: int = None,
                            save_dir: str = None):
    """
    绘制原始分割信号和拟合信号的对比图，包括5个高斯成分
    :param data: 分段数据DataFrame
    :param fitted_signal: 拟合信号列表
    :param original_signal: 原始信号列表
    :param beat_idx: 心拍索引列表
    :param time_points: 时间点列表
    :param symbol: 心拍符号列表
    :param parameters: 高斯参数列表 [a_i, b_i, c_i] * 5
    :param target_symbols: 目标symbol类型列表，如 ['N', 'V', 'L', 'R']，为None则绘制所有
    :param beat_indices: 要绘制的心拍数量，如果为None则绘制所有
    :param save_dir: 保存图片的目录
    """
    # 过滤目标symbol类型
    if target_symbols is not None:
        # 创建过滤掩码
        mask = data['symbol'].isin(target_symbols)

        # 应用过滤
        filtered_data = data[mask]
        filtered_fitted = [fitted_signal[i] for i in range(len(fitted_signal)) if mask.iloc[i]]
        filtered_original = [original_signal[i] for i in range(len(original_signal)) if mask.iloc[i]]
        filtered_beat_idx = [beat_idx[i] for i in range(len(beat_idx)) if mask.iloc[i]]
        filtered_time_points = [time_points[i] for i in range(len(time_points)) if mask.iloc[i]]
        filtered_symbol = [symbol[i] for i in range(len(symbol)) if mask.iloc[i]]
        filtered_parameters = [parameters[i] for i in range(len(parameters)) if mask.iloc[i]]

        print(f"过滤后数据: 总共 {len(filtered_data)} 个心拍，symbol类型: {set(filtered_symbol)}")
    else:
        # 使用所有数据
        filtered_data = data
        filtered_fitted = fitted_signal
        filtered_original = original_signal
        filtered_beat_idx = beat_idx
        filtered_time_points = time_points
        filtered_symbol = symbol
        filtered_parameters = parameters

    # 确定要绘制的心拍数量
    if beat_indices is None:
        beat_indices = len(filtered_original)
    else:
        beat_indices = min(beat_indices, len(filtered_original))

    # 确保保存目录存在
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 5个高斯成分的颜色
    gaussian_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    # gaussian_colors = ['#6F6F6F', '#C0321A', '#629C35', '#DD7C4F', '#6C61AF']
    # gaussian_colors = ['#A5AEB7', '#925EB0', '#7E99F4', '#CC7C71', '#7AB656']
    # gaussian_colors = ['#A5AEB7', '#925EB0', '#7E99F4', '#CC7C71', '#7AB656']
    gaussian_labels = ['Gaussian P', 'Gaussian Q', 'Gaussian R', 'Gaussian S', 'Gaussian T']

    for i in range(beat_indices):
        print(f'开始绘制第 {i + 1}/{beat_indices} 个心拍 (Symbol: {filtered_symbol[i]})')

        # 获取当前心拍的数据
        current_beat_idx = filtered_beat_idx[i]
        current_original = filtered_original[i] if isinstance(filtered_original[i], list) else filtered_original[
            i].tolist()
        current_fitted = filtered_fitted[i] if isinstance(filtered_fitted[i], list) else filtered_fitted[i].tolist()
        current_time = filtered_time_points[i] if isinstance(filtered_time_points[i], list) else filtered_time_points[
            i].tolist()
        current_params = filtered_parameters[i] if i < len(filtered_parameters) else []
        current_symbol = filtered_symbol[i]

        # 确保信号长度一致
        min_length = min(len(current_original), len(current_fitted), len(current_time))
        current_original = current_original[:min_length]
        current_fitted = current_fitted[:min_length]
        current_time = current_time[:min_length]

        fig, ax = plt.subplots(figsize=(10, 5))
        # 绘制原始信号（必显示，图例必加）
        ax.plot(current_time, current_original,
                label='Original Signal',
                color='#2E86AB',
                linewidth=2.5,
                alpha=0.8)
        # 绘制拟合信号（必显示，图例必加）
        ax.plot(current_time, current_fitted,
                label='Fitted Signal',
                color='#A23B72',
                linewidth=2,
                linestyle='--',
                alpha=0.9)

        # 存储实际要显示的图例标签（动态过滤）
        active_labels = ['Original Signal', 'Fitted Signal']

        # 绘制5个高斯成分（仅当参数存在时绘制，且添加图例）
        # 绘制5个高斯成分（仅当参数存在时绘制，且添加图例）
        if current_params and len(current_params) >= 15:  # 不加基线时修改为15
            time_array = np.array(current_time)
            baseline_B = current_params[15]  # 第16个参数是基线B（索引15）

            for j in range(5):
                a = current_params[j * 3]
                b = current_params[j * 3 + 1]
                c = current_params[j * 3 + 2]
                # 过滤无效高斯成分（幅度接近0或标准差异常的不绘制）
                if abs(a) < 1e-6 or c < 1e-3:
                    continue
                # 加上基线B
                # gaussian_component = baseline_B + a * np.exp(-(time_array - b) ** 2 / (2 * c ** 2))    # 加基线
                gaussian_component = a * np.exp(-(time_array - b) ** 2 / (2 * c ** 2))         # 不加基线
                ax.plot(current_time, gaussian_component,
                        label=gaussian_labels[j],
                        color=gaussian_colors[j],
                        linewidth=1.5,
                        linestyle=':',
                        alpha=0.7)
                active_labels.append(gaussian_labels[j])

        # 绘制特征点（仅当存在峰值时添加图例）
        legend_added = set()  # 跟踪已添加的特征点图例

        def _plot_feature_points_adaptive(ax, segment_data, time_points: list, signal: list):
            colors = {
                'P': '#FF6B6B', 'Q': '#4ECDC4', 'R': '#FFD166', 'S': '#06D6A0', 'T': '#118AB2'
            }
            for wave, color in colors.items():
                possible_keys = [f'{wave}', f'{wave}_peak', f'{wave}_peaks', f'{wave.upper()}']
                found_key = None
                for key in possible_keys:
                    if key in segment_data:
                        found_key = key
                        break
                if not found_key:
                    continue
                wave_data = segment_data[found_key]
                peaks = []
                if isinstance(wave_data, list):
                    peaks = np.where(np.array(wave_data) == 1)[0]
                elif isinstance(wave_data, (int, float)) and wave_data >= 0:
                    peaks = [int(wave_data)]
                # 仅当有有效峰值时绘制并添加图例
                if peaks:
                    for idx in peaks:
                        if idx < len(time_points) and idx < len(signal):
                            label = f'{wave} wave' if wave not in legend_added else ""
                            if label:
                                legend_added.add(wave)
                                active_labels.append(label)  # 记录有效图例
                            ax.plot(time_points[idx], signal[idx], 'o',
                                    color=color, markersize=8, markeredgecolor='white', markeredgewidth=1,
                                    label=label)

        # 调用自适应特征点绘制
        _plot_feature_points_adaptive(ax=ax,
                                      segment_data=filtered_data.iloc[i] if i < len(filtered_data) else {},
                                      time_points=current_time,
                                      signal=current_original)

        ax.legend(
            fontsize=10,
            loc='best',
            borderaxespad=0.5,
            frameon=True,
            fancybox=True,
            ncol=2,  # 固定为2列
            columnspacing=1.0,
            labelspacing=0.5
        )

        ax.set_xlabel('Time (ms)', fontsize=16)
        ax.set_ylabel('Amplitude (mV)', fontsize=16)
        # ax.set_title(f'ECG Signal Fitting Comparison - Beat {current_beat_idx} (Symbol: {current_symbol})', fontsize=16)
        ax.tick_params(axis='both', labelsize=14)
        # 保存图片
        if save_dir:
            # 在文件名中包含symbol类型
            if current_symbol == '/':
                plot_path = os.path.join(save_dir, f'fitting_comparison_beat_{current_beat_idx}_I.png')
            else:
                plot_path = os.path.join(save_dir, f'fitting_comparison_beat_{current_beat_idx}_{current_symbol}.png')
            plt.savefig(plot_path, dpi=1000, bbox_inches='tight')
            plt.close()
            print(f"心拍 {current_beat_idx} ({current_symbol}) 的拟合对比图已保存至: {plot_path}")
        else:
            plt.show()



def visualize_5_gaussian_components(
    params: pd.DataFrame,  # 包含A_1~A_5、mu_1~mu_5、sigma_1~sigma_5、B列的DataFrame
    ecg: NDArray,          # 完整ECG信号（一维数组）
    fs: int,               # 采样率
    record_name: str
) -> plt.Figure:
    """
    可视化5个高斯成分的参数分布+ECG信号+基线参数B
    :param params: 每个心拍的高斯参数DataFrame，列包含A_1~A_5、mu_1~mu_5、sigma_1~sigma_5、B
    :param ecg: 完整ECG信号（一维数组）
    :param fs: 采样率
    :param record_name: 记录名称（用于标题）
    :return: 绘制好的matplotlib Figure对象
    """
    if params is None or ecg is None:
        raise ValueError("params或ecg不能为None")
    fig = plt.figure(figsize=(18, 12), facecolor='#f8f9fa')
    # 3行6列：第0行显示ECG；第1行显示5个成分+颜色条；第2行显示基线B
    gs = fig.add_gridspec(
        3, 6,
        height_ratios=[0.6, 1, 0.6],  # 行高比例
        hspace=0.3, wspace=0.2,       # 子图间距
        width_ratios=[1,1,1,1,1,0.1]  # 列宽（前5列放成分，最后1列放颜色条）
    )
    ax_ecg = fig.add_subplot(gs[0, :])  # 第0行占所有列
    time_axis = np.arange(len(ecg)) / fs  # 时间轴（秒）
    ax_ecg.plot(time_axis, ecg,
                linestyle='-', color='#ef7678', linewidth=0.8, zorder=1)
    # 计算1秒窗口的滚动均值
    window_size = int(1 * fs)
    rolling_mean = pd.Series(ecg).rolling(window=window_size, min_periods=1).median()
    ax_ecg.plot(time_axis, rolling_mean,
                linestyle='-', color='#f1cc74', linewidth=2, zorder=2)
    ax_ecg.set_xlim(0, len(ecg)/fs)
    ax_ecg.set_title(f'ECG Signal - {record_name}', fontsize=14)
    ax_ecg.set_xlabel('Time (s)', fontsize=12)
    ax_ecg.set_ylabel('Amplitude (mV)', fontsize=12)
    ax_ecg.grid(True, linestyle='--', alpha=0.7)

    components = [1,2,3,4,5]  # 5个成分（对应P/Q/R/S/T）
    wave_names = ['P', 'Q', 'R', 'S', 'T']  # 成分对应的波形名称
    scatter = None  # 用于后续颜色条

    for i, comp in enumerate(components):
        ax = fig.add_subplot(gs[1, i])  # 第1行，第i列
        # 提取当前成分的参数
        A = params[f'A_{comp}']
        mu = params[f'mu_{comp}'] - params['mu_3']  # 以R波（第3个成分）为中心
        sigma = params[f'sigma_{comp}']
        # 绘制散点图
        scatter = ax.scatter(
            mu, sigma,
            s=15, alpha=0.7,
            c=A*10, cmap='Oranges',  # 颜色对应幅度A
            edgecolors='black', linewidth=0.5
        )
        # 设置子图标签
        ax.set_xlabel(f'$\\mu_{comp}$ ({wave_names[i]})', fontsize=12)
        ax.set_ylabel(f'$\\sigma_{comp}$', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)

    cax = fig.add_subplot(gs[1, 5])  # 第1行最后1列
    cbar = fig.colorbar(scatter, cax=cax)
    cbar.set_label('Amplitude (scaled)', fontsize=12)

    ax_b = fig.add_subplot(gs[2, :])  # 第2行占所有列
    ax_b.plot(params.index, params['B'],
              marker='o', linestyle='-', color='#ef7678',
              linewidth=2, markersize=8,
              markerfacecolor='white', markeredgewidth=2)
    ax_b.set_xlim(0, len(params.index))
    ax_b.set_xlabel('Index of Heart Beats', fontsize=12)
    ax_b.set_ylabel('Value of B', fontsize=12)
    ax_b.grid(True, linestyle='--', alpha=0.7)


    # 调整子图间距
    plt.subplots_adjust(top=0.95, bottom=0.05, left=0.05, right=0.95)
    plt.savefig(f'D:/PycharmProjects/EcgGmm/GaussECG/MIT ARR可视化图/{record_name}.png', dpi=1000, bbox_inches='tight')
    return fig


if __name__ == '__main__':
    from Toolbox.DBtool import mitArr
    file_names = [
    '100', '101', '102', '103', '104', '105', '106', '107',
                  '108', '109', '111', '112', '113', '114', '115', '116',
                  '117', '118', '119', '121', '122', '123', '124', '200',
                  '201', '202', '203',
                  '205', '207', '208', '209', '210',
                  '212', '213', '214', '215', '217', '219', '220', '221',
                  '222', '223', '228', '230', '231', '232', '233', '234']

    # 测试单个文件
    def main(name, path, type):
        db = mitArr()
        fs = db.record(name).fs
        df_fitting = read_csv(path, 'fitting_results.csv')
        print("所有symbol类型:", df_fitting['symbol'].unique())
        full_ecg = []
        for seg in df_fitting['original_signal']:
            full_ecg.extend(seg)  # 拼接每个心拍的信号
        full_ecg = np.array(full_ecg)  # 转为一维数组
        params_list = []
        for param_seq in df_fitting['parameters']:
            # 假设param_seq是 [A_1, mu_1, sigma_1, ..., A_5, mu_5, sigma_5, B]
            params_dict = {
                'A_1': param_seq[0], 'mu_1': param_seq[1], 'sigma_1': param_seq[2],
                'A_2': param_seq[3], 'mu_2': param_seq[4], 'sigma_2': param_seq[5],
                'A_3': param_seq[6], 'mu_3': param_seq[7], 'sigma_3': param_seq[8],
                'A_4': param_seq[9], 'mu_4': param_seq[10], 'sigma_4': param_seq[11],
                'A_5': param_seq[12], 'mu_5': param_seq[13], 'sigma_5': param_seq[14],
                'B': param_seq[15]
            }
            params_list.append(params_dict)
        params_df = pd.DataFrame(params_list)  # 转为包含A_1~A_5等列的DataFrame
        # 绘制5个高斯成分
        # 调用可视化函数
        # visualize_5_gaussian_components(
        #     params=params_df,  # 修正：传入解析后的参数DataFrame
        #     ecg=full_ecg,  # 修正：传入拼接后的完整ECG信号
        #     fs=fs,
        #     record_name=name
        # )
        plot_fitting_comparison(data=df_fitting,
                                fitted_signal=df_fitting['fitted_signal'],
                                original_signal=df_fitting['original_signal'],
                                beat_idx=df_fitting['beat_index'],
                                time_points=df_fitting['time_points'],
                                symbol=df_fitting['symbol'],
                                parameters=df_fitting['parameters'],
                                target_symbols=[type],
                                # beat_indices=10,
                                save_dir=path + '/view')
    for name in ['200']:
        ty = 'N'
        # for name in file_names:
        #     os.makedirs(f'D:/PycharmProjects/EcgGmm/GaussECG/signal_and_params_of_5_gaussian_VG/{name}')
        # path1 = f'D:/PycharmProjects/EcgKits-master/GaussECG/signal_and_params_of_5_gaussian_VG_1_skip/{name}'
        path2 = f'D:/PycharmProjects/EcgGmm/GaussECG/signal_and_params_of_5_gaussian_VG/{name}'
        # main(name, path1, ty)
        main(name, path2, ty)

