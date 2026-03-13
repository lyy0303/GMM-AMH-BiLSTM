# -*- coding: utf-8 -*-
"""
plot figure for ECG
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from prominence_delineator import ProminenceDelineator

"""Picture format"""
FontSize: int = 16  # label font size
plt.rcParams['font.size'] = FontSize - 2
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams["savefig.transparent"] = True
plt.rcParams["savefig.dpi"] = 900
plt.rcParams["savefig.bbox"] = 'tight'

Color1 = ['#c74d26', ]
Color = ['#f7a6ac', '#f7b7d2', '#eec186', '#eef0a7', '#b2dbb9', '#b8e5fa']
WAVE_CONFIG = {
    'P': ('^', 'o', '#e74c3c'),    # P波：on/off=上三角，波峰=圆点，颜色=红色
    'R': ('v', 'o', '#2ecc71'),    # R波：on/off=下三角，波峰=圆点，颜色=绿色
    'T': ('<', 'o', '#3498db'),    # T波：on/off=左三角，波峰=圆点，颜色=蓝色
    'Q': ('>', 'o', '#f39c12'),    # Q波（可选）：on/off=右三角，波峰=圆点，颜色=橙色
    'S': ('D', 'o', '#9b59b6')     # S波（可选）：on/off=菱形，波峰=圆点，颜色=紫色
}


def viz_ecg(
        signal: NDArray,
        fs: int,
        path: str | None = None,
):
    """
    绘制心电信号，自定义PRT等波形的标记样式和颜色：
    - PRT波峰：圆点
    - P_on/P_off：同色三角形，R_on/R_off：同色三角形，以此类推
    :param signal: 心电信号
    :param fs: 采样频率
    :param path: 图片保存路径（None则直接显示）
    """
    ProDelineator = ProminenceDelineator(sampling_frequency=fs)
    rpeaks = ProDelineator.find_rpeaks(signal)
    ecg = ProDelineator.clean_ecg(signal)
    wave = ProDelineator.find_waves(ecg, rpeaks=rpeaks, include_nodetections=True)
    fig, ax = plt.subplots(1, 1, figsize=(16, 5))

    # 绘制原始ECG信号
    x = np.arange(len(signal)) / fs
    ax.plot(x, signal, label='ECG', color=Color1[0], linewidth=1)

    ncol = 1
    if wave is not None:
        # 用于存储图例项（避免重复）
        legend_handles = []
        legend_labels = []

        for item in wave.keys():
            positions = wave.get(item)
            if not positions:  # 跳过空列表
                continue

            # 1. 匹配波形配置（根据item前缀判断属于哪个波形，如P_on→P，R_peak→R）
            wave_prefix = None
            for prefix in WAVE_CONFIG.keys():
                if item.startswith(prefix):
                    wave_prefix = prefix
                    break
            if wave_prefix is None:
                continue  # 跳过未配置的波形项

            # 2. 获取当前波形的样式（标记+颜色）
            on_off_marker, peak_marker, color = WAVE_CONFIG[wave_prefix]

            # 3. 区分是“起止位置（on/off）”还是“波峰（peak）”
            if item.endswith(('on', 'off')):
                marker = on_off_marker  # 起止位置用三角形（或配置的标记）
                label_suffix = item.split('_')[-1]  # 提取on/off
                legend_label = f'{wave_prefix}_{label_suffix}'
            elif item in (wave_prefix, f'{wave_prefix}_peak'):  # 波峰项（如P、R、T或P_peak）
                marker = peak_marker  # 波峰用圆点
                legend_label = f'{wave_prefix}_peak'
            else:
                continue  # 跳过其他未定义项

            # 4. 过滤无效位置（None + 索引越界）
            valid_positions = [
                pos for pos in positions
                if pos is not None and isinstance(pos, (int, np.int64)) and 0 <= pos < len(signal)
            ]
            if not valid_positions:
                continue

            # 5. 转换为时间轴（x轴）和信号值（y轴）
            valid_times = [pos / fs for pos in valid_positions]
            valid_signals = [signal[pos] for pos in valid_positions]

            # 6. 绘制散点，并记录图例（避免重复）
            scatter = ax.scatter(valid_times, valid_signals, color=color, marker=marker, s=50)
            if legend_label not in legend_labels:
                legend_labels.append(legend_label)
                legend_handles.append(scatter)

        # 7. 配置图例（添加ECG线的图例）
        ecg_line = ax.plot([], [], label='ECG', color=Color1[0], linewidth=0.7)[0]
        legend_handles.insert(0, ecg_line)
        legend_labels.insert(0, 'ECG')

        # 调整图例列数（根据图例数量自动分配）
        ncol = min(len(legend_labels), 4)  # 最多4列，避免重叠

    # 设置坐标轴范围和刻度
    signal_range = max(signal) - min(signal)
    ax.set_ylim(
        min(signal) - signal_range * 0.1,
        max(signal) + signal_range * 0.02
    )
    ax.set_xlim(-0.1, len(signal) / fs + 0.1)
    ax.set_xticks(np.linspace(0, int(len(signal) / fs), 9))
    ax.set_xlabel('Time (s)', fontsize=FontSize)  # 原代码是ms，这里修正为s（根据x轴计算逻辑）
    ax.set_ylabel('Amplitude', fontsize=FontSize)  # 补充y轴标签

    # 调整布局和图例（将图例放到上方）
    plt.tight_layout(rect=(0, 0, 1, 1.2))  # 预留顶部空间（rect=[左, 下, 右, 上]）
    ax.legend(
        handles=legend_handles,
        labels=legend_labels,
        ncol=ncol,  # 列数保持自动分配
        loc='upper center',  # 图例定位到上方居中
        fontsize=FontSize - 6,
        handletextpad=0.3,
        columnspacing=0.8,
        frameon=False,
        bbox_to_anchor=(0.5, 1.15)  # 调整锚点，让图例显示在图的上方
    )

    # 保存或显示图片
    if path is not None:
        fig.savefig(path, bbox_inches='tight')  # tight裁剪空白
        plt.close(fig)
    else:
        plt.show()


def viz_heartbeats(data: pd.DataFrame, path: str = None):
    """
    Heartbeats Visualization
    以R波峰位置为0坐标点绘制每个心拍在一个图层。
    :param data: ecg signal
    :param path:
    :return:
    """
    cols = {'signal', 'index', 'cluster_id'}
    if not cols.issubset(data.columns):
        missing = cols - set(data.columns)
        raise ValueError(f"Missing required columns: {missing}")

    heartbeat_mean = data.groupby('index')['signal'].mean()

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    data_grouped = data.groupby('cluster_id')
    alpha = 1 / np.log2(np.log2(len(data_grouped)))
    data_grouped.plot(x='index', y='signal', ax=ax, c='#858786', alpha=alpha, linewidth=alpha, legend=False, zorder=2)
    ax.plot(heartbeat_mean, c='red', zorder=1)
    ax.axvline(0, c='#ed2225', linestyle='--', zorder=0)
    ax.set_xlabel('Indices (with R-wave peak as 0)', fontsize=FontSize)

    if path is not None:
        fig.savefig(path)
        plt.close(fig)
    else:
        plt.show()


if __name__ == '__main__':
    from DBtool import mitArr
    def main():
        mit = mitArr()
        dt = mit.record('124')
        signal = dt.signal[ :1000]
        fs = dt.fs
        viz_ecg(signal=signal,
                fs=fs,
                path=f'viz_ecg_example_of{dt.name}.png'
                )
        # data = pd.DataFrame({'signal': signal, 'index': index, 'cluster_id': cluster_id})
        # viz_heartbeats(data, 'test.png')

    main()


