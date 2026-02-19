#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
在线蓝牙数据处理脚本（实时监控 + 可视化）
实时读取正在写入的CSV文件（由bluetooth_scanner写入），计算运动状态和聚类，
并输出JSON供Pygame可视化，同时显示实时聚类结果的散点图。
"""

import os
import time
import json
import numpy as np
from collections import deque
from datetime import datetime
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

# ==================== 可自定义的配置参数 ====================
# 1. 输入文件：当日CSV文件路径（与扫描器输出保持一致）
today = datetime.now().strftime("%Y-%m-%d")
CSV_PATH = f"/home/sghwr/USELESS/code/bt_csi_project/data/csv/devices_{today}.csv"

# 2. 输出文件（固定）：供Pygame实时读取的JSON文件路径
OUTPUT_JSON = "/home/sghwr/USELESS/code/bt_csi_project/json/pygame_data.json"

# 3. 输出文件（每日备份）：带日期的JSON备份文件路径（可选）
OUTPUT_JSON_DAILY = f"/home/sghwr/USELESS/code/bt_csi_project/backup/pygame_data_{today}.json"

# 算法参数（可根据需要调整）
MAX_HISTORY = 10                     # 每个设备保留的最大RSSI历史记录数
MOTION_VAR_THRESHOLD = 5.0           # 运动状态方差阈值
MIN_HISTORY_FOR_CLUSTER = 5          # 参与聚类所需的最小历史记录数
CLUSTER_EPS = 0.8                    # DBSCAN邻域半径
CLUSTER_MIN_SAMPLES = 2              # 最小簇样本数
CLUSTER_INTERVAL = 3                  # 每几个batch聚类一次
USE_DIFF_FEATURE = True               # 是否使用差分序列作为特征
POLL_INTERVAL = 0.5                   # 检查文件新内容的间隔（秒）
# =========================================================


class DeviceProcessor:
    """设备状态处理器，维护设备字典、运动状态和聚类"""

    def __init__(self, config):
        self.config = config
        self.devices = {}          # mac -> 设备信息
        self.batch_counter = 0      # 已处理的batch计数（用于控制聚类频率）

    def update_device(self, mac, rssi, batch_id, name=None):
        """更新设备的历史RSSI队列，并计算运动状态"""
        if mac not in self.devices:
            self.devices[mac] = {
                'name': name if name else mac[-8:],
                'rssi_history': deque(maxlen=self.config['MAX_HISTORY']),
                'last_batch': batch_id,
                'motion_state': 'static',
                'cluster': -1
            }
        dev = self.devices[mac]
        dev['last_batch'] = batch_id
        dev['rssi_history'].append(rssi)

        # 计算运动状态（至少需要2个样本）
        if len(dev['rssi_history']) >= 2:
            var = np.var(dev['rssi_history'])
            dev['motion_state'] = 'moving' if var > self.config['MOTION_VAR_THRESHOLD'] else 'static'

    def _extract_features(self):
        """内部方法：提取用于聚类的特征矩阵和MAC列表"""
        macs = []
        features = []
        for mac, dev in self.devices.items():
            hist = list(dev['rssi_history'])
            if len(hist) >= self.config['MIN_HISTORY_FOR_CLUSTER']:
                recent = hist[-self.config['MIN_HISTORY_FOR_CLUSTER']:]
                if self.config['USE_DIFF_FEATURE']:
                    feat = np.diff(recent)   # 差分后长度 = MIN_HISTORY_FOR_CLUSTER - 1
                else:
                    feat = recent
                features.append(feat)
                macs.append(mac)
        return macs, np.array(features) if features else None

    def run_clustering(self):
        """执行DBSCAN聚类，更新设备的cluster标签"""
        macs, X = self._extract_features()
        if X is None or len(macs) < 2:
            return
        clustering = DBSCAN(eps=self.config['CLUSTER_EPS'],
                            min_samples=self.config['CLUSTER_MIN_SAMPLES']).fit(X)
        labels = clustering.labels_
        for mac, label in zip(macs, labels):
            self.devices[mac]['cluster'] = int(label)

    def get_plot_data(self):
        """
        返回用于绘图的MAC列表、降维后的二维特征、对应的聚类标签
        """
        macs = []
        features = []
        for mac, dev in self.devices.items():
            hist = list(dev['rssi_history'])
            if len(hist) >= self.config['MIN_HISTORY_FOR_CLUSTER']:
                recent = hist[-self.config['MIN_HISTORY_FOR_CLUSTER']:]
                if self.config['USE_DIFF_FEATURE']:
                    feat = np.diff(recent)
                else:
                    feat = recent
                features.append(feat)
                macs.append(mac)
        if not features:
            return [], np.array([]), []
        X = np.array(features)

        # 降维至2D用于显示
        if X.shape[1] > 2:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            X_2d = pca.fit_transform(X)
        elif X.shape[1] == 1:
            X_2d = np.hstack([X, np.zeros((X.shape[0], 1))])
        else:
            X_2d = X[:, :2]

        labels = [self.devices[mac]['cluster'] for mac in macs]
        return macs, X_2d, labels

    def get_devices_status(self):
        """获取所有设备的当前状态列表（用于导出JSON）"""
        result = []
        for mac, dev in self.devices.items():
            if not dev['rssi_history']:
                continue
            rssi = dev['rssi_history'][-1]
            motion = 1 if dev['motion_state'] == 'moving' else 0
            result.append({
                'mac': mac,
                'rssi': rssi,
                'motion': motion,
                'cluster': dev.get('cluster', -1)
            })
        return result

    def export_to_json(self):
        """将当前设备状态写入JSON文件（固定文件名 + 每日备份）"""
        data = self.get_devices_status()
        # 确保输出目录存在
        os.makedirs(os.path.dirname(self.config['OUTPUT_JSON']), exist_ok=True)
        os.makedirs(os.path.dirname(self.config['OUTPUT_JSON_DAILY']), exist_ok=True)

        with open(self.config['OUTPUT_JSON'], 'w') as f:
            json.dump(data, f, indent=2)
        with open(self.config['OUTPUT_JSON_DAILY'], 'w') as f:
            json.dump(data, f, indent=2)
        print(f"[{time.strftime('%H:%M:%S')}] 已导出 {len(data)} 个设备到 {self.config['OUTPUT_JSON']}")


def follow_file(filepath):
    """生成器：类似 'tail -f'，实时读取文件新增的行"""
    # 等待文件创建
    while not os.path.exists(filepath):
        time.sleep(1)
    with open(filepath, 'r') as f:
        # 移动到文件末尾
        f.seek(0, os.SEEK_END)
        while True:
            line = f.readline()
            if not line:
                time.sleep(POLL_INTERVAL)
                continue
            yield line.strip()


def main():
    # 初始化配置字典
    config = {
        'MAX_HISTORY': MAX_HISTORY,
        'MOTION_VAR_THRESHOLD': MOTION_VAR_THRESHOLD,
        'MIN_HISTORY_FOR_CLUSTER': MIN_HISTORY_FOR_CLUSTER,
        'CLUSTER_EPS': CLUSTER_EPS,
        'CLUSTER_MIN_SAMPLES': CLUSTER_MIN_SAMPLES,
        'USE_DIFF_FEATURE': USE_DIFF_FEATURE,
        'OUTPUT_JSON': OUTPUT_JSON,
        'OUTPUT_JSON_DAILY': OUTPUT_JSON_DAILY
    }
    processor = DeviceProcessor(config)

    # 确保目录存在
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    os.makedirs(os.path.dirname(OUTPUT_JSON_DAILY), exist_ok=True)

    print(f"等待 CSV 文件: {CSV_PATH}")
    lines = follow_file(CSV_PATH)
    print("开始监控文件...")

    # ---------- 初始化图形 ----------
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter([], [], c=[], cmap='tab10', s=50)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title(f'实时聚类结果 (每 {CLUSTER_INTERVAL} batch 更新)')
    ax.grid(True)

    # 用于鼠标悬停显示的标注
    annot = ax.annotate("", xy=(0,0), xytext=(5,5), textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="w"),
                        arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)

    # 存储当前绘图数据，供悬停回调使用
    plot_macs = []
    plot_features_2d = np.array([])

    def on_hover(event):
        nonlocal plot_macs, plot_features_2d
        if event.inaxes != ax:
            return
        cont, ind = scatter.contains(event)
        if cont:
            idx = ind['ind'][0]
            if idx < len(plot_macs):
                pos = scatter.get_offsets()[idx]
                mac = plot_macs[idx]
                name = processor.devices[mac]['name']
                label = processor.devices[mac]['cluster']
                annot.xy = pos
                annot.set_text(f"{name}\n{mac}\n簇: {label}")
                annot.set_visible(True)
                fig.canvas.draw_idle()
        else:
            if annot.get_visible():
                annot.set_visible(False)
                fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", on_hover)
    # ---------------------------------

    current_batch = None
    try:
        for line in lines:
            if not line:
                continue
            parts = line.split(',')
            if len(parts) < 5:
                continue  # 格式不对，跳过
            # 假设顺序: timestamp, batch, mac, name, rssi
            try:
                # timestamp = parts[0].strip()   # 暂不使用
                batch = int(parts[1].strip())
                mac = parts[2].strip()
                name = parts[3].strip() if len(parts) > 3 and parts[3].strip() else None
                rssi = int(parts[4].strip())
            except (ValueError, IndexError) as e:
                print(f"解析行失败: {line}, 错误: {e}")
                continue

            # 如果遇到新的batch，触发聚类检查
            if current_batch is not None and batch != current_batch:
                processor.batch_counter += 1
                if processor.batch_counter % CLUSTER_INTERVAL == 0:
                    processor.run_clustering()
                    processor.export_to_json()

                    # 更新图形数据
                    plot_macs, plot_features_2d, plot_labels = processor.get_plot_data()
                    if len(plot_macs) > 0:
                        scatter.set_offsets(plot_features_2d)
                        scatter.set_array(np.array(plot_labels))
                        # 调整坐标轴范围
                        xmin, xmax = plot_features_2d[:,0].min(), plot_features_2d[:,0].max()
                        ymin, ymax = plot_features_2d[:,1].min(), plot_features_2d[:,1].max()
                        margin = 0.2
                        ax.set_xlim(xmin - margin, xmax + margin)
                        ax.set_ylim(ymin - margin, ymax + margin)
                    else:
                        scatter.set_offsets(np.empty((0, 2)))  # 清空散点图
                    fig.canvas.draw_idle()

            current_batch = batch
            processor.update_device(mac, rssi, batch, name)

            # 让GUI处理事件（必须，否则窗口会无响应）
            plt.pause(0.01)

    except KeyboardInterrupt:
        print("\n用户中断，退出。")
    finally:
        plt.ioff()
        plt.close('all')


if __name__ == "__main__":
    main()