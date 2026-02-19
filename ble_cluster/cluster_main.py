#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
在线蓝牙数据处理脚本
实时读取正在写入的CSV文件（由bluetooth_scanner写入），计算运动状态和聚类，
并输出JSON供Pygame可视化。
"""

import pandas as pd
import numpy as np
from collections import deque
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import os
import json
from datetime import datetime

# ==================== 可自定义的配置参数 ====================
# 1. 输入文件：当日CSV文件路径（支持动态日期，也可改为固定文件名）
today = datetime.now().strftime("%Y-%m-%d")                # 例如 "2026-02-19"
CSV_PATH = f"/home/sghwr/USELESS/code/bt_csi_project/csv/devices_{today}.csv"                          # <--- 在这里修改CSV文件路径

# 2. 输出文件（固定）：供Pygame实时读取的JSON文件路径
OUTPUT_JSON = "/home/sghwr/USELESS/code/bt_csi_project/json/pygame_data.json"                           # <--- 在这里修改实时输出JSON路径

# 3. 输出文件（每日备份）：带日期的JSON备份文件路径（可选）
OUTPUT_JSON_DAILY = f"/home/sghwr/USELESS/code/bt_csi_project/backup/pygame_data_{today}.json"            # <--- 在这里修改备份JSON路径

# 算法参数（可根据需要调整）
MAX_HISTORY = 10
MOTION_VAR_THRESHOLD = 5.0
MIN_HISTORY_FOR_CLUSTER = 5
CLUSTER_EPS = 0.8
CLUSTER_MIN_SAMPLES = 2
CLUSTER_INTERVAL = 3
USE_DIFF_FEATURE = True
USE_PCA = True
POLL_INTERVAL = 1.0
# =========================================================

# 设备字典：mac -> 设备信息
devices = {}

# 用于绘图的数据缓存
plot_macs = []
plot_features_2d = []   # 降维后的二维坐标
plot_labels = []        # 聚类标签

# 初始化绘图
fig, ax = plt.subplots(figsize=(10, 8))
scatter = ax.scatter([], [], c=[], cmap='tab10', s=50)
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_title('实时聚类结果 (每 {} batch 更新)'.format(CLUSTER_INTERVAL))
ax.grid(True)

# 用于文本标注
annot = ax.annotate("", xy=(0,0), xytext=(5,5), textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w"),
                    arrowprops=dict(arrowstyle="->"))
annot.set_visible(False)

def update_device(mac, rssi, batch_id, name=None):
    """更新设备的历史RSSI队列，并计算运动状态"""
    if mac not in devices:
        devices[mac] = {
            'name': name if name else mac[-8:],
            'rssi_history': deque(maxlen=MAX_HISTORY),
            'last_batch': batch_id,
            'motion_state': 'static',
            'cluster': -1
        }
    dev = devices[mac]
    dev['last_batch'] = batch_id
    dev['rssi_history'].append(rssi)

    # 计算运动状态（至少需要2个样本）
    if len(dev['rssi_history']) >= 2:
        var = np.var(dev['rssi_history'])
        dev['motion_state'] = 'moving' if var > MOTION_VAR_THRESHOLD else 'static'

def extract_features():
    """从设备字典中提取特征矩阵和对应的MAC列表"""
    macs = []
    features = []
    for mac, dev in devices.items():
        hist = list(dev['rssi_history'])
        if len(hist) >= MIN_HISTORY_FOR_CLUSTER:
            recent = hist[-MIN_HISTORY_FOR_CLUSTER:]
            if USE_DIFF_FEATURE:
                feat = np.diff(recent)   # 差分序列
            else:
                feat = recent
            features.append(feat)
            macs.append(mac)
    return macs, np.array(features)

def run_clustering():
    """执行DBSCAN聚类，并更新设备标签"""
    global plot_macs, plot_features_2d, plot_labels
    macs, X = extract_features()
    if len(macs) < 2:
        plot_macs = macs
        plot_features_2d = X if len(X) > 0 else np.array([])
        plot_labels = [-1] * len(macs)
        return

    clustering = DBSCAN(eps=CLUSTER_EPS, min_samples=CLUSTER_MIN_SAMPLES).fit(X)
    labels = clustering.labels_

    for mac, label in zip(macs, labels):
        devices[mac]['cluster'] = int(label)

    plot_macs = macs
    if USE_PCA and X.shape[1] > 2:
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X)
    else:
        if X.shape[1] == 1:
            X_2d = np.hstack([X, np.zeros((X.shape[0], 1))])
        elif X.shape[1] >= 2:
            X_2d = X[:, :2]
        else:
            X_2d = X
    plot_features_2d = X_2d
    plot_labels = labels

def update_plot(frame):
    """动画更新函数"""
    if len(plot_macs) == 0:
        return scatter,
    scatter.set_offsets(plot_features_2d)
    scatter.set_array(np.array(plot_labels))
    if len(plot_features_2d) > 0:
        xmin, xmax = plot_features_2d[:,0].min(), plot_features_2d[:,0].max()
        ymin, ymax = plot_features_2d[:,1].min(), plot_features_2d[:,1].max()
        margin = 0.2
        ax.set_xlim(xmin - margin, xmax + margin)
        ax.set_ylim(ymin - margin, ymax + margin)
    return scatter,

def on_hover(event):
    """鼠标悬停显示设备信息"""
    if event.inaxes != ax:
        return
    cont, ind = scatter.contains(event)
    if cont:
        idx = ind['ind'][0]
        pos = scatter.get_offsets()[idx]
        mac = plot_macs[idx]
        label = plot_labels[idx]
        name = devices[mac]['name']
        annot.xy = pos
        text = f"{name}\n{mac}\n簇: {label}"
        annot.set_text(text)
        annot.set_visible(True)
        fig.canvas.draw_idle()
    else:
        if annot.get_visible():
            annot.set_visible(False)
            fig.canvas.draw_idle()

def get_devices_status():
    """获取所有设备的当前状态（用于导出JSON）"""
    result = []
    for mac, dev in devices.items():
        if not dev['rssi_history']:
            continue
        motion = 1 if dev['motion_state'] == 'moving' else 0
        rssi = dev['rssi_history'][-1]
        result.append({
            'mac': mac,
            'rssi': rssi,
            'motion': motion,
            'cluster': dev.get('cluster', -1)
        })
    return result

def export_to_json():
    """将当前设备状态写入JSON文件（固定文件名 + 每日备份）"""
    data = get_devices_status()
    # 写入固定文件供Pygame实时读取
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(data, f, indent=2)
    # 同时写入带日期的备份文件
    with open(OUTPUT_JSON_DAILY, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"[{time.strftime('%H:%M:%S')}] 已导出 {len(data)} 个设备到 {OUTPUT_JSON} 和 {OUTPUT_JSON_DAILY}")

def cluster():
    # 检查当日CSV文件是否存在，若不存在则等待
    if not os.path.exists(CSV_PATH):
        print(f"等待当日文件 {CSV_PATH} 创建...")
        while not os.path.exists(CSV_PATH):
            time.sleep(1)

    print(f"开始读取文件: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    # 假设CSV有列：batch, mac, rssi, name（可选）
    batches = df.groupby('batch')

    plt.ion()
    fig.canvas.mpl_connect("motion_notify_event", on_hover)

    batch_counter = 0
    for batch_id, group in batches:
        for _, row in group.iterrows():
            mac = row['mac']
            rssi = row['rssi']
            name = row.get('name', None)
            update_device(mac, rssi, batch_id, name)

        batch_counter += 1
        if batch_counter % CLUSTER_INTERVAL == 0:
            run_clustering()
            export_to_json()                     # 每次聚类后导出JSON
            update_plot(None)
            plt.pause(0.1)
            print(f"Batch {batch_id} 处理完成，当前活动设备数: {len(devices)}")

    print("所有数据处理完毕，保持窗口显示...")
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    try:
        cluster()
    except KeyboardInterrupt:
        print("\n用户中断，退出。")