#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
旧版本，请勿使用
旧版本，请勿使用
旧版本，请勿使用
"""
import os
import time
import json
import pandas as pd
import numpy as np
from collections import deque
from sklearn.cluster import DBSCAN

# ==================== 配置参数 ====================
CSV_PATH = "bluetooth_data.csv"          # 扫描程序实时写入的CSV文件路径
OUTPUT_JSON = "pygame_data.json"         # 输出给Pygame的JSON文件路径
MAX_HISTORY = 10                          # 每个设备保存的最大历史记录数
MOTION_VAR_THRESHOLD = 5.0                # 运动状态方差阈值
MIN_HISTORY_FOR_CLUSTER = 5                # 参与聚类所需的最少历史记录数
CLUSTER_EPS = 0.8                          # DBSCAN邻域半径
CLUSTER_MIN_SAMPLES = 2                    # 最小簇样本数
CLUSTER_INTERVAL = 3                        # 每几个batch聚类一次（设为1表示每个batch都聚类）
USE_DIFF_FEATURE = True                     # 使用差分序列
POLL_INTERVAL = 1.0                         # 检查文件新内容的间隔（秒）
# =================================================

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

    def get_devices_status(self):
        """获取所有设备的当前状态列表（用于导出）"""
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
        """将当前设备状态写入JSON文件（供Pygame读取）"""
        data = self.get_devices_status()
        with open(self.config['OUTPUT_JSON'], 'w') as f:
            json.dump(data, f, indent=2)
        print(f"[{time.strftime('%H:%M:%S')}] 已导出 {len(data)} 个设备到 {self.config['OUTPUT_JSON']}")

def follow_file(filepath):
    """生成器：类似 'tail -f'，实时读取文件新增的行"""
    with open(filepath, 'r') as f:
        # 移动到文件末尾
        f.seek(0, os.SEEK_END)
        while True:
            line = f.readline()
            if not line:
                time.sleep(POLL_INTERVAL)
                continue
            yield line

def main():
    # 初始化处理器
    config = {
        'MAX_HISTORY': MAX_HISTORY,
        'MOTION_VAR_THRESHOLD': MOTION_VAR_THRESHOLD,
        'MIN_HISTORY_FOR_CLUSTER': MIN_HISTORY_FOR_CLUSTER,
        'CLUSTER_EPS': CLUSTER_EPS,
        'CLUSTER_MIN_SAMPLES': CLUSTER_MIN_SAMPLES,
        'USE_DIFF_FEATURE': USE_DIFF_FEATURE,
        'OUTPUT_JSON': OUTPUT_JSON
    }
    processor = DeviceProcessor(config)

    # 检查CSV文件是否存在，若不存在则等待
    if not os.path.exists(CSV_PATH):
        print(f"等待文件 {CSV_PATH} 创建...")
        while not os.path.exists(CSV_PATH):
            time.sleep(1)

    print(f"开始监控文件: {CSV_PATH}")
    lines = follow_file(CSV_PATH)

    # 读取CSV的列名（假设第一行是标题）
    # 由于文件可能正在写入，我们需要先读取一次标题行
    # 简单处理：假设文件已存在且有标题，我们读取第一行作为列名
    with open(CSV_PATH, 'r') as f:
        first_line = f.readline().strip()
        header = first_line.split(',')
        # 期望列: timestamp, batch, mac, name, rssi (可能name为空)
    print(f"CSV列名: {header}")

    current_batch = None
    batch_rows = []  # 缓存当前batch的所有行（如果需要按batch整体处理，但这里逐行更新也可）

    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split(',')
        if len(parts) < 5:
            continue  # 格式不对，跳过
        # 假设顺序: timestamp, batch, mac, name, rssi
        try:
            timestamp = parts[0].strip()
            batch = int(parts[1].strip())
            mac = parts[2].strip()
            name = parts[3].strip() if len(parts) > 3 and parts[3].strip() else None
            rssi = int(parts[4].strip())
        except (ValueError, IndexError) as e:
            print(f"解析行失败: {line}, 错误: {e}")
            continue

        # 如果遇到新的batch，且之前有batch处理完，则可以考虑触发聚类（可选）
        if current_batch is not None and batch != current_batch:
            # 上一个batch结束，可以执行一些操作（例如增加batch计数）
            processor.batch_counter += 1
            if processor.batch_counter % CLUSTER_INTERVAL == 0:
                processor.run_clustering()
                processor.export_to_json()
        current_batch = batch

        # 更新设备
        processor.update_device(mac, rssi, batch, name)

    # 注意：以上循环会一直运行，因为follow_file是无限的

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n用户中断，退出。")