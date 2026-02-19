#!/usr/bin/env python3
"""
简单的文件输出模块 - 将设备数据写入 CSV 文件
不依赖接口，直接实现
"""

import csv
import os
from datetime import datetime

class FileOutput:
    """
    简单的文件输出类，将设备数据追加到 CSV 文件
    每天生成一个文件，格式：devices_YYYY-MM-DD.csv
    """

    def __init__(self, base_dir='~/USELESS/code/bt_csi_project/data/csv'):
        """
        初始化文件输出

        参数:
            base_dir: 数据存储根目录，默认为用户目录下的 bt_csi_project/data
        """
        self.base_dir = os.path.expanduser(base_dir)
        self.ensure_directory()

    def ensure_directory(self):
        """确保数据目录存在"""
        os.makedirs(self.base_dir, exist_ok=True)

    def _get_filepath(self):
        """根据当前日期生成文件路径"""
        today = datetime.now().strftime('%Y-%m-%d')
        return os.path.join(self.base_dir, f"devices_{today}.csv")

    def write(self, devices, batch_number):
        """
        将一批设备数据写入 CSV 文件

        参数:
            devices: 设备列表，每个设备包含 'mac', 'name', 'rssi' 字段
            batch_number: 扫描批次号
        """
        if not devices:
            return

        filepath = self._get_filepath()
        file_exists = os.path.isfile(filepath)

        # 以追加模式打开文件
        with open(filepath, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            # 如果是新文件，先写入表头
            if not file_exists:
                writer.writerow(['timestamp', 'batch', 'mac', 'name', 'rssi'])

            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            for dev in devices:
                writer.writerow([
                    timestamp,
                    batch_number,
                    dev['mac'],
                    dev.get('name', 'unknown'),
                    dev.get('rssi', '')
                ])

    def close(self):
        """不需要关闭文件，因为每次 write 都会自动关闭（保留此方法便于后续扩展）"""
        # 方法体仅包含文档字符串，没有执行语句，符合 Python 语法
        # 如果需要实际关闭操作，可在此添加
        pass