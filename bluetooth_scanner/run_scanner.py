#!/usr/bin/env python3
"""
扫描器启动脚本
负责：创建扫描器实例、注册文件输出、运行扫描循环、处理退出信号
"""

#import signal
import sys
import time
from .scanner import BluetoothScanner
from .storage import FileOutput
#from .config import CONFIG

# 全局变量，用于在信号处理函数中控制循环
running = True
scanner = None
file_output = None
batch_number = 0

'''def signal_handler(sig, frame):
    """处理 Ctrl+C 信号，优雅退出"""
    global running
    print("\n正在停止扫描...")
    running = False
'''
def run(interval=1, length=20, t=60):
    global scanner, file_output, running, batch_number

    # 设置信号处理
    #signal.signal(signal.SIGINT, signal_handler)

    # 创建扫描器和文件输出
    scanner = BluetoothScanner(interval, length, t)  
    file_output = FileOutput()    # 使用默认目录 ~/bt_csi_project/data

    print("蓝牙扫描器启动，按 Ctrl+C 停止")
    print(f"扫描间隔: {scanner.scan_interval}秒, 数据目录: {file_output.base_dir}")

    # 主循环
    while running:
        # 执行一次扫描
        devices = scanner.scan_once()

        # 如果有设备，写入文件
        if devices:
            batch_number += 1
            file_output.write(devices, batch_number)
            print(f"批次 {batch_number}: 发现 {len(devices)} 个设备，已保存")

        # 等待下一次扫描（检查 running 标志以便快速退出）
        for _ in range(scanner.scan_interval):
            if not running:
                break
            time.sleep(1)

    # 清理工作
    file_output.close()
    print("扫描器已停止，数据文件已关闭")
    sys.exit(0)

if __name__ == "__main__":
    run(1, 20, 60)