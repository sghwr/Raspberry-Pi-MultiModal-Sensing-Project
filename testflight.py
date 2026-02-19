#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
启动脚本：同时运行蓝牙扫描器（后台线程）和实时聚类GUI（主线程）
- 扫描器：将设备数据写入CSV
- 聚类：实时读取CSV，显示聚类散点图，并输出JSON供Pygame使用
"""

import threading
import sys
from bluetooth_scanner.run_scanner import run as scanner_run
from ble_cluster import cluster_main

def main():
    # 启动扫描器作为守护线程（主线程退出时自动终止）
    scan_thread = threading.Thread(
        target=scanner_run,
        args=(1, 20, 60),   # scan_interval, history_length, timeout
        daemon=True
    )
    scan_thread.start()
    print("✅ 蓝牙扫描器已启动（守护线程）")

    print("🔄 启动实时聚类 GUI（主线程）...")
    print("⏳ 等待 CSV 文件生成，图形窗口将自动打开")
    print("按 Ctrl+C 终止所有任务\n")

    try:
        # 聚类主函数运行在主线程，包含 matplotlib GUI 循环
        cluster_main.main()
    except KeyboardInterrupt:
        print("\n⚠️  收到中断信号，正在退出...")
    except Exception as e:
        print(f"\n❌ 聚类模块发生异常: {e}")
    finally:
        print("程序已终止。")

if __name__ == "__main__":
    main()