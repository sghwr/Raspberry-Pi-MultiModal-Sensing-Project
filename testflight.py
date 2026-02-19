import threading
# 完全删掉signal相关代码！靠Python默认的KeyboardInterrupt终止
from bluetooth_scanner.run_scanner import run
from ble_cluster import cluster_main

def main():
    # 1. 创建守护线程（主线程死则子线程死）
    t_scan = threading.Thread(target=run, args=(1, 20, 60), daemon=True)
    t_cluster = threading.Thread(target=cluster_main.cluster, daemon=True)
    
    # 2. 启动双线程（并行运行）
    print("✅ 扫描+聚类已启动 | 按 Ctrl+C 立即停止")
    t_scan.start()
    t_cluster.start()
    
    # 3. 主线程阻塞（仅捕获Ctrl+C，无其他逻辑）
    try:
        # 用join()等待所有子线程，但子线程是无限运行的，所以实际是等Ctrl+C
        # 多个子线程用循环join，避免主线程被单个线程绑定
        for t in [t_scan, t_cluster]:
            t.join()
    except KeyboardInterrupt:
        # 捕获Ctrl+C，直接打印提示后退出（主线程退→守护线程全退）
        print("\n⚠️  已收到停止指令 | 程序已终止")

if __name__ == "__main__":
    main()