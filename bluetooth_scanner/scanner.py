#!/usr/bin/env python3
"""
蓝牙扫描器核心模块（使用 Bleak 库）
负责：扫描蓝牙设备、管理设备生命周期、提供数据接口
"""

import asyncio
import time
import logging
import os
import threading
from collections import deque, OrderedDict
from logging.handlers import RotatingFileHandler
from bleak import BleakScanner

#from config import CONFIG

class BluetoothScanner:
    """蓝牙扫描器主类"""

    def __init__(self, scan_interval , history_length, timeout):
        """
        初始化扫描器

        参数:
            scan_interval: 扫描间隔（秒）
            history_length: 每个设备保留的RSSI历史记录数
            timeout: 设备超过此秒数未出现即视为离开
        """
        self.scan_interval = scan_interval
        self.history_length = history_length
        self.timeout = timeout

        # 设备存储字典
        self.devices = OrderedDict()

        # 线程锁
        self.lock = threading.Lock()

        # 运行控制标志
        self.running = False

        # 统计信息
        self.total_scans = 0
        self.total_devices_found = 0

        # 配置日志系统
        self.setup_logging()

        self.logger.info("蓝牙扫描器初始化完成")
        self.logger.info(f"配置: 扫描间隔={scan_interval}秒, 历史记录数={history_length}, 超时={timeout}秒")

    def setup_logging(self):
        """配置日志系统"""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.DEBUG)

        if self.logger.handlers:
            return

        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # 文件处理器
        log_dir = os.path.expanduser('~/bt_csi_project/logs')
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, 'scanner.log')
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=1024*1024,
            backupCount=3,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        # 错误日志专用文件
        error_handler = RotatingFileHandler(
            os.path.join(log_dir, 'error.log'),
            maxBytes=1024*1024,
            backupCount=2,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        self.logger.addHandler(error_handler)

        self.logger.debug("日志系统初始化完成")

    def scan_once(self):
        """
        执行一次扫描，使用 Bleak 获取设备 MAC、名称和 RSSI
        """
        self.logger.debug("开始一次扫描 (Bleak)...")

        async def scan():
            # 使用 return_adv=True 来获取广播数据
            devices = await BleakScanner.discover(timeout=1, return_adv=True)
            result = []
            for addr, (device, adv_data) in devices.items():
                # device 是 BLEDevice 对象，adv_data 是 AdvertisementData 对象
                # RSSI 在 adv_data.rssi 中
                result.append({
                    'mac': addr.upper(),
                    'name': device.name,          # 可能为 None
                    'rssi': adv_data.rssi         # 从广播数据中获取 RSSI
                })
            return result

        try:
            devices = asyncio.run(scan())
            self.logger.info(f"扫描完成，发现 {len(devices)} 个设备")
            self.total_scans += 1
            self.total_devices_found += len(devices)
            return devices
        except Exception as e:
            self.logger.error(f"扫描过程发生异常: {e}", exc_info=True)
            return []

    def update_device(self, device_info):
        """
        更新设备信息

        参数:
            device_info: 字典，包含 'mac', 'name', 'rssi'
        """
        mac = device_info['mac']
        name = device_info['name']
        rssi = device_info['rssi']
        now = time.time()

        with self.lock:
            if mac not in self.devices:
                # 新设备
                self.logger.info(f"发现新设备: {mac} 名称:{name} RSSI:{rssi}")
                self.devices[mac] = {
                    'name': name,
                    'first_seen': now,
                    'last_seen': now,
                    'rssi_history': deque(maxlen=self.history_length),
                    'appearance_count': 1,
                    'rssi_avg': rssi,
                    'rssi_std': 0.0
                }
                self.devices[mac]['rssi_history'].append(rssi)
            else:
                # 已有设备
                old = self.devices[mac]
                old['last_seen'] = now
                old['appearance_count'] += 1
                # 更新名称（如果新名称不为空且与之前不同）
                if name and name != old['name']:
                    self.logger.info(f"设备 {mac} 名称更新: {old['name']} -> {name}")
                    old['name'] = name
                # 添加 RSSI 到历史
                old['rssi_history'].append(rssi)
                # 更新统计信息
                if len(old['rssi_history']) > 1:
                    avg, std = self.calculate_stats(old['rssi_history'])
                    old['rssi_avg'] = avg
                    old['rssi_std'] = std
                self.logger.debug(f"更新设备 {mac}，当前RSSI:{rssi}，历史长度:{len(old['rssi_history'])}")

    def calculate_stats(self, rssi_history):
        """
        计算 RSSI 历史列表的平均值和标准差
        """
        if len(rssi_history) == 0:
            return 0.0, 0.0
        n = len(rssi_history)
        avg = sum(rssi_history) / n
        variance = sum((x - avg) ** 2 for x in rssi_history) / n
        std = variance ** 0.5
        return avg, std

    def cleanup_old_devices(self):
        """
        清理超过 timeout 未出现的设备
        """
        now = time.time()
        expired = []
        with self.lock:
            for mac, info in list(self.devices.items()):
                if now - info['last_seen'] > self.timeout:
                    expired.append((mac, info['name']))
                    del self.devices[mac]

        for mac, name in expired:
            self.logger.info(f"设备过期移除: {mac} 名称:{name}")

    def scan_loop(self):
        """
        主循环：持续扫描
        """
        self.running = True
        self.logger.info("扫描循环启动")

        try:
            while self.running:
                devices = self.scan_once()
                for dev in devices:
                    self.update_device(dev)
                self.cleanup_old_devices()
                if self.running:
                    time.sleep(self.scan_interval)
        except KeyboardInterrupt:
            self.logger.info("收到中断信号，停止扫描")
        except Exception as e:
            self.logger.error(f"扫描循环异常: {e}", exc_info=True)
        finally:
            self.running = False
            self.logger.info("扫描循环结束")

    def get_devices(self):
        """
        获取当前所有设备信息（供Flask调用）
        """
        result = []
        with self.lock:
            now = time.time()
            for mac, info in self.devices.items():
                status = 'active' if (now - info['last_seen'] <= self.timeout) else 'inactive'
                latest_rssi = info['rssi_history'][-1] if info['rssi_history'] else None
                result.append({
                    'mac': mac,
                    'name': info['name'],
                    'first_seen': info['first_seen'],
                    'last_seen': info['last_seen'],
                    'rssi': latest_rssi,
                    'rssi_avg': info['rssi_avg'],
                    'rssi_std': info['rssi_std'],
                    'appearance_count': info['appearance_count'],
                    'status': status
                })
        result.sort(key=lambda x: x['last_seen'], reverse=True)
        return result

    def get_device_count(self):
        with self.lock:
            return len(self.devices)

    def get_summary(self):
        with self.lock:
            active_count = sum(1 for info in self.devices.values() if time.time() - info['last_seen'] <= self.timeout)
        return {
            'total_scans': self.total_scans,
            'total_devices_found': self.total_devices_found,
            'current_devices': len(self.devices),
            'active_devices': active_count
        }


# 测试
if __name__ == "__main__":
    scanner = BluetoothScanner()
    devices = scanner.scan_once()
    print(f"发现 {len(devices)} 个设备:")
    for d in devices:
        print(f"  MAC: {d['mac']}, 名称: {d['name']}, RSSI: {d['rssi']}")
    for d in devices:
        scanner.update_device(d)
    print("\n当前设备列表:")
    for d in scanner.get_devices():
        print(f"  {d['mac']} - {d['name']} - RSSI最新:{d['rssi']} - 状态:{d['status']}")
    print("\n摘要:", scanner.get_summary())