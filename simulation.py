import csv
import time
from datetime import datetime

def simulate_realtime_data(csv_file, callback, speed=1.0):
    """
    模拟实时数据流
    :param csv_file: CSV文件路径
    :param callback: 数据处理回调函数
    :param speed: 时间加速倍数（1.0表示实时，2.0表示2倍速）
    """
    # 读取CSV数据并转换为字典列表
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        data = [row for row in reader]

    prev_time = 0.0
    for i, row in enumerate(data):
        # 转换数据类型
        current_time = float(row['time'])
        values = {k: float(v) for k, v in row.items()}

        # 计算时间间隔（根据实际时间差）
        if i > 0:
            delay = (current_time - prev_time) / speed
            time.sleep(delay)

        # 调用回调函数处理数据（可自定义）
        callback(values)

        prev_time = current_time

if __name__ == "__main__":
    # 示例回调函数 - 可替换为实际处理逻辑
    def print_data(data):
        """简单的打印输出"""
        print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] Received: "
              f"Time={data['time']:.2f}s, "
              f"F_in={data['F_in']:.5f}, "
              f"Displacement={data['Displacement']:.5f}")

    # 开始模拟（参数说明：）
    # friction.csv - 数据文件
    # print_data - 数据处理函数
    # speed=1.0 - 实时速度
    simulate_realtime_data('friction.csv', print_data, speed=1.0)