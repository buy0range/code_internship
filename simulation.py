import pandas as pd
import time

def simulate_realtime_data(data_path, time_column):
    # 读取数据
    data = pd.read_csv(data_path)
    df = data[['Time', 'coe','Absolute_Energy']]
    records = df.to_dict('records')
    relative_times = data[time_column].tolist()
    
    if len(records) == 0:
        print("数据为空，无法回放。")
        return
    
    # 计算时间间隔（第一个数据立即发送）
    start_time = time.time()
    
    for record in records:
        target_time = start_time + record[time_column]
        current_time = time.time()
        wait_time = target_time - current_time
        
        if wait_time > 0:
            time.sleep(wait_time)
        
        # 此处替换为实际的数据发送逻辑
        print(record)


simulate_realtime_data('original_data.csv', 'Time')