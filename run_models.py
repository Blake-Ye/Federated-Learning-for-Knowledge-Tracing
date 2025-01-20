import os
import time

# 文件路径列表
python_files = [
    "server_Krum.py",
    "server_FedProx.py",
]

# 执行所有文件
for file in python_files:
    if os.path.isfile(file):
        print(f"Executing {file}...")
        os.system(f"python {file}")
        print(f"Waiting for 1 minute before executing the next file...")
        time.sleep(60)  # 暂停 1 分钟
    else:
        print(f"{file} does not exist.")
