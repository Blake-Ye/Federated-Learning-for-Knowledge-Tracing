import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_metrics(directory):
    assist2009_files = []
    assist2015_files = []

    # 遍历目录下所有 CSV 文件，将文件分为两类
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            if 'ASSIST2009' in filename:
                assist2009_files.append(filename)
            elif 'ASSIST2015' in filename:
                assist2015_files.append(filename)

    # 创建字典来存储不同模型的数据（包含数据集信息）
    loss_data_2009 = {}
    accuracy_data_2009 = {}
    auc_data_2009 = {}

    loss_data_2015 = {}
    accuracy_data_2015 = {}
    auc_data_2015 = {}

    # 处理 ASSIST2009 数据集文件
    for filename in assist2009_files:
        file_path = os.path.join(directory, filename)
        try:
            df = pd.read_csv(file_path)
            if 'round' in df.columns:
                is_round = True
            elif 'epoch' in df.columns:
                is_round = False
            else:
                continue  # 如果没有 round 或 epoch，跳过该文件

            model_name = filename.split('_')[0]  # 获取模型名称
            if 'round' in df.columns:
                loss_data_2009[model_name] = df['loss']
                accuracy_data_2009[model_name] = df['accuracy']
                auc_data_2009[model_name] = df['auc']
            else:  # 处理 epoch 数据
                loss_data_2009[model_name] = df['loss']
                accuracy_data_2009[model_name] = df['accuracy']
                auc_data_2009[model_name] = df['auc']

        except Exception as e:
            print(f"Error reading {filename}: {e}")

    # 处理 ASSIST2015 数据集文件
    for filename in assist2015_files:
        file_path = os.path.join(directory, filename)
        try:
            df = pd.read_csv(file_path)
            if 'round' in df.columns:
                is_round = True
            elif 'epoch' in df.columns:
                is_round = False
            else:
                continue  # 如果没有 round 或 epoch，跳过该文件

            model_name = filename.split('_')[0]  # 获取模型名称
            if 'round' in df.columns:
                loss_data_2015[model_name] = df['loss']
                accuracy_data_2015[model_name] = df['accuracy']
                auc_data_2015[model_name] = df['auc']
            else:  # 处理 epoch 数据
                loss_data_2015[model_name] = df['loss']
                accuracy_data_2015[model_name] = df['accuracy']
                auc_data_2015[model_name] = df['auc']

        except Exception as e:
            print(f"Error reading {filename}: {e}")

    def save_plot(data, y_label, title, file_name, folder_path):
        plt.figure(figsize=(8, 6))
        for model_name, values in data.items():
            plt.plot(df['round'] if 'round' in df.columns else df['epoch'], values, label=f'{model_name}')
        plt.xlabel('Round' if 'round' in df.columns else 'Epoch')
        plt.ylabel(y_label)
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        os.makedirs(folder_path, exist_ok=True)
        plt.savefig(os.path.join(folder_path, f'{file_name}.png'))
        plt.close()

    # 绘制 ASSIST2009 数据集的图像
    if loss_data_2009:  # 确保有数据可绘制
        if 'round' in df.columns:
            folder_path = os.path.join(directory, 'Federated Learning', 'ASSIST2009')
        else:
            folder_path = os.path.join(directory, 'Non-Federated', 'ASSIST2009')

        save_plot(loss_data_2009, 'Loss', 'ASSIST2009 Loss Comparison', 'loss', folder_path)
        save_plot(accuracy_data_2009, 'Accuracy', 'ASSIST2009 Accuracy Comparison', 'accuracy', folder_path)
        save_plot(auc_data_2009, 'AUC', 'ASSIST2009 AUC Comparison', 'auc', folder_path)
    else:
        print("No data for ASSIST2009 to plot.")

    # 绘制 ASSIST2015 数据集的图像
    if loss_data_2015:  # 确保有数据可绘制
        if 'round' in df.columns:
            folder_path = os.path.join(directory, 'Federated Learning', 'ASSIST2015')
        else:
            folder_path = os.path.join(directory, 'Non-Federated', 'ASSIST2015')

        save_plot(loss_data_2015, 'Loss', 'ASSIST2015 Loss Comparison', 'loss', folder_path)
        save_plot(accuracy_data_2015, 'Accuracy', 'ASSIST2015 Accuracy Comparison', 'accuracy', folder_path)
        save_plot(auc_data_2015, 'AUC', 'ASSIST2015 AUC Comparison', 'auc', folder_path)
    else:
        print("No data for ASSIST2015 to plot.")


# 调用函数，指定目录路径
directory_path = 'non_federated_learning'  # 替换为你的文件夹路径
plot_metrics(directory_path)
