import os
import pandas as pd

def find_best_metrics(directory):
    # 遍历目录下所有 CSV 文件
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            file_path = os.path.join(directory, filename)
            try:
                # 读取 CSV 文件
                df = pd.read_csv(file_path)

                # 判断文件的列名，确定 epoch 列
                epoch_col = 'epoch' if 'epoch' in df.columns else 'round' if 'round' in df.columns else None

                if epoch_col and all(col in df.columns for col in [epoch_col, 'loss', 'accuracy', 'auc']):
                    # 找到 loss 最小的行
                    min_loss_row = df.loc[df['loss'].idxmin()]

                    # 找到 accuracy 和 auc 最大的行
                    max_accuracy_row = df.loc[df['accuracy'].idxmax()]
                    max_auc_row = df.loc[df['auc'].idxmax()]

                    # 输出文件名以及相应的最佳数据，保留原始列名
                    print(f"File: {filename}")
                    print(f"  - Best loss: {epoch_col} {min_loss_row[epoch_col]} | Loss: {min_loss_row['loss']}")
                    print(f"  - Best accuracy: {epoch_col} {max_accuracy_row[epoch_col]} | Accuracy: {max_accuracy_row['accuracy']}")
                    print(f"  - Best AUC: {epoch_col} {max_auc_row[epoch_col]} | AUC: {max_auc_row['auc']}")
                    print("="*50)
                else:
                    print(f"File {filename} does not have the expected columns.")
            except Exception as e:
                print(f"Error reading {filename}: {e}")

# 调用函数，指定目录路径
directory_path = 'federated_learning_Krum'  # 替换为你的文件夹路径
find_best_metrics(directory_path)
