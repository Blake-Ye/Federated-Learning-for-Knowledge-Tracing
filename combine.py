from PIL import Image
import os

# 设置你的项目路径
parent_dir = 'federated_learning_FedTrimmedAvg'  # 修改为你的真实路径
base_dir = os.path.join(parent_dir, 'Federated Learning')

# 数据集名称与目标文件夹
dataset_dirs = ['ASSIST2009', 'ASSIST2015']
output_dir = os.path.join(base_dir, 'merged')
os.makedirs(output_dir, exist_ok=True)

# 图像名称列表
image_names = ['accuracy.png', 'auc.png', 'loss.png']

# 遍历每种图像
for image_name in image_names:
    image_paths = [os.path.join(base_dir, dataset, image_name) for dataset in dataset_dirs]

    # 加载并统一为 RGB 模式
    images = [Image.open(path).convert("RGB") for path in image_paths]

    # 统一宽度（缩放到最小宽度）并使用高质量插值算法
    min_width = min(img.width for img in images)
    resized_images = [
        img.resize(
            (min_width, int(img.height * min_width / img.width)),
            resample=Image.LANCZOS
        ) for img in images
    ]

    # 计算总高度
    total_height = sum(img.height for img in resized_images)
    merged_image = Image.new('RGB', (min_width, total_height))

    # 粘贴图像
    y_offset = 0
    for img in resized_images:
        merged_image.paste(img, (0, y_offset))
        y_offset += img.height

    # 保存合并后的图像
    output_path = os.path.join(output_dir, f'merged_vertical_{image_name}')
    merged_image.save(output_path, format='PNG', optimize=True)
    print(f"Saved: {output_path}")
