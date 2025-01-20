import os
import argparse
import json
import pickle
import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import SGD, Adam
from data_loaders.assist2009 import ASSIST2009
from data_loaders.assist2015 import ASSIST2015
from data_loaders.algebra2005 import Algebra2005
from data_loaders.statics2011 import Statics2011
from models.dkt import DKT
from models.dkt_plus import DKTPlus
from models.dkvmn import DKVMN
from models.kqn import KQN
from models.sakt import SAKT
from models.gkt import PAM, MHA
from models.utils import collate_fn
import csv


# 创建保存训练过程评估结果的目录和文件
def create_results_csv(model_name, dataset_name):
    results_dir = "non_federated_learning"
    os.makedirs(results_dir, exist_ok=True)
    csv_file = os.path.join(results_dir, f"{model_name}_{dataset_name}_training.csv")

    # 创建CSV文件并写入表头
    if not os.path.exists(csv_file):
        with open(csv_file, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'loss', 'accuracy', 'auc'])

    return csv_file


def main(model_name, dataset_name):
    if not os.path.isdir("ckpts"):
        os.mkdir("ckpts")

    ckpt_path = os.path.join("ckpts", model_name)
    if not os.path.isdir(ckpt_path):
        os.mkdir(ckpt_path)

    ckpt_path = os.path.join(ckpt_path, dataset_name)
    if not os.path.isdir(ckpt_path):
        os.mkdir(ckpt_path)

    with open("config.json") as f:
        config = json.load(f)
        model_config = config[model_name]
        train_config = config["train_config"]

    batch_size = train_config["batch_size"]
    num_epochs = train_config["num_epochs"]
    train_ratio = train_config["train_ratio"]
    learning_rate = train_config["learning_rate"]
    optimizer = train_config["optimizer"]  # can be [sgd, adam]
    seq_len = train_config["seq_len"]

    if dataset_name == "ASSIST2009":
        dataset = ASSIST2009(seq_len)
    elif dataset_name == "ASSIST2015":
        dataset = ASSIST2015(seq_len)
    elif dataset_name == "Algebra2005":
        dataset = Algebra2005(seq_len)
    elif dataset_name == "Statics2011":
        dataset = Statics2011(seq_len)

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    with open(os.path.join(ckpt_path, "model_config.json"), "w") as f:
        json.dump(model_config, f, indent=4)
    with open(os.path.join(ckpt_path, "train_config.json"), "w") as f:
        json.dump(train_config, f, indent=4)

    # Initialize model
    if model_name == "dkt":
        model = DKT(dataset.num_q, **model_config).to(device)
    elif model_name == "dkt+":
        model = DKTPlus(dataset.num_q, **model_config).to(device)
    elif model_name == "dkvmn":
        model = DKVMN(dataset.num_q, **model_config).to(device)
    elif model_name == "sakt":
        model = SAKT(dataset.num_q, **model_config).to(device)
    elif model_name == "kqn":
        model = KQN(dataset.num_q, **model_config).to(device)
    elif model_name == "gkt":
        if model_config["method"] == "PAM":
            model = PAM(dataset.num_q, **model_config).to(device)
        elif model_config["method"] == "MHA":
            model = MHA(dataset.num_q, **model_config).to(device)
    else:
        print("The wrong model name was used...")
        return

    # Split dataset into train and test sets
    train_size = int(len(dataset) * train_ratio)
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(
        dataset, [train_size, test_size]
    )

    # Loading indices if available
    if os.path.exists(os.path.join(dataset.dataset_dir, "train_indices.pkl")):
        with open(
                os.path.join(dataset.dataset_dir, "train_indices.pkl"), "rb"
        ) as f:
            train_dataset.indices = pickle.load(f)
        with open(
                os.path.join(dataset.dataset_dir, "test_indices.pkl"), "rb"
        ) as f:
            test_dataset.indices = pickle.load(f)
    else:
        with open(
                os.path.join(dataset.dataset_dir, "train_indices.pkl"), "wb"
        ) as f:
            pickle.dump(train_dataset.indices, f)
        with open(
                os.path.join(dataset.dataset_dir, "test_indices.pkl"), "wb"
        ) as f:
            pickle.dump(test_dataset.indices, f)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=test_size, shuffle=True,
        collate_fn=collate_fn
    )

    # Initialize optimizer
    if optimizer == "sgd":
        opt = SGD(model.parameters(), learning_rate, momentum=0.9)
    elif optimizer == "adam":
        opt = Adam(model.parameters(), learning_rate)

    # Create CSV to save training results
    csv_file = create_results_csv(model_name, dataset_name)

    # Train the model and save results
    aucs, accs, loss_means = model.train_model(
        train_loader, test_loader, num_epochs, opt, ckpt_path
    )

    # Save results to CSV
    for epoch in range(num_epochs):
        with open(csv_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, loss_means[epoch], accs[epoch], aucs[epoch]])

    with open(os.path.join(ckpt_path, "aucs.pkl"), "wb") as f:
        pickle.dump(aucs, f)
    with open(os.path.join(ckpt_path, "loss_means.pkl"), "wb") as f:
        pickle.dump(loss_means, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="dkt",
        help="The name of the model to train. \
            The possible models are in [dkt, dkt+, dkvmn, sakt, gkt,kqn]. \
            The default model is dkt."
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="ASSIST2009",
        help="The name of the dataset to use in training. \
            The possible datasets are in \
            [ASSIST2009, ASSIST2015, Algebra2005, Statics2011]. \
            The default dataset is ASSIST2009."
    )
    args = parser.parse_args()

    main(args.model_name, args.dataset_name)
