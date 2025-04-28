# server.py
import csv
import os
import json
from collections import OrderedDict
from logging import DEBUG
from typing import List, Tuple

import flwr as fl
import torch
from flwr.common.logger import update_console_handler
from flwr.server import Driver, LegacyContext, ServerApp, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.server.workflow import DefaultWorkflow, SecAggPlusWorkflow
from secaggexample.workflow_with_log import SecAggPlusWorkflowWithLogs
from flwr.common import Context, Metrics, ndarrays_to_parameters
# 导入数据集和模型模块（请根据实际路径调整）
from data_loaders.algebra2005 import Algebra2005
from data_loaders.assist2009 import ASSIST2009
from data_loaders.assist2015 import ASSIST2015
from data_loaders.statics2011 import Statics2011
from models.dkt import DKT
from models.dkt_plus import DKTPlus
from models.dkvmn import DKVMN
from models.gkt import PAM, MHA
from models.kqn import KQN
from models.saint import SAINT
from models.sakt import SAKT
from models.utils import collate_fn

from torch.utils.data import DataLoader

# 读取配置文件
with open(r"../knowledge-tracing-collection-pytorch-main/config.json") as f:
    config = json.load(f)

model_name = 'dkt'
dataset_name = 'ASSIST2009'

model_config = config[model_name]
train_config = config["train_config"]
batch_size = train_config["batch_size"]
num_epochs = train_config["num_epochs"]
train_ratio = train_config["train_ratio"]
learning_rate = train_config["learning_rate"]
optimizer = train_config["optimizer"]
seq_len = train_config["seq_len"]

# 加载数据集（这里只用于全局评估，可根据需要修改为测试集）
if dataset_name == "ASSIST2009":
    dataset = ASSIST2009(seq_len)
elif dataset_name == "ASSIST2015":
    dataset = ASSIST2015(seq_len)
elif dataset_name == "Algebra2005":
    dataset = Algebra2005(seq_len)
elif dataset_name == "Statics2011":
    dataset = Statics2011(seq_len)

# 这里简单地使用整个数据集作为评估集（在实际场景下你可以单独划分测试集）
test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

device = "cuda" if torch.cuda.is_available() else "cpu"

# 根据 config 中指定的 model_name 加载模型

if model_name == "dkt":
    model = DKT(dataset.num_q, **model_config).to(device)
elif model_name == "dkt+":
    model = DKTPlus(dataset.num_q, **model_config).to(device)
elif model_name == "dkvmn":
    model = DKVMN(dataset.num_q, **model_config).to(device)
elif model_name == "sakt":
    model = SAKT(dataset.num_q, **model_config).to(device)
elif model_name == "saint":
    model = SAINT(dataset.num_q, **model_config).to(device)
elif model_name == "kqn":
    model = KQN(dataset.num_q, **model_config).to(device)
elif model_name == "gkt":
    if model_config["method"] == "PAM":
        model = PAM(dataset.num_q, **model_config).to(device)
    elif model_config["method"] == "MHA":
        model = MHA(dataset.num_q, **model_config).to(device)

# 创建存放结果的目录和 CSV 文件（记录每轮评估结果）
results_dir = "../federated_learning_secureAggregation"
os.makedirs(results_dir, exist_ok=True)
csv_file = os.path.join(results_dir, f"{model_name}_{dataset_name}_FL.csv")
if not os.path.exists(csv_file):
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['round', 'loss', 'accuracy', 'auc'])


def get_evaluate_fn(test_loader):
    def evaluate_fn(server_round: int, parameters, config):
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({
            k: torch.tensor(v, dtype=torch.float32, device=device) for k, v in params_dict
        })
        model.load_state_dict(state_dict, strict=True)
        auc, acc, loss = model.evaluate(test_loader)

        # Save evaluation results to CSV file
        with open(csv_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([server_round, loss, acc, auc])

        return loss, {'accuracy': acc, 'auc': auc}

    return evaluate_fn


def get_initial_parameters(model):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


# 使用 Flower 的分布式 ServerApp 模式
app = ServerApp()


@app.main()
def main(driver: Driver, context: Context) -> None:
    # 从 run_config 中读取是否启用安全聚合
    is_demo = context.run_config["is-demo"]
    strategy = FedAvg(
        fraction_fit=0.33,  # 每轮选择 33% 的客户端进行训练
        fraction_evaluate=0.33,  # 每轮选择 33% 的客户端进行评估
        min_available_clients=6,
        evaluate_fn=get_evaluate_fn(test_loader),
    )
    num_rounds = context.run_config["num-server-rounds"]
    legacy_context = LegacyContext(
        context=context,
        config=ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )
    if is_demo:
        update_console_handler(DEBUG, True, True)
        fit_workflow = SecAggPlusWorkflowWithLogs(
            num_shares=context.run_config["num-shares"],
            reconstruction_threshold=context.run_config["reconstruction-threshold"],
            max_weight=50000,
            timeout=context.run_config["timeout"],
        )
    else:
        fit_workflow = SecAggPlusWorkflow(
            num_shares=context.run_config["num-shares"],
            reconstruction_threshold=context.run_config["reconstruction-threshold"],
            max_weight=context.run_config["max-weight"],
        )

    workflow = DefaultWorkflow(fit_workflow=fit_workflow)
    workflow(driver, legacy_context)
