import random
import time
import json
import torch
import flwr as fl
import ray
from torch.optim import Adam
from collections import OrderedDict
from torch.utils.data import DataLoader, random_split
from flwr.common import Context, Metrics, ndarrays_to_parameters
from flwr_datasets import FederatedDataset
from models.dkt import DKT
from models.dkt_plus import DKTPlus
from models.dkvmn import DKVMN
from models.gkt import MHA, PAM
from models.kqn import KQN
from models.saint import SAINT
from models.sakt import SAKT
from models.utils import collate_fn
from data_loaders.assist2009 import ASSIST2009
from data_loaders.assist2015 import ASSIST2015
from data_loaders.algebra2005 import Algebra2005
from data_loaders.statics2011 import Statics2011
from flwr.client import ClientApp
from flwr.client.mod import secaggplus_mod

with open(r"../knowledge-tracing-collection-pytorch-main/config.json") as f:
    config = json.load(f)
train_config = config["train_config"]
batch_size = train_config["batch_size"]
seq_len = train_config["seq_len"]
learning_rate = train_config["learning_rate"]

model_name = 'dkt'
dataset_name = 'ASSIST2009'

if dataset_name == "ASSIST2009":
    dataset = ASSIST2009(seq_len)
elif dataset_name == "ASSIST2015":
    dataset = ASSIST2015(seq_len)
elif dataset_name == "Algebra2005":
    dataset = Algebra2005(seq_len)
elif dataset_name == "Statics2011":
    dataset = Statics2011(seq_len)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model_config = config[model_name]

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

train_size = int(len(dataset) * train_config["train_ratio"])
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(
    dataset, [train_size, test_size]
)
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True,
    collate_fn=collate_fn
)
test_loader = DataLoader(
    test_dataset, batch_size=test_size, shuffle=True,
    collate_fn=collate_fn
)


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, train_loader, test_loader, model, is_demo=False, timeout=0) -> None:
        super().__init__()
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model = model.to(device)
        self.is_demo = is_demo
        self.timeout = timeout
        self.opt = Adam(self.model.parameters(), lr=learning_rate)  # Use Adam optimizer

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v).to(device) for k, v in params_dict})
        self.model.load_state_dict(state_dict)

    def get_parameters(self, config):
        """Return model parameters"""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]  # Move from GPU to CPU

    def fit(self, parameters, config):
        """Local training"""
        self.set_parameters(parameters)
        print('--------------------------------------------------------------hhhhhhhh-')
        self.model.train_model(self.train_loader, self.test_loader, num_epochs=5, opt=self.opt, ckpt_path='ckpts')
        return self.get_parameters(config), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        """Evaluate the model on the test set"""
        self.set_parameters(parameters)
        auc, acc, loss = self.model.evaluate(self.test_loader)
        return loss, len(self.test_loader.dataset), {'accuracy': acc, 'auc': auc}


def client_fn(context: fl.common.Context):
    """Return a FlowerClient with data partitions"""
    partition_id = int(context.node_config.get("partition-id"))
    num_partitions = int(context.node_config.get("num-partitions"))

    # Partition the training data
    train_dataset = train_loader.dataset  # Get the training dataset
    train_partition_size = len(train_dataset) // num_partitions
    train_start_idx = partition_id * train_partition_size
    train_end_idx = (partition_id + 1) * train_partition_size if partition_id != num_partitions - 1 else len(
        train_dataset)

    # Add random offset to the partition range for increased data diversity
    random_offset = random.randint(-5, 5)
    train_start_idx = max(0, train_start_idx + random_offset)
    train_end_idx = min(len(train_dataset), train_end_idx + random_offset)

    # Create training data subset
    client_train_subset = torch.utils.data.Subset(train_dataset, range(train_start_idx, train_end_idx))

    # Create a new training DataLoader
    client_train_loader = DataLoader(
        client_train_subset, batch_size=train_loader.batch_size,
        shuffle=True, collate_fn=train_loader.collate_fn  # Enable random shuffle
    )

    # Partition the test data
    test_dataset = test_loader.dataset  # Get the test dataset
    test_partition_size = len(test_dataset) // num_partitions
    test_start_idx = partition_id * test_partition_size
    test_end_idx = (partition_id + 1) * test_partition_size if partition_id != num_partitions - 1 else len(test_dataset)

    # Randomly adjust the test data partition range
    random_offset = random.randint(-3, 3)
    test_start_idx = max(0, test_start_idx + random_offset)
    test_end_idx = min(len(test_dataset), test_end_idx + random_offset)

    # Create test data subset
    client_test_subset = torch.utils.data.Subset(test_dataset, range(test_start_idx, test_end_idx))

    # Create a new test DataLoader
    client_test_loader = DataLoader(
        client_test_subset, batch_size=test_loader.batch_size,
        shuffle=False, collate_fn=test_loader.collate_fn
    )

    # Dynamically assign device with some randomness
    available_devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else [
        "cpu"]
    device = random.choice(available_devices)  # Randomly choose an available device
    print(f"Client {partition_id} is using device {device}")

    # Create and return a FlowerClient instance
    return FlowerClient(
        train_loader=client_train_loader,
        test_loader=client_test_loader,
        model=model.to(device)  # Ensure the model is on the client's device
    ).to_client()


app = ClientApp(
    client_fn=client_fn,
    mods=[secaggplus_mod]
)
