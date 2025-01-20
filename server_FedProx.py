import csv
import os
import json
import flwr as fl
from collections import OrderedDict
from client import test_loader, generate_client_fn, train_loader, model_name, dataset_name
from data_loaders.algebra2005 import Algebra2005
from data_loaders.assist2009 import ASSIST2009
from data_loaders.assist2015 import ASSIST2015
from data_loaders.statics2011 import Statics2011
from models.dkt import DKT
import torch
from models.dkt_plus import DKTPlus
from models.dkvmn import DKVMN
from models.gkt import PAM, MHA
from models.kqn import KQN
from models.saint import SAINT
from models.sakt import SAKT

# Read the configuration file
with open("config.json") as f:
    config = json.load(f)

model_config = config[model_name]
train_config = config["train_config"]

# Configuration parameters
batch_size = train_config["batch_size"]
num_epochs = train_config["num_epochs"]
train_ratio = train_config["train_ratio"]
learning_rate = train_config["learning_rate"]
optimizer = train_config["optimizer"]  # can be [sgd, adam]
seq_len = train_config["seq_len"]

# Load the dataset
if dataset_name == "ASSIST2009":
    dataset = ASSIST2009(seq_len)
elif dataset_name == "ASSIST2015":
    dataset = ASSIST2015(seq_len)
elif dataset_name == "Algebra2005":
    dataset = Algebra2005(seq_len)
elif dataset_name == "Statics2011":
    dataset = Statics2011(seq_len)

train_size = int(len(dataset) * train_config["train_ratio"])

# Check the device, choose CPU or GPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the model
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

# Create a directory for saving results in CSV file
results_dir = "federated_learning_FedProx"
os.makedirs(results_dir, exist_ok=True)
csv_file = os.path.join(results_dir, f"{model_name}_{dataset_name}FL.csv")

# Generate CSV file headers (if file doesn't exist)
if not os.path.exists(csv_file):
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['round', 'loss', 'accuracy', 'auc'])


# Update evaluation function to save results for each round
def get_evalulate_fn(test_loader):
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


# Define DPFedAvgAdaptive strategy with adaptive clipping


# 初始化策略
# Define FedAvgM strategy

# Define FedTrimmedAvg strategy
strategy = fl.server.strategy.FedProx(
    fraction_fit=0.33,  # Select 33% of clients for training per round
    fraction_evaluate=0.33,  # Select 33% of clients for evaluation per round
    min_available_clients=6,  # At least 6 clients required
    evaluate_fn=get_evalulate_fn(test_loader),  # Pass evaluation function
    proximal_mu=0.1,
)

# Get client function generator
client_fn_callback = generate_client_fn(train_loader, test_loader)

# Start simulation
history = fl.simulation.start_simulation(
    client_fn=client_fn_callback,  # Pass client function to generate each client
    num_clients=6,
    config=fl.server.ServerConfig(num_rounds=100),  # Set the total rounds of simulation to 100
    strategy=strategy,  # Use the strategy defined above
    client_resources={"num_cpus": 3, "num_gpus": 0.5}  # Assign 0.5 GPUs for each client
)
