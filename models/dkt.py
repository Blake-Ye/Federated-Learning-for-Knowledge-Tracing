import os

import numpy as np
import torch

from torch.nn import Module, Embedding, LSTM, Linear, Dropout
from torch.nn.functional import one_hot, binary_cross_entropy
from sklearn import metrics


class DKT(Module):
    '''
        Args:
            num_q: the total number of the questions(KCs) in the given dataset
            emb_size: the dimension of the embedding vectors in this model
            hidden_size: the dimension of the hidden vectors in this model
    '''

    def __init__(self, num_q, emb_size, hidden_size):
        super().__init__()
        self.num_q = num_q
        self.emb_size = emb_size
        self.hidden_size = hidden_size

        self.interaction_emb = Embedding(self.num_q * 2, self.emb_size)
        self.lstm_layer = LSTM(
            self.emb_size, self.hidden_size, batch_first=True
        )
        self.out_layer = Linear(self.hidden_size, self.num_q)
        self.dropout_layer = Dropout()

    def forward(self, q, r):
        '''
            Args:
                q: the question(KC) sequence with the size of [batch_size, n]
                r: the response sequence with the size of [batch_size, n]

            Returns:
                y: the knowledge level about the all questions(KCs)
        '''
        x = q + self.num_q * r

        h, _ = self.lstm_layer(self.interaction_emb(x))
        y = self.out_layer(h)
        y = self.dropout_layer(y)
        y = torch.sigmoid(y)

        return y

    def train_model(
            self, train_loader, test_loader, num_epochs, opt, ckpt_path
    ):
        '''
            Args:
                train_loader: the PyTorch DataLoader instance for training
                test_loader: the PyTorch DataLoader instance for test
                num_epochs: the number of epochs
                opt: the optimization to train this model
                ckpt_path: the path to save this model's parameters
        '''
        aucs = []
        accs = []
        loss_means = []

        max_auc = 0

        for i in range(1, num_epochs + 1):
            loss_mean = []

            for data in train_loader:
                q, r, qshft, rshft, m = data

                self.train()

                y = self(q.long(), r.long())
                y = (y * one_hot(qshft.long(), self.num_q)).sum(-1)

                y = torch.masked_select(y, m)
                t = torch.masked_select(rshft, m)

                opt.zero_grad()
                loss = binary_cross_entropy(y, t)
                loss.backward()
                opt.step()

                loss_mean.append(loss.detach().cpu().numpy())

            with torch.no_grad():
                for data in test_loader:
                    q, r, qshft, rshft, m = data

                    self.eval()

                    y = self(q.long(), r.long())
                    y = (y * one_hot(qshft.long(), self.num_q)).sum(-1)

                    y = torch.masked_select(y, m).detach().cpu()
                    t = torch.masked_select(rshft, m).detach().cpu()

                    auc = metrics.roc_auc_score(
                        y_true=t.numpy(), y_score=y.numpy()
                    )
                    y_pred = (y.numpy() > 0.5).astype(int)  # 将概率值转换为0/1预测
                    acc = (y_pred == t.numpy()).mean()

                    loss_mean = np.mean(loss_mean)

                    print(
                        "Epoch: {},   AUC: {:.4f},   ACC: {:.4f},   Loss Mean: {:.4f}"
                        .format(i, auc, acc, loss_mean)
                    )

                    if auc > max_auc:
                        torch.save(
                            self.state_dict(),
                            os.path.join(
                                ckpt_path, "model.ckpt"
                            )
                        )
                        max_auc = auc

                    aucs.append(auc)
                    accs.append(acc)
                    loss_means.append(loss_mean)

        return aucs, accs, loss_means

    def evaluate(self, test_loader):
        """
            Evaluate the model on the test dataset.

            Args:
                test_loader: the PyTorch DataLoader instance for the test dataset

            Returns:
                test_loss: the average loss on the test dataset
                test_acc: the accuracy on the test dataset
                test_auc: the AUC score on the test dataset
        """
        self.eval()

        losses = []
        all_y = []
        all_t = []

        with torch.no_grad():
            for data in test_loader:
                q, r, qshft, rshft, m = data

                # Forward pass
                y = self(q.long(), r.long())
                y = (y * one_hot(qshft.long(), self.num_q)).sum(-1)

                # Mask predictions and targets
                y = torch.masked_select(y, m).detach().cpu()
                t = torch.masked_select(rshft, m).detach().cpu()

                # Compute loss
                loss = binary_cross_entropy(y, t)
                losses.append(loss.item())

                # Store predictions and true values for metrics calculation
                all_y.append(y)
                all_t.append(t)

        # Concatenate all predictions and targets
        all_y = torch.cat(all_y).numpy()
        all_t = torch.cat(all_t).numpy()

        # Compute metrics
        test_loss = np.mean(losses)
        test_auc = metrics.roc_auc_score(y_true=all_t, y_score=all_y)
        y_pred = (all_y > 0.5).astype(int)
        test_acc = (y_pred == all_t).mean()

        return test_auc,test_acc,test_loss