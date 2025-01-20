import os
import numpy as np
import torch
from torch.nn import Module, Embedding, LSTM, Linear, Dropout, Sequential, ReLU
from torch.nn.functional import binary_cross_entropy
from sklearn import metrics


class KQN(Module):
    def __init__(self, num_q, dim_v, dim_s, hidden_size):
        super().__init__()
        self.num_q = num_q
        self.dim_v = dim_v
        self.dim_s = dim_s
        self.hidden_size = hidden_size

        self.x_emb = Embedding(self.num_q * 2, self.dim_v)
        self.knowledge_encoder = LSTM(self.dim_v, self.dim_v, batch_first=True)
        self.out_layer = Linear(self.dim_v, self.dim_s)
        self.dropout_layer = Dropout()

        self.q_emb = Embedding(self.num_q, self.dim_v)
        self.skill_encoder = Sequential(
            Linear(self.dim_v, self.hidden_size),
            ReLU(),
            Linear(self.hidden_size, self.dim_v),
            ReLU()
        )

    def forward(self, q, r, qry):
        # Knowledge State Encoding
        x = q + self.num_q * r
        x = self.x_emb(x)
        h, _ = self.knowledge_encoder(x)
        ks = self.out_layer(h)
        ks = self.dropout_layer(ks)

        # Skill Encoding
        e = self.q_emb(qry)
        o = self.skill_encoder(e)
        s = o / torch.norm(o, p=2)

        p = torch.sigmoid((ks * s).sum(-1))

        return p

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

            # Training loop
            for data in train_loader:
                q, r, qshft, rshft, m = data

                self.train()

                p = self(q.long(), r.long(), qshft.long())
                p = torch.masked_select(p, m)
                t = torch.masked_select(rshft, m)

                opt.zero_grad()
                loss = binary_cross_entropy(p, t)
                loss.backward()
                opt.step()

                loss_mean.append(loss.detach().cpu().numpy())

            loss_mean = np.mean(loss_mean)

            # Evaluate on test_loader
            auc, acc, eval_loss = self.evaluate(test_loader)

            print(
                "Epoch: {},   AUC: {},   Accuracy: {},   Loss Mean: {},   Eval Loss: {}"
                .format(i, auc, acc, loss_mean, eval_loss)
            )

            # Save the best model
            if auc > max_auc:
                torch.save(
                    self.state_dict(),
                    os.path.join(ckpt_path, "model.ckpt")
                )
                max_auc = auc

            aucs.append(auc)
            accs.append(acc)
            loss_means.append(loss_mean)

        return aucs, accs, loss_means

    def evaluate(self, test_loader):
        '''
        Evaluate the model using the test_loader.

        Args:
            test_loader: PyTorch DataLoader for test data

        Returns:
            auc: AUC score on the test set
            acc: Accuracy on the test set
            loss: Mean binary cross-entropy loss on the test set
        '''
        self.eval()

        all_preds = []
        all_labels = []
        all_losses = []

        with torch.no_grad():
            for data in test_loader:
                q, r, qshft, rshft, m = data

                p = self(q.long(), r.long(), qshft.long())
                p = torch.masked_select(p, m).detach().cpu()
                t = torch.masked_select(rshft, m).detach().cpu()

                loss = binary_cross_entropy(p, t, reduction='mean')
                all_losses.append(loss.item())

                all_preds.append(p.numpy())
                all_labels.append(t.numpy())

        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)

        auc = metrics.roc_auc_score(y_true=all_labels, y_score=all_preds)
        acc = np.mean((all_preds >= 0.5) == all_labels)
        mean_loss = np.mean(all_losses)

        return auc, acc, mean_loss
