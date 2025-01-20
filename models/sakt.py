import os
import numpy as np
import torch
from torch.nn import Module, Parameter, Embedding, Sequential, Linear, ReLU, \
    MultiheadAttention, LayerNorm, Dropout
from torch.nn.init import kaiming_normal_
from torch.nn.functional import binary_cross_entropy
from sklearn import metrics


class SAKT(Module):
    '''
        This implementation has a reference from: \
        https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#TransformerEncoderLayer

        Args:
            num_q: the total number of the questions(KCs) in the given dataset
            n: the length of the sequence of the questions or responses
            d: the dimension of the hidden vectors in this model
            num_attn_heads: the number of the attention heads in the \
                multi-head attention module in this model
            dropout: the dropout rate of this model
    '''

    def __init__(self, num_q, n, d, num_attn_heads, dropout):
        super().__init__()
        self.num_q = num_q
        self.n = n
        self.d = d
        self.num_attn_heads = num_attn_heads
        self.dropout = dropout

        self.M = Embedding(self.num_q * 2, self.d)
        self.E = Embedding(self.num_q, d)
        self.P = Parameter(torch.Tensor(self.n, self.d))

        kaiming_normal_(self.P)

        self.attn = MultiheadAttention(
            self.d, self.num_attn_heads, dropout=self.dropout
        )
        self.attn_dropout = Dropout(self.dropout)
        self.attn_layer_norm = LayerNorm(self.d)

        self.FFN = Sequential(
            Linear(self.d, self.d),
            ReLU(),
            Dropout(self.dropout),
            Linear(self.d, self.d),
            Dropout(self.dropout),
        )
        self.FFN_layer_norm = LayerNorm(self.d)

        self.pred = Linear(self.d, 1)

    def forward(self, q, r, qry):
        '''
            Args:
                q: the question(KC) sequence with the size of [batch_size, n]
                r: the response sequence with the size of [batch_size, n]
                qry: the query sequence with the size of [batch_size, m], \
                    where the query is the question(KC) what the user wants \
                    to check the knowledge level of

            Returns:
                p: the knowledge level about the query
                attn_weights: the attention weights from the multi-head \
                    attention module
        '''
        x = q + self.num_q * r

        M = self.M(x).permute(1, 0, 2)
        E = self.E(qry).permute(1, 0, 2)
        P = self.P.unsqueeze(1)

        causal_mask = torch.triu(
            torch.ones([E.shape[0], M.shape[0]]), diagonal=1
        ).bool()

        M = M + P

        S, attn_weights = self.attn(E, M, M, attn_mask=causal_mask)
        S = self.attn_dropout(S)
        S = S.permute(1, 0, 2)
        M = M.permute(1, 0, 2)
        E = E.permute(1, 0, 2)

        S = self.attn_layer_norm(S + M + E)

        F = self.FFN(S)
        F = self.FFN_layer_norm(F + S)

        p = torch.sigmoid(self.pred(F)).squeeze()

        return p, attn_weights

    def train_model(self, train_loader, test_loader, num_epochs, opt, ckpt_path):
        '''
        Train the model and evaluate it on test_loader after every epoch.

        Args:
            train_loader: PyTorch DataLoader for training data
            test_loader: PyTorch DataLoader for test data
            num_epochs: Number of epochs to train
            opt: Optimizer for training
            ckpt_path: Path to save model checkpoints

        Returns:
            aucs: List of AUCs for each epoch
            accs: List of accuracies for each epoch
            loss_means: List of mean losses for each epoch
        '''
        aucs = []
        accs = []
        loss_means = []

        max_auc = 0

        for epoch in range(1, num_epochs + 1):
            loss_mean = []

            # Training loop
            for data in train_loader:
                q, r, qshft, rshft, m = data

                self.train()

                p, _ = self(q.long(), r.long(), qshft.long())
                p = torch.masked_select(p, m)
                t = torch.masked_select(rshft, m)

                opt.zero_grad()
                loss = binary_cross_entropy(p, t)
                loss.backward()
                opt.step()

                loss_mean.append(loss.detach().cpu().numpy())

            # Compute mean loss for this epoch
            loss_mean = np.mean(loss_mean)

            # Evaluate on test_loader
            self.eval()
            all_preds = []
            all_labels = []

            with torch.no_grad():
                for data in test_loader:
                    q, r, qshft, rshft, m = data

                    p, _ = self(q.long(), r.long(), qshft.long())
                    p = torch.masked_select(p, m).detach().cpu()
                    t = torch.masked_select(rshft, m).detach().cpu()

                    all_preds.append(p.numpy())
                    all_labels.append(t.numpy())

            all_preds = np.concatenate(all_preds)
            all_labels = np.concatenate(all_labels)

            auc = metrics.roc_auc_score(y_true=all_labels, y_score=all_preds)
            acc = np.mean((all_preds >= 0.5) == all_labels)

            print(f"Epoch: {epoch}, AUC: {auc:.4f}, Accuracy: {acc:.4f}, Loss Mean: {loss_mean:.4f}")

            # Save the best model
            if auc > max_auc:
                torch.save(self.state_dict(), os.path.join(ckpt_path, "model.ckpt"))
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

                p, _ = self(q.long(), r.long(), qshft.long())
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

