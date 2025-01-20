import numpy as np
import torch

from torch.nn import Module, Parameter, Embedding, Linear, Transformer
from torch.nn.init import normal_
from torch.nn.functional import binary_cross_entropy
from sklearn import metrics


class SAINT(Module):
    def __init__(
        self, num_q, n, d, num_attn_heads, dropout, num_tr_layers=1
    ):
        super().__init__()
        self.num_q = num_q
        self.n = n
        self.d = d
        self.num_attn_heads = num_attn_heads
        self.dropout = dropout
        self.num_tr_layers = num_tr_layers

        self.E = Embedding(self.num_q, self.d)
        self.R = Embedding(2, self.d)
        self.P = Parameter(torch.Tensor(self.n, self.d))
        self.S = Parameter(torch.Tensor(1, self.d))

        normal_(self.P)
        normal_(self.S)

        self.transformer = Transformer(
            self.d,
            self.num_attn_heads,
            num_encoder_layers=self.num_tr_layers,
            num_decoder_layers=self.num_tr_layers,
            dropout=self.dropout,
        )

        self.pred = Linear(self.d, 1)

    def forward(self, q, r):
        batch_size = r.shape[0]

        E = self.E(q).permute(1, 0, 2)

        R = self.R(r[:, :-1]).permute(1, 0, 2)
        S = self.S.repeat(batch_size, 1).unsqueeze(0)
        R = torch.cat([S, R], dim=0)

        P = self.P.unsqueeze(1)

        mask = self.transformer.generate_square_subsequent_mask(self.n)
        R = self.transformer(
            E + P, R + P, mask, mask, mask
        )
        R = R.permute(1, 0, 2)

        p = torch.sigmoid(self.pred(R)).squeeze()

        return p

    def discover_concepts(self, q, r):
        queries = torch.LongTensor([list(range(self.num_q))] * self.n)\
            .permute(1, 0)

        x = q + self.num_q * r
        x = x.repeat(self.num_q, 1)

        M = self.M(x).permute(1, 0, 2)
        E = self.E(queries).permute(1, 0, 2)
        P = self.P.unsqueeze(1)

        causal_mask = torch.triu(
            torch.ones([E.shape[0], M.shape[0]]), diagonal=1
        ).bool()

        M += P

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

    def train_model(self, train_loader, test_loader, num_epochs, opt):
        aucs = []
        accs = []
        loss_means = []

        for i in range(1, num_epochs + 1):
            loss_mean = []

            for data in train_loader:
                q, r, _, _, m = data

                self.train()

                p = self(q, r)
                p = torch.masked_select(p, m)
                t = torch.masked_select(r, m).float()

                opt.zero_grad()
                loss = binary_cross_entropy(p, t)
                loss.backward()
                opt.step()

                loss_mean.append(loss.detach().cpu().numpy())

            with torch.no_grad():
                all_preds = []
                all_labels = []
                for data in test_loader:
                    q, r, _, _, m = data

                    self.eval()

                    p = self(q, r)
                    p = torch.masked_select(p, m).detach().cpu()
                    t = torch.masked_select(r, m).float().detach().cpu()

                    all_preds.append(p.numpy())
                    all_labels.append(t.numpy())

                all_preds = np.concatenate(all_preds)
                all_labels = np.concatenate(all_labels)

                auc = metrics.roc_auc_score(y_true=all_labels, y_score=all_preds)
                acc = np.mean((all_preds >= 0.5) == all_labels)
                loss_mean = np.mean(loss_mean)

                print(
                    "Epoch: {},   AUC: {},   Accuracy: {},   Loss Mean: {}"
                    .format(i, auc, acc, loss_mean)
                )

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
                q, r, _, _, m = data

                p = self(q, r)
                p = torch.masked_select(p, m).detach().cpu()
                t = torch.masked_select(r, m).float().detach().cpu()

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
