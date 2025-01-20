import os
import numpy as np
import torch
from torch.nn import Module, Parameter, Embedding, Linear
from torch.nn.init import kaiming_normal_
from torch.nn.functional import binary_cross_entropy
from sklearn import metrics


class DKVMN(Module):
    '''
        Args:
            num_q: the total number of the questions(KCs) in the given dataset
            dim_s: the dimension of the state vectors in this model
            size_m: the memory size of this model
    '''
    def __init__(self, num_q, dim_s, size_m):
        super().__init__()
        self.num_q = num_q
        self.dim_s = dim_s
        self.size_m = size_m

        self.k_emb_layer = Embedding(self.num_q, self.dim_s)
        self.Mk = Parameter(torch.Tensor(self.size_m, self.dim_s))
        self.Mv0 = Parameter(torch.Tensor(self.size_m, self.dim_s))

        kaiming_normal_(self.Mk)
        kaiming_normal_(self.Mv0)

        self.v_emb_layer = Embedding(self.num_q * 2, self.dim_s)

        self.f_layer = Linear(self.dim_s * 2, self.dim_s)
        self.p_layer = Linear(self.dim_s, 1)

        self.e_layer = Linear(self.dim_s, self.dim_s)
        self.a_layer = Linear(self.dim_s, self.dim_s)

    def forward(self, q, r):
        '''
            Args:
                q: the question(KC) sequence with the size of [batch_size, n]
                r: the response sequence with the size of [batch_size, n]

            Returns:
                p: the knowledge level about q
                Mv: the value matrices from q, r
        '''
        x = q + self.num_q * r

        batch_size = x.shape[0]
        Mvt = self.Mv0.unsqueeze(0).repeat(batch_size, 1, 1)

        Mv = [Mvt]

        k = self.k_emb_layer(q)
        v = self.v_emb_layer(x)

        w = torch.softmax(torch.matmul(k, self.Mk.T), dim=-1)

        # Write Process
        e = torch.sigmoid(self.e_layer(v))
        a = torch.tanh(self.a_layer(v))

        for et, at, wt in zip(
            e.permute(1, 0, 2), a.permute(1, 0, 2), w.permute(1, 0, 2)
        ):
            Mvt = Mvt * (1 - (wt.unsqueeze(-1) * et.unsqueeze(1))) + \
                (wt.unsqueeze(-1) * at.unsqueeze(1))
            Mv.append(Mvt)

        Mv = torch.stack(Mv, dim=1)

        # Read Process
        f = torch.tanh(
            self.f_layer(
                torch.cat(
                    [
                        (w.unsqueeze(-1) * Mv[:, :-1]).sum(-2),
                        k
                    ],
                    dim=-1
                )
            )
        )
        p = torch.sigmoid(self.p_layer(f)).squeeze()

        return p, Mv

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
                q, r, _, _, m = data

                self.train()

                p, _ = self(q.long(), r.long())
                p = torch.masked_select(p, m)
                t = torch.masked_select(r, m).float()

                opt.zero_grad()
                loss = binary_cross_entropy(p, t)
                loss.backward()
                opt.step()

                loss_mean.append(loss.detach().cpu().numpy())

            loss_mean = np.mean(loss_mean)

            # Evaluation on test_loader
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
                q, r, _, _, m = data

                p, _ = self(q.long(), r.long())
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
