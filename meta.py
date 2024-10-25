import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch import optim
from sklearn.metrics import precision_recall_fscore_support
import numpy as np

from learner import Learner
from copy import deepcopy


class Meta(nn.Module):
    """
    Meta Learner
    """
    def __init__(self, args, config):
        """

        :param args:
        """
        super(Meta, self).__init__()

        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.n_way = args.n_way
        self.k_spt = args.k_spt
        self.k_qry = args.k_qry
        self.task_num = args.task_num
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test
        self.lambda_reg = args.lambda_reg  # Regularization coefficient

        self.net = Learner(config)
        self.meta_optim = optim.Adam(self.net.parameters(), lr=self.meta_lr)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def clip_grad_by_norm_(self, grad, max_norm):
        """
        in-place gradient clipping.
        :param grad: list of gradients
        :param max_norm: maximum norm allowable
        :return:
        """

        total_norm = 0
        counter = 0
        for g in grad:
            param_norm = g.data.norm(2)
            total_norm += param_norm.item() ** 2
            counter += 1
        total_norm = total_norm ** (1.0 / 2)

        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for g in grad:
                g.data.mul_(clip_coef)

        return total_norm / counter

    def forward(self, x_spt, y_spt, x_qry, y_qry):
        """

        :param x_spt:   [b, setsz, c_, h, w]
        :param y_spt:   [b, setsz]
        :param x_qry:   [b, querysz, c_, h, w]
        :param y_qry:   [b, querysz]
        :return:
        """

        task_num, setsz, c_, embedsz = x_spt.size()
        # task_num, setsz, c_, embedsz, w = x_spt.size()
        querysz = x_qry.size(1)

        losses_q = [
            0 for _ in range(self.update_step + 1)
        ]  # losses_q[i] is the loss on step i
        corrects = [0 for _ in range(self.update_step + 1)]

        for i in range(task_num):

            # 1. run the i-th task and compute loss for k=0
            logits = self.net(x_spt[i], vars=None, bn_training=True)
            loss = F.cross_entropy(logits, y_spt[i])

            grad = torch.autograd.grad(loss, self.net.parameters())
            fast_weights = list(
                map(
                    lambda p: p[1] - self.update_lr * p[0],
                    zip(grad, self.net.parameters()),
                )
            )

            # this is the loss and accuracy before first update
            with torch.no_grad():
                # [setsz, nway]
                logits_q = self.net(x_qry[i], self.net.parameters(), bn_training=True)
                loss_q = F.cross_entropy(logits_q, y_qry[i])
                losses_q[0] += loss_q

                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry[i]).sum().item()
                corrects[0] = corrects[0] + correct

            # this is the loss and accuracy after the first update
            with torch.no_grad():
                # [setsz, nway]
                logits_q = self.net(x_qry[i], fast_weights, bn_training=True)
                loss_q = F.cross_entropy(logits_q, y_qry[i])
                losses_q[1] += loss_q
                # [setsz]
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry[i]).sum().item()
                corrects[1] = corrects[1] + correct

            for k in range(1, self.update_step):
                # 1. run the i-th task and compute loss for k=1~K-1
                logits = self.net(x_spt[i], fast_weights, bn_training=True)
                loss = F.cross_entropy(logits, y_spt[i])

                # 2. compute grad on theta_pi
                grad = torch.autograd.grad(loss, fast_weights)
                # 3. theta_pi = theta_pi - train_lr * grad
                fast_weights = list(
                    map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights))
                )

                logits_q = self.net(x_qry[i], fast_weights, bn_training=True)
                # loss_q will be overwritten and just keep the loss_q on last update step.
                loss_q = F.cross_entropy(logits_q, y_qry[i])
                losses_q[k + 1] += loss_q

                with torch.no_grad():
                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    correct = (
                        torch.eq(pred_q, y_qry[i]).sum().item()
                    )  # convert to numpy
                    corrects[k + 1] = corrects[k + 1] + correct

        # end of all tasks
        # sum over all losses on query set across all tasks
        loss_q = losses_q[-1] / task_num

        # optimize theta parameters
        self.meta_optim.zero_grad()
        loss_q.backward()
        self.meta_optim.step()

        accs = np.array(corrects) / (querysz * task_num)

        return accs

    def finetunning(self, x_spt, y_spt, x_qry, y_qry):
        """
        :param x_spt:   [setsz, c_, h, w]
        :param y_spt:   [setsz]
        :param x_qry:   [querysz, c_, h, w]
        :param y_qry:   [querysz]
        :return:
        """

        assert len(x_spt.shape) == 3

        querysz = x_qry.size(0)

        corrects = [0 for _ in range(self.update_step_test + 1)]
        precisions = [0 for _ in range(self.update_step_test + 1)]
        recalls = [0 for _ in range(self.update_step_test + 1)]
        f1s = [0 for _ in range(self.update_step_test + 1)]
        all_pred_q = []
        all_y_qry = []

        # in order to not ruin the state of running_mean/variance and bn_weight/bias
        # we finetunning on the copied model instead of self.net
        net = deepcopy(self.net)

        # 1. run the i-th task and compute loss for k=0
        logits = net(x_spt)
        loss = F.cross_entropy(logits, y_spt)
        grad = torch.autograd.grad(loss, net.parameters())
        fast_weights = list(
            map(lambda p: p[1] - self.update_lr * p[0], zip(grad, net.parameters()))
        )

        # this is the loss and accuracy before first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q = net(x_qry, net.parameters(), bn_training=True)
            # [setsz]
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            all_pred_q.append(pred_q.cpu().numpy())
            all_y_qry.append(y_qry.cpu().numpy())
            # scalar
            correct = torch.eq(pred_q, y_qry).sum().item()
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_qry.cpu().numpy(),
                pred_q.cpu().numpy(),
                average="weighted",
                zero_division=0,
            )
            corrects[0] = corrects[0] + correct
            precisions[0] = precisions[0] + precision
            recalls[0] = recalls[0] + recall
            f1s[0] = f1s[0] + f1

        # this is the loss and accuracy after the first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q = net(x_qry, fast_weights, bn_training=True)
            # [setsz]
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            all_pred_q.append(pred_q.cpu().numpy())
            all_y_qry.append(y_qry.cpu().numpy())
            # scalar
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_qry.cpu().numpy(),
                pred_q.cpu().numpy(),
                average="weighted",
                zero_division=0,
            )
            correct = torch.eq(pred_q, y_qry).sum().item()
            corrects[1] = corrects[1] + correct
            precisions[1] = precisions[1] + precision
            recalls[1] = recalls[1] + recall
            f1s[1] = f1s[1] + f1

        for k in range(1, self.update_step_test):
            # 1. run the i-th task and compute loss for k=1~K-1
            logits = net(x_spt, fast_weights, bn_training=True)
            loss = F.cross_entropy(logits, y_spt)
            # 2. compute grad on theta_pi
            grad = torch.autograd.grad(loss, fast_weights)
            # 3. theta_pi = theta_pi - train_lr * grad
            fast_weights = list(
                map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights))
            )

            logits_q = net(x_qry, fast_weights, bn_training=True)
            # loss_q will be overwritten and just keep the loss_q on last update step.
            loss_q = F.cross_entropy(logits_q, y_qry)

            with torch.no_grad():
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                all_pred_q.append(pred_q.cpu().numpy())
                all_y_qry.append(y_qry.cpu().numpy())
                precision, recall, f1, _ = precision_recall_fscore_support(
                    y_qry.cpu().numpy(),
                    pred_q.cpu().numpy(),
                    average="weighted",
                    zero_division=0,
                )
                correct = torch.eq(pred_q, y_qry).sum().item()  # convert to numpy
                corrects[k + 1] = corrects[k + 1] + correct
                precisions[k + 1] = precisions[k + 1] + precision
                recalls[k + 1] = recalls[k + 1] + recall
                f1s[k + 1] = f1s[k + 1] + f1
        del net

        accs = np.array(corrects) / querysz

        # Flatten the lists of predictions and true labels
        all_pred_q = np.concatenate(all_pred_q)
        all_y_qry = np.concatenate(all_y_qry)

        # Calculate precision, recall, and F1 score

        return accs, precisions, recalls, f1s


def main():
    pass


if __name__ == "__main__":
    main()
