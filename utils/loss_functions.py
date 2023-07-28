import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys

sys.path.insert(
    0, "../../timeline_generation/"
)  # Adds higher directory to python modules path

from utils.io.data_handler import remove_padding_from_tensor


def get_focal_alphas_from_labels(labels):
    """
    Requires as input all the labels (3D array) from the dataset
    (e.g. all from training set).

    Pass the output of this function as the argument alpha= for the FocalLoss class.
    """

    # Focal alpha:
    tr_ = np.array(labels)
    class_proba = [
        len(np.where(tr_ == 0)[0]) / len(tr_),
        len(np.where(tr_ == 1)[0]) / len(tr_),
        len(np.where(tr_ == 2)[0]) / len(tr_),
    ]
    inv_class_proba = np.sqrt(1 / np.array(class_proba))
    alphas_focal = list(inv_class_proba)

    return alphas_focal


def create_loss_function(
    train_loader,
    val_loader,
    which_loss="cross_entropy",
    gamma: float = 2.0,  # For focal loss.
    beta: float = -0.9999,  # For class balanced losses.
):
    """
    Returns 2 loss functions, the first for the training set and second for the
    validation set. If is focal loss, will require as input the dataset to
    compute alphas (dependent on number of classes in datasets).
    """

    if which_loss == "focal_loss":
        loss_fn_train = create_focal_loss_from_data_loader(
            data_loader=train_loader, gamma=gamma
        )
        loss_fn_validation = create_focal_loss_from_data_loader(
            data_loader=val_loader, gamma=gamma
        )

    elif which_loss == "class_balanced_focal_loss":
        loss_fn_train = create_class_balanced_loss_from_data_loader(
            train_loader, beta=beta, gamma=gamma, loss_type="focal"
        )
        loss_fn_validation = create_class_balanced_loss_from_data_loader(
            val_loader, beta=beta, gamma=gamma, loss_type="focal"
        )
    elif which_loss == "class_balanced_cross_entropy":
        loss_fn_train = create_class_balanced_loss_from_data_loader(
            train_loader, beta=beta, gamma=gamma, loss_type="cross_entropy"
        )
        loss_fn_validation = create_class_balanced_loss_from_data_loader(
            val_loader, beta=beta, gamma=gamma, loss_type="cross_entropy"
        )

    # Cross-entropy as default
    else:
        loss_fn_train = nn.CrossEntropyLoss()
        loss_fn_validation = nn.CrossEntropyLoss()

    return loss_fn_train, loss_fn_validation


def create_focal_loss_from_data_loader(data_loader, gamma=2):

    labels = get_labels_from_dataloader_for_focal_loss(data_loader)
    alphas = get_focal_alphas_from_labels(labels)
    loss_fn = FocalLoss(gamma=gamma, alpha=alphas)

    return loss_fn


def get_labels_from_dataloader_for_focal_loss(data_loader, remove_padding=True):
    """
    Simply returns the labels in the correct order, from a dataloader.
    """

    y_trues = []
    for _, (_, y_true) in enumerate(data_loader):

        n_classes = 3

        # Output should be (N, C) - removes batch
        y_true = y_true.view(-1, n_classes)

        # Remove padding, if desired
        if remove_padding:
            # print("=== `get_labels_from_dataloader_for_focal_loss` ===")
            # print("y_true.shape:", y_true.shape)

            # Remove padding from test set. Padding removed automatically by model
            y_true = remove_padding_from_tensor(
                y_true, padding_value=-123.0, return_mask_only=False
            )
            # print("y_true.shape:", y_true.shape)

        # Label encoding
        y_true = torch.argmax(y_true, dim=1)

        # Store all predictions and true values, for all samples
        y_trues.append(y_true)

    labels = torch.cat(y_trues)

    return labels


def get_number_of_samples_per_class(labels: list):
    """
    Takes a flat input tensor, e.g. [0,1,2,2,2,2,1] and returns the number of
    labels per class [1, 2, 4].
    """

    n_samples_per_class = torch.bincount(labels)

    return n_samples_per_class


class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


def create_class_balanced_loss_from_data_loader(
    data_loader, beta=-0.9999, gamma=2, loss_type="focal"
):
    """
    Returns the loss function. Then use it on the appropriate training/ validation
    set as follows:

    loss = focal_loss(logits, labels)

    Args:
        data_loader (_type_): _description_
        predictions (_type_): _description_
        beta (float, optional): _description_. Defaults to 0.9999.
        gamma (int, optional): _description_. Defaults to 2.
        loss_type (str, optional): _description_. Defaults to "focal".

    Returns:
        _type_: _description_
    """

    labels = get_labels_from_dataloader_for_focal_loss(data_loader)
    samples_per_class = get_number_of_samples_per_class(labels)

    loss = ClassBalancedLoss(
        loss_type=loss_type,
        beta=beta,
        fl_gamma=gamma,
        samples_per_class=samples_per_class,  # Number of samples per class in the training/ validation dataset
        class_balanced=True,
    )

    return loss


def focal_loss(logits, labels, alpha=None, gamma: float = 2.0):
    """Compute the focal loss between `logits` and the ground truth `labels`.
    Focal loss = -alpha_t * (1-pt)^gamma * log(pt)
    where pt is the probability of being classified to the true class.
    pt = p (if true class), otherwise pt = 1 - p. p = sigmoid(logit).
    Args:
      logits: A float tensor of size [batch, num_classes].
      labels: A float tensor of size [batch, num_classes].
      alpha: A float tensor of size [batch_size]
        specifying per-example weight for balanced cross entropy.
      gamma: A float scalar modulating loss from hard and easy examples.
    Returns:
      focal_loss: A float32 scalar representing normalized total loss.
    """

    bc_loss = F.binary_cross_entropy_with_logits(
        input=logits, target=labels, reduction="none"
    )

    if gamma == 0.0:
        modulator = 1.0
    else:
        modulator = torch.exp(
            -gamma * labels * logits - gamma * torch.log(1 + torch.exp(-1.0 * logits))
        )

    loss = modulator * bc_loss

    if alpha is not None:
        weighted_loss = alpha * loss
        focal_loss = torch.sum(weighted_loss)
    else:
        focal_loss = torch.sum(loss)

    focal_loss /= torch.sum(labels)
    return focal_loss


class ClassBalancedLoss(torch.nn.Module):
    def __init__(
        self,
        loss_type: str = "focal",
        beta: float = 0.999,
        fl_gamma: float = 2.0,
        samples_per_class=None,  # Number of samples per class in the training/ validation dataset
        class_balanced=False,
    ):
        """
        Source: https://github.com/fcakyon/balanced-loss/blob/main/balanced_loss/losses.py

        Compute the Class Balanced Loss between `logits` and the ground truth `labels`.
        Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, logits)
        where Loss is one of the standard losses used for Neural Networks.
        reference: https://openaccess.thecvf.com/content_CVPR_2019/papers/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.pdf

        Args:
            loss_type: string. One of "focal_loss", "cross_entropy",
                "binary_cross_entropy", "softmax_binary_cross_entropy".

            beta: float. Hyperparameter for Class balanced loss.
            [0.9, 0.99, 0.999, 0.9999]. Controls how fast effective number of
            samples grow, as number of samples increases.

            fl_gamma: float. Hyperparameter for Focal loss.

            samples_per_class: A python list of size [num_classes].
                Required if class_balance is True.

            class_balanced: bool. Whether to use class balanced loss.

        Returns:
            Loss instance
        """
        super(ClassBalancedLoss, self).__init__()

        if class_balanced is True and samples_per_class is None:
            raise ValueError(
                "samples_per_class cannot be None when class_balanced is True"
            )

        self.loss_type = loss_type
        self.beta = beta
        self.fl_gamma = fl_gamma
        self.samples_per_class = samples_per_class
        self.class_balanced = class_balanced

    def forward(
        self,
        logits: torch.tensor,
        labels: torch.tensor,
    ):
        """
        Compute the Class Balanced Loss between `logits` and the ground truth `labels`.
        Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, logits)
        where Loss is one of the standard losses used for Neural Networks.
        Args:
            logits: A float tensor of size [batch, num_classes].
            labels: An int tensor of size [batch].
        Returns:
            cb_loss: A float tensor representing class balanced loss
        """

        batch_size = logits.size(0)
        num_classes = logits.size(1)
        labels_one_hot = F.one_hot(labels, num_classes).float()

        if self.class_balanced:
            effective_num = 1.0 - np.power(self.beta, self.samples_per_class)
            weights = (1.0 - self.beta) / np.array(effective_num)
            weights = weights / np.sum(weights) * num_classes
            weights = torch.tensor(weights, device=logits.device).float()

            if self.loss_type != "cross_entropy":
                weights = weights.unsqueeze(0)
                weights = weights.repeat(batch_size, 1) * labels_one_hot
                weights = weights.sum(1)
                weights = weights.unsqueeze(1)
                weights = weights.repeat(1, num_classes)
        else:
            weights = None

        if self.loss_type == "focal":
            cb_loss = self.focal_loss(
                logits, labels_one_hot, alpha=weights, gamma=self.fl_gamma
            )
        elif self.loss_type == "cross_entropy":
            cb_loss = F.cross_entropy(
                input=logits, target=labels_one_hot, weight=weights
            )
        elif self.loss_type == "binary_cross_entropy":
            cb_loss = F.binary_cross_entropy_with_logits(
                input=logits, target=labels_one_hot, weight=weights
            )
        elif self.loss_type == "softmax_binary_cross_entropy":
            pred = logits.softmax(dim=1)
            cb_loss = F.binary_cross_entropy(
                input=pred, target=labels_one_hot, weight=weights
            )

        return cb_loss

    def focal_loss(self, logits, labels, alpha=None, gamma: float = 2.0):
        """Compute the focal loss between `logits` and the ground truth `labels`.
        Focal loss = -alpha_t * (1-pt)^gamma * log(pt)
        where pt is the probability of being classified to the true class.
        pt = p (if true class), otherwise pt = 1 - p. p = sigmoid(logit).
        Args:
        logits: A float tensor of size [batch, num_classes].
        labels: A float tensor of size [batch, num_classes].
        alpha: A float tensor of size [batch_size]
            specifying per-example weight for balanced cross entropy.
        gamma: A float scalar modulating loss from hard and easy examples.
        Returns:
        focal_loss: A float32 scalar representing normalized total loss.
        """

        # TODO: include softmax in the loss function - this is for class balanced loss
        logits = F.log_softmax(logits, dim=-1)

        bc_loss = F.binary_cross_entropy_with_logits(
            input=logits, target=labels, reduction="none"
        )

        if gamma == 0.0:
            modulator = 1.0
        else:
            modulator = torch.exp(
                -gamma * labels * logits
                - gamma * torch.log(1 + torch.exp(-1.0 * logits))
            )

        loss = modulator * bc_loss

        if alpha is not None:
            weighted_loss = alpha * loss
            focal_loss = torch.sum(weighted_loss)
        else:
            focal_loss = torch.sum(loss)

        focal_loss /= torch.sum(labels)
        return focal_loss
