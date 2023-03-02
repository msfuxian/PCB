import mindspore
from mindspore import Tensor
from mindspore.ops import functional as F
from mindspore import nn
import mindspore.ops as ops
import mindspore.numpy as np
from mindspore.common.parameter import Parameter

from .accuracy import accuracy
from .cross_entropy_loss import cross_entropy


def nonzero(input):
    """Return indexes of all nonzero/True elements.
    """
    temp = Tensor(np.zeros(input.shape))
    if input.dtype == mindspore.bool_:
        for ri in input:
            for ci in ri:
                if ci == True:
                    temp[ri.index_of_parent_, ci.index_of_parent_] = 1
                else:
                    temp[ri.index_of_parent_, ci.index_of_parent_] = 0
    else:
        temp = input
    output = Tensor(np.zeros((ops.count_nonzero(temp).asnumpy().item(), 2)))

    r = -1
    ind = -1
    for rows in temp:
        c = -1
        r = r + 1
        for elem in rows:
            c = c + 1
            if elem != 0:
                ind = ind + 1
                output[ind, :] = Tensor([r, c])
    return output


def reduce_loss(loss, reduction):
    """Reduce loss as specified.
    Args:
        loss (Tensor): Elementwise loss tensor.
    Return:
        Tensor: Reduced loss tensor.
    """
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    return loss


def weight_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
    if weight is not None:
        loss = loss * weight
    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        loss = reduce_loss(loss,  reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == 'mean':
            loss = loss.sum() / avg_factor
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss


def seesaw_ce_loss(cls_score,
                   labels,
                   label_weights,
                   cum_samples,
                   num_classes,
                   p,
                   q,
                   eps,
                   reduction='mean',
                   avg_factor=None):
    """Calculate the Seesaw CrossEntropy loss.
    Args:
        cls_score (torch.Tensor): The prediction with shape (N, C),
             C is the number of classes.
        labels (torch.Tensor): The learning label of the prediction.
        label_weights (torch.Tensor): Sample-wise loss weight.
        cum_samples (torch.Tensor): Cumulative samples for each category.
        num_classes (int): The number of classes.
        p (float): The ``p`` in the mitigation factor.
        q (float): The ``q`` in the compenstation factor.
        eps (float): The minimal value of divisor to smooth
             the computation of compensation factor
        reduction (str, optional): The method used to reduce the loss.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
    Returns:
        torch.Tensor: The calculated loss
    """
    assert cls_score.size(-1) == num_classes
    assert len(cum_samples) == num_classes

    onehot_labels = F.one_hot(labels, num_classes, on_value=Tensor(1.0, mindspore.float32), off_value=Tensor(0.0, mindspore.float32))
    seesaw_weights = cls_score.new_ones(onehot_labels.size())

    # mitigation factor
    if p > 0:
        sample_ratio_matrix = cum_samples[None, :].clamp(
            min=1) / cum_samples[:, None].clamp(min=1)
        index = (sample_ratio_matrix < 1.0).float()
        sample_weights = sample_ratio_matrix.pow(p) * index + (1 - index)
        mitigation_factor = sample_weights[labels.long(), :]
        seesaw_weights = seesaw_weights * mitigation_factor

    # compensation factor
    if q > 0:
        scores = F.softmax(cls_score.detach())
        self_scores = scores[
            ops.arange(0, len(scores)).to(scores.device).long(),
            labels.long()]
        score_matrix = scores / self_scores[:, None].clamp(min=eps)
        index = (score_matrix > 1.0).float()
        compensation_factor = score_matrix.pow(q) * index + (1 - index)
        seesaw_weights = seesaw_weights * compensation_factor

    cls_score = cls_score + (seesaw_weights.log() * (1 - onehot_labels))

    loss = F.cross_entropy(cls_score, labels, weight=Tensor(1.0, mindspore.float32), reduction='none')

    if label_weights is not None:
        label_weights = label_weights.float()
    loss = weight_reduce_loss(
        loss, weight=label_weights, reduction=reduction, avg_factor=avg_factor)
    return loss

class SeesawLoss(nn.Cell):
    def __init__(self,
                 use_sigmoid=False,
                 p=0.8,
                 q=2.0,
                 num_classes=1203,
                 eps=1e-2,
                 reduction='mean',
                 loss_weight=1.0,
                 return_dict=True):
        super(SeesawLoss, self).__init__()
        assert not use_sigmoid
        self.use_sigmoid = False
        self.p = p
        self.q = q
        self.num_classes = num_classes
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.return_dict = return_dict

        # 0 for pos, 1 for neg
        self.cls_criterion = seesaw_ce_loss

        # cumulative samples for each category
        cum_samples = np.zeros(self.num_classes + 1, mindspore.float32)
        self.cum_samples = Parameter(cum_samples, name='cum_samples')

        # custom output channels of the classifier
        self.custom_cls_channels = True
        # custom activation of cls_score
        self.custom_activation = True
        # custom accuracy of the classsifier
        self.custom_accuracy = True

    def _split_cls_score(self, cls_score):
        # split cls_score to cls_score_classes and cls_score_objectness
        assert ops.shape(cls_score)[-1] == self.num_classes + 2
        cls_score_classes = cls_score[..., :-2]
        cls_score_objectness = cls_score[..., -2:]
        return cls_score_classes, cls_score_objectness

    def get_cls_channels(self, num_classes):
        assert num_classes == self.num_classes
        return num_classes + 2

    def get_activation(self, cls_score):
        cls_score_classes, cls_score_objectness = self._split_cls_score(
            cls_score)
        score_classes = F.softmax(cls_score_classes)
        score_objectness = F.softmax(cls_score_objectness)
        score_pos = score_objectness[..., [0]]
        score_neg = score_objectness[..., [1]]
        score_classes = score_classes * score_pos
        scores = ops.concat([score_classes, score_neg])
        return scores

    def get_accuracy(self, cls_score, labels):
        pos_inds = labels < self.num_classes
        obj_labels = int(labels == self.num_classes)
        obj_labels = Tensor(obj_labels, mindspore.int64)
        cls_score_classes, cls_score_objectness = self._split_cls_score(
            cls_score)
        acc_objectness = accuracy(cls_score_objectness, obj_labels)
        acc_classes = accuracy(cls_score_classes[pos_inds], labels[pos_inds])
        acc = dict()
        acc['acc_objectness'] = acc_objectness
        acc['acc_classes'] = acc_classes
        return acc

    def construct(self,
                cls_score,
                labels,
                label_weights=None,
                avg_factor=None,
                reduction_override=None):

        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        assert cls_score.size(-1) == self.num_classes + 2
        pos_inds = labels < self.num_classes
        # 0 for pos, 1 for neg
        obj_labels = int(labels == self.num_classes)
        obj_labels = Tensor(obj_labels, mindspore.int64)

        # accumulate the samples for each category
        unique_labels = labels.unique()
        for u_l in unique_labels:
            inds_ = labels == u_l.item()
            self.cum_samples[u_l] += np.sum(inds_)

        if label_weights is not None:
            label_weights = label_weights.float()
        else:
            label_weights = labels.new_ones(labels.size(), mindspore.float32)

        cls_score_classes, cls_score_objectness = self._split_cls_score(
            cls_score)
        # calculate loss_cls_classes (only need pos samples)
        if np.sum(pos_inds) > 0:
            loss_cls_classes = self.loss_weight * self.cls_criterion(
                cls_score_classes[pos_inds], labels[pos_inds],
                label_weights[pos_inds], self.cum_samples[:self.num_classes],
                self.num_classes, self.p, self.q, self.eps, reduction,
                avg_factor)
        else:
            loss_cls_classes = cls_score_classes[pos_inds].sum()
        # calculate loss_cls_objectness
        loss_cls_objectness = self.loss_weight * cross_entropy(
            cls_score_objectness, obj_labels, label_weights, reduction,
            avg_factor)

        if self.return_dict:
            loss_cls = dict()
            loss_cls['loss_cls_objectness'] = loss_cls_objectness
            loss_cls['loss_cls_classes'] = loss_cls_classes
        else:
            loss_cls = loss_cls_classes + loss_cls_objectness
        return loss_cls