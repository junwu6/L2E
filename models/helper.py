from torch.autograd import Function
import torch
import numpy as np
from models import mmd


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class L2ProjFunction(torch.autograd.Function):
    """
    This function defines the L2 projection for the input z.
    The forward pass uses a binary search and saves some quantities for the backward pass.
    The backward pass computes the Jacobian-vector product as in the Appendix of the paper.
    Note: if the forward pass loops forever, you may relax the termination condition a little bit.
    """

    @staticmethod
    def forward(self, z, dim=-1):

        z = z.transpose(dim, -1)
        left = torch.min(z, dim=-1, keepdim=True)[0]
        right = torch.max(z, dim=-1, keepdim=True)[0] + 1.0
        alpha_norm = torch.tensor(100.0, dtype=torch.float, device=z.device)
        one = torch.tensor(1.0, dtype=torch.float, device=z.device)
        # zero = torch.tensor(0.0, dtype=torch.float, device=z.device)
        # while not torch.allclose(right - left, zero):
        while not torch.allclose(alpha_norm, one):
            mid = left + (right - left) * 0.5
            alpha = torch.relu(mid - z)
            alpha_norm = torch.norm(alpha, dim=-1, keepdim=True)
            right[alpha_norm > 1.0] = mid[alpha_norm > 1.0]
            left[alpha_norm <= 1.0] = mid[alpha_norm <= 1.0]
        K = alpha.sum(-1, keepdim=True)
        alpha = alpha / K
        s = (alpha > 0).float()  # support, positivity mask
        zs = z * s
        S = s.sum(-1, keepdim=True)
        A = zs.sum(-1, keepdim=True) ** 2 - S * ((zs ** 2).sum(-1, keepdim=True) - 1)  # should have A > 0
        self.save_for_backward(alpha, K, s, S, A, torch.tensor(dim))
        return alpha.transpose(dim, -1)

    @staticmethod
    def backward(self, grad_output):

        alpha, K, s, S, A, dim = self.saved_tensors
        dim = dim.item()
        grad_output = grad_output.transpose(dim, -1)
        # first part
        vhat = (s * grad_output).sum(-1, keepdim=True) / S
        grad1 = (s / K) * (vhat - grad_output)
        # second part
        alpha_s = alpha * s - s / S
        grad2 = S / A.sqrt() * alpha_s * (alpha_s * grad_output).sum(-1, keepdim=True)

        return (grad1 - grad2).transpose(dim, -1)


def np_softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def np_sigmoid(x):
    """Numerically stable sigmoid function."""
    y = np.zeros_like(x, dtype=np.float32)
    y[x >= 0] = 1. / (1. + np.exp(-x[x >= 0]))
    y[x < 0] = np.exp(x[x < 0]) / (1. + np.exp(x[x < 0]))
    return y


def get_mask(d):  # d is batch x num_domain (one-hot)
    masks = np.zeros((d.shape[0], d.shape[1]), dtype=np.float32)
    for i in range(d.shape[1]):
        sum_d = max(np.sum(d[:, i]), 1)  # at least one sample for that domain i
        masks[:, i] = d[:, i] / sum_d
    return masks


def get_fspecific_domain_weights(d, pred, target_idx):
    masks = get_mask(d)  # compute average mask
    num_domains = d.shape[1]
    fdomain_weights = np.zeros((num_domains,), dtype=np.float32)
    for i in range(num_domains):
        if pred is None:
            fdomain_weights[i] = 1.
        else:
            # this is E_{D_s} f_s - E_{D_/s} f_s
            fdomain_weights[i] = np.sum((masks[:, i] - masks[:, target_idx]) * pred)
    fdomain_weights = np_sigmoid(fdomain_weights)  # this seems to be different from the paper...
    fdomain_weights[target_idx] = 1
    return fdomain_weights.astype('float32')


def compute_weights(d, pred, batch_size):

    num_domains = d.shape[1]
    num_tot_sample = num_domains * batch_size

    domain_weights = np.zeros((num_domains, num_domains), dtype=np.float32)
    for i in range(num_domains):
        domain_weights[:, i] = get_fspecific_domain_weights(d, pred[:, i], i)
        # Note: the i has weight 1, others have sigmoid weights

    f_weights = np.zeros((num_domains,), dtype=np.float32)
    for i in range(num_domains):
        temp = np.repeat(np.reshape(domain_weights[:, i], (1, num_domains)), num_tot_sample, axis=0)
        masks = get_mask(d)
        masks[:, i] = -masks[:, i]
        temp = np.sum(temp * masks, axis=1)
        f_weights[i] = np.sum(temp * pred[:, i].reshape(-1))  # why reshape?
    t_idx = -1
    f_weights[t_idx] = -1000
    f_weights = np_softmax(f_weights)
    f_weights[t_idx] = 1

    weights = np.zeros(d.shape, dtype=np.float32)
    for i in range(num_domains):
        domain_weights_repeat = np.repeat(np.reshape(domain_weights[:, i], (1, num_domains)),
                                          num_tot_sample, axis=0)
        masks = get_mask(d)
        masks[:, i] = -masks[:, i]
        temp = d * masks * domain_weights_repeat
        weights[:, i] = f_weights[i] * np.sum(temp, axis=1)

    return weights, f_weights[:-1]


def msda_regulizer(features_s, features_t, moment_order=4):

    def euclidean(x1, x2):
        return ((x1 - x2) ** 2).sum().sqrt()

    n_domains = len(features_s)
    moment_reg = 0.
    features_power_s, features_power_t = list(features_s), features_t
    for k in range(moment_order):
        # compute moment difference then multiply to get the next moment
        for d1 in range(n_domains):
            moment_reg = moment_reg + euclidean(features_power_t.mean(0),
                                                features_power_s[d1].mean(0))
            for d2 in range(d1 + 1, n_domains):
                moment_reg = moment_reg + euclidean(features_power_s[d1].mean(0),
                                                    features_power_s[d2].mean(0))
            features_power_s[d1] = features_power_s[d1] * features_s[d1]
        features_power_t = features_power_t * features_t

    return moment_reg / (moment_order*n_domains*n_domains)
