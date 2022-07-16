import torch.nn as nn
from torch.autograd import Function
from torchvision import models
import numpy as np
import torch
logc = np.log(2.*np.pi)
c = - 0.5 * np.log(2*np.pi)


def estimate_normal_logpdf(x, mu, log_sigma_sq):
    return - 0.5 * logc - log_sigma_sq / 2. - torch.div(torch.square(torch.sub(x, mu)), 2 * torch.exp(log_sigma_sq))


def estimate_gaussian_marg(mu, log_sigma_sq):
    return - 0.5 * (logc + (torch.square(mu) + torch.exp(log_sigma_sq)))


def estimate_gaussian_ent(log_sigma_sq):
    return - 0.5 * (logc + 1.0 + log_sigma_sq)


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class TransLATEModel(nn.Module):
    def __init__(self, num_classes, option='resnet18', use_bottleneck=True, bottleneck_width=256, width=1024):
        super(TransLATEModel, self).__init__()
        self.dim = 2048
        if option == 'resnet18':
            model_ft = models.resnet18(pretrained=True)
            self.dim = 512
        if option == 'resnet50':
            model_ft = models.resnet50(pretrained=True)
        if option == 'resnet101':
            model_ft = models.resnet101(pretrained=True)
        if option == 'resnet152':
            model_ft = models.resnet152(pretrained=True)
        mod = list(model_ft.children())
        mod.pop()
        self.base_network = nn.Sequential(*mod)
        self.use_bottleneck = use_bottleneck
        self.num_classes = num_classes
        self.mu = 1.0
        hidden_dim = bottleneck_width if self.use_bottleneck else self.dim
        self.dim_x = hidden_dim
        self.dim_z = 2048

        self.bottleneck_layer = nn.Sequential(
            nn.Linear(self.dim, bottleneck_width),
            nn.BatchNorm1d(bottleneck_width),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.classifier_layer = nn.Sequential(
            nn.Linear(hidden_dim, width),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(width, num_classes)
        )

        self.discriminator = nn.Sequential(
            nn.Linear(self.dim_z, width),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(width, 2)
        )
        self.criterion = nn.CrossEntropyLoss()

        # vae
        self.encoder_z = nn.Sequential(
            nn.Linear(self.dim_x + self.num_classes, self.dim_z*2),
        )
        self.decoder_x = nn.Sequential(
            nn.Linear(self.dim_z + self.num_classes, self.dim_x*2),
        )
        self.decoder_y = nn.Sequential(
            nn.Linear(self.dim_z, self.num_classes*2),
        )

    def forward(self, s_inputs, s_outputs, t_inputs, alpha):
        # t_feats = self.base_network(t_inputs).view(t_inputs.shape[0], -1)
        # if self.use_bottleneck:
        #     t_feats = self.bottleneck_layer(t_feats)
        #
        # loss = 0.
        # for i in range(len(s_inputs)):
        #     s_feats = self.base_network(s_inputs[i]).view(s_inputs[i].shape[0], -1)
        #     if self.use_bottleneck:
        #         s_feats = self.bottleneck_layer(s_feats)
        #     s_preds = self.classifier_layer(s_feats)
        #     class_loss = self.criterion(s_preds, s_outputs[i])

        all_s_inputs = torch.cat(s_inputs, dim=0)
        all_s_feats = self.base_network(all_s_inputs).view(all_s_inputs.shape[0], -1)
        t_feats = self.base_network(t_inputs).view(t_inputs.shape[0], -1)
        if self.use_bottleneck:
            all_s_feats = self.bottleneck_layer(all_s_feats)
            t_feats = self.bottleneck_layer(t_feats)

        loss = 0.0
        for i in range(len(s_inputs)):
            s_feats = all_s_feats[s_inputs[i].shape[0] * i:s_inputs[i].shape[0] * (i + 1)]
            s_preds = self.classifier_layer(s_feats)
            class_loss = self.criterion(s_preds, s_outputs[i])

            vae_loss, all_z = self.label_informed_DA(s_feats, s_outputs[i], t_feats)
            domain_preds = self.discriminator(ReverseLayerF.apply(all_z, alpha))
            domain_labels = np.array([0] * s_inputs[i].shape[0] + [1] * t_feats.shape[0])
            domain_labels = torch.tensor(domain_labels, requires_grad=False, dtype=torch.long, device=s_inputs[i].device)
            domain_loss = self.criterion(domain_preds, domain_labels)
            loss += (class_loss + domain_loss + vae_loss*0.1) * (self.mu**(len(s_inputs)-i-1))
        return loss/len(s_inputs)

    def label_informed_DA(self, s_feats, s_outputs, t_feats):
        ''' Labelled data points '''
        x = s_feats
        y = self.one_hot(s_outputs, label_size=self.num_classes)
        z, mu, logsigma = self._generate_zxy(x, y)
        x_recon_mu, x_recon_sigma = self._generate_xzy(z, y)
        y_recon_mu, y_recon_sigma = self._generate_yz(z)
        vae_loss = - estimate_normal_logpdf(x, x_recon_mu, x_recon_sigma).mean(1) \
                   - estimate_normal_logpdf(y, y_recon_mu, y_recon_sigma).mean(1) \
                   - estimate_gaussian_marg(mu, logsigma).mean(1) \
                   + estimate_gaussian_ent(logsigma).mean(1)
        vae_loss = vae_loss.mean(0)

        ''' Unabelled data points '''
        x_unlab = t_feats
        psudo_unlab_y = self.classifier_layer(x_unlab)
        psudo_unlab_y = torch.nn.functional.softmax(psudo_unlab_y, dim=1)
        unlab_vae_loss = 0.
        for label in range(self.num_classes):
            _y_ulab = self.one_hot(torch.tensor([label]*x_unlab.shape[0], requires_grad=False, device=s_feats.device), label_size=self.num_classes)
            z_ulab, z_ulab_mu, z_ulab_sigma = self._generate_zxy(x_unlab, _y_ulab)
            x_recon_mu_unlab, x_recon_sigma_unlab = self._generate_xzy(z_ulab, _y_ulab)
            y_recon_mu_unlab, y_recon_sigma_unlab = self._generate_yz(z_ulab)
            _L_ulab = - estimate_normal_logpdf(x_unlab, x_recon_mu_unlab, x_recon_sigma_unlab).mean(1) \
                      - estimate_normal_logpdf(_y_ulab, y_recon_mu_unlab, y_recon_sigma_unlab).mean(1) \
                      - estimate_gaussian_marg(z_ulab_mu, z_ulab_sigma).mean(1) \
                      + estimate_gaussian_ent(z_ulab_sigma).mean(1)
            q_y_x = psudo_unlab_y[:, label]
            unlab_vae_loss += torch.mul(q_y_x, torch.sub(_L_ulab, torch.log(q_y_x))).mean(0)
        vae_loss += unlab_vae_loss

        '''Latent feature'''
        psudo_all_y = torch.cat([y, psudo_unlab_y], dim=0)
        all_z, _, _ = self._generate_zxy(torch.cat([x, x_unlab], dim=0), psudo_all_y)

        return vae_loss, all_z

    def _generate_zxy(self, x, y):
        xy = torch.cat([x, y], dim=1)
        mu, logsigma = self.encoder_z(xy).chunk(2, dim=-1)
        sample = mu + torch.mul(torch.exp(logsigma / 2), torch.empty([x.shape[0], self.dim_z], dtype=torch.float32, device=x.device).normal_(mean=0, std=1))
        return sample, mu, logsigma

    def _generate_xzy(self, z, y):
        yz = torch.cat([y, z], dim=1)
        x_recon_mu, x_recon_sigma = self.decoder_x(yz).chunk(2, dim=-1)
        return x_recon_mu, x_recon_sigma

    def _generate_yz(self, z):
        y_recon_mu, y_recon_sigma = self.decoder_y(z).chunk(2, dim=-1)
        return y_recon_mu, y_recon_sigma

    def one_hot(self, x, label_size):
        out = torch.zeros(len(x), label_size).to(x.device)
        out[torch.arange(len(x)), x.squeeze()] = 1
        return out

    def inference(self, x):
        x = self.base_network(x).view(x.shape[0], -1)
        if self.use_bottleneck:
            x = self.bottleneck_layer(x)
        return self.classifier_layer(x)
