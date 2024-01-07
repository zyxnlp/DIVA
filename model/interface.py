import torch
from torch.nn import functional as F
from transformers.modeling_utils import PreTrainedModel
from torch import nn
from torch.nn.init import xavier_uniform_
from transformers import (AutoModel, AutoTokenizer, set_seed, AutoConfig,AutoModelForMaskedLM,activations,AdamW, get_linear_schedule_with_warmup,BigBirdConfig,BigBirdTokenizer,BigBirdModel)
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss,KLDivLoss
import numpy as np

import pytorch_lightning as pl



ACT2FN = activations.ACT2FN

# Bert MLM head interface
class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

class BertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states

class BertOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)

    def forward(self, sequence_output: torch.Tensor) -> torch.Tensor:
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores




class VaeLayer(PreTrainedModel):
    def __init__(self, config, hidden_size, latent_size, var_scale, eta_bn_prop, **kwargs):
        super().__init__(config, **kwargs)

        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.prior_covar_weights = None
        self.var_scale = var_scale
        self.eta_bn_prop = eta_bn_prop

        self.encoder_layer_c = nn.Linear(self.hidden_size, self.latent_size, bias=True)
        self.encoder_layer_t = nn.Linear(self.hidden_size, self.latent_size, bias=True)
        self.encoder_layer_y = nn.Linear(self.hidden_size, self.latent_size, bias=True)

        self.encoder_dropout_layer = nn.Dropout(p=0.2)
        xavier_uniform_(self.encoder_layer_c.weight)
        xavier_uniform_(self.encoder_layer_t.weight)
        xavier_uniform_(self.encoder_layer_y.weight)

        # create the mean and variance components of the VAE
        self.mean_layer_c = nn.Linear(self.latent_size, self.latent_size)
        self.logvar_layer_c = nn.Linear(self.latent_size, self.latent_size)

        self.mean_layer_t = nn.Linear(self.latent_size, self.latent_size)
        self.logvar_layer_t = nn.Linear(self.latent_size, self.latent_size)

        self.mean_layer_y = nn.Linear(self.latent_size, self.latent_size)
        self.logvar_layer_y = nn.Linear(self.latent_size, self.latent_size)



        self.mean_bn_layer = nn.BatchNorm1d(self.latent_size, eps=0.001, momentum=0.001, affine=True)
        self.mean_bn_layer.weight.data.copy_(torch.ones(self.latent_size))
        self.mean_bn_layer.weight.requires_grad = False
        self.logvar_bn_layer = nn.BatchNorm1d(self.latent_size, eps=0.001, momentum=0.001, affine=True)
        self.logvar_bn_layer.weight.data.copy_(torch.ones(self.latent_size))
        self.logvar_bn_layer.weight.requires_grad = False

        self.z_dropout_layer = nn.Dropout(p=0.2)

        # create the decoder
        self.decoder_layer = nn.Linear(self.latent_size*3, self.hidden_size, bias=True)
        # self.bate_layer = nn.Linear(self.latent_size, self.hidden_size)
        xavier_uniform_(self.decoder_layer.weight)

        # create a final batchnorm layer
        self.eta_bn_layer = nn.BatchNorm1d(self.hidden_size, eps=0.001, momentum=0.001, affine=True)
        self.eta_bn_layer.weight.data.copy_(torch.ones(self.hidden_size))
        self.eta_bn_layer.weight.requires_grad = False



    def encoder(self, input):
        en0_x_c = self.encoder_layer_c(input)
        en0_x_t = self.encoder_layer_t(input)
        en0_x_y = self.encoder_layer_y(input)

        encoder_output_c = F.softplus(en0_x_c)
        encoder_output_t = F.softplus(en0_x_t)
        encoder_output_y = F.softplus(en0_x_y)
        encoder_output_do_c = self.encoder_dropout_layer(encoder_output_c)
        encoder_output_do_t = self.encoder_dropout_layer(encoder_output_t)
        encoder_output_do_y = self.encoder_dropout_layer(encoder_output_y)

        posterior_mean_c = self.mean_layer_c(encoder_output_do_c)
        posterior_logvar_c = self.logvar_layer_c(encoder_output_do_c)
        posterior_mean_t = self.mean_layer_t(encoder_output_do_t)
        posterior_logvar_t = self.logvar_layer_t(encoder_output_do_t)
        posterior_mean_y = self.mean_layer_y(encoder_output_do_y)
        posterior_logvar_y = self.logvar_layer_y(encoder_output_do_y)

        posterior_mean_bn_c = self.mean_bn_layer(posterior_mean_c)
        posterior_logvar_bn_c = self.logvar_bn_layer(posterior_logvar_c)
        posterior_mean_bn_t = self.mean_bn_layer(posterior_mean_t)
        posterior_logvar_bn_t = self.logvar_bn_layer(posterior_logvar_t)
        posterior_mean_bn_y = self.mean_bn_layer(posterior_mean_y)
        posterior_logvar_bn_y = self.logvar_bn_layer(posterior_logvar_y)

        posterior_var_c = posterior_logvar_bn_c.exp()
        posterior_var_t = posterior_logvar_bn_t.exp()
        posterior_var_y = posterior_logvar_bn_y.exp()
        eps_c = input.data.new().resize_as_(posterior_mean_bn_c.data).normal_()
        eps_t = input.data.new().resize_as_(posterior_mean_bn_t.data).normal_()
        eps_y = input.data.new().resize_as_(posterior_mean_bn_y.data).normal_()

        z_c = posterior_mean_bn_c + posterior_var_c.sqrt() * eps_c * self.var_scale
        z_t = posterior_mean_bn_t + posterior_var_t.sqrt() * eps_t * self.var_scale
        z_y = posterior_mean_bn_y + posterior_var_y.sqrt() * eps_y * self.var_scale

        return (z_c,z_t,z_y),(posterior_mean_c,posterior_mean_t,posterior_mean_y),(posterior_logvar_c,posterior_logvar_t,posterior_logvar_y)


    def decoder(self, z):
        theta = self.z_dropout_layer(z)

        X_recon_no_bn = self.decoder_layer(theta)
        X_recon_bn = self.eta_bn_layer(X_recon_no_bn)
        return self.eta_bn_prop * X_recon_bn + (1.0 - self.eta_bn_prop) * X_recon_no_bn

class DivaModel(pl.LightningModule):
    def __init__(self, args):
        super(DivaModel,self).__init__()
        self.args = args

        self.use_mlm = bool(args.use_mlm)
        self.use_ort = bool(args.use_ort)
        self.use_mmd = bool(args.use_mmd)


        self.dropout = nn.Dropout(self.args.dropout)
        self.encoder_model = f"{self.args.bert_folder}"


        """Init config, pre-trained model, and tokenizer"""
        self.config = AutoConfig.from_pretrained(self.encoder_model)
        self.tokenizer = AutoTokenizer.from_pretrained(self.encoder_model)
        self.encoder_bert = AutoModel.from_pretrained(self.encoder_model,num_labels=2)  # maybe consider the causal language model

        if self.args.use_mlm:
            self.cls_mlm = BertOnlyMLMHead(config=self.config)  # change this if the model is not bigbird

        self.num_labels = self.encoder_bert.config.num_labels


        "Init vae encoder"
        self.hid_dim = self.config.hidden_size
        self.lat_dim = self.args.latent_dim

        self.alpha =1.0
        prior_mean = (np.log(self.alpha) - np.mean(np.log(self.alpha)))
        prior_var = (((1.0 / self.alpha) * (1 - (2.0 / self.args.latent_dim))) + (1.0 / (self.args.latent_dim * self.args.latent_dim)) * np.sum(1.0 /self. alpha))
        self.prior_mean = torch.tensor(prior_mean).to(self.device)
        self.prior_var = torch.tensor(prior_var).to(self.device)
        self.prior_logvar = self.prior_var.log().to(self.device)
        self.prior_mean.requires_grad = False
        self.prior_logvar.requires_grad = False

        self.c_sect_embeddings = nn.Embedding(12, self.lat_dim)
        self.c_size_embeddings = nn.Embedding(2, self.lat_dim)

        if self.args.task =="mov":
            self.vae = VaeLayer(self.config, hidden_size=self.hid_dim, latent_size=self.lat_dim, var_scale=0.1, eta_bn_prop=0.0)

            self.q_t0_cls = nn.Sequential(nn.Linear(self.lat_dim*4,self.lat_dim*4), #  z_c +z_y + c_sect + c_size
                                       nn.Sigmoid(),#
                                          nn.Linear(self.lat_dim * 4, self.lat_dim * 4),
                                          self.dropout,
                                      nn.LayerNorm(self.lat_dim * 4, eps=self.config.layer_norm_eps),
                                          self.dropout,
                                      nn.Linear(self.lat_dim * 4,2))

            self.q_t1_cls = nn.Sequential(nn.Linear(self.lat_dim*4,self.lat_dim*4), #  z_c +z_y + c_sect + c_size
                                       nn.Sigmoid(),#
                                          nn.Linear(self.lat_dim * 4, self.lat_dim * 4),
                                          self.dropout,
                                      nn.LayerNorm(self.lat_dim * 4, eps=self.config.layer_norm_eps),
                                          self.dropout,
                                      nn.Linear(self.lat_dim * 4,2))

        elif self.args.task =="vol":
            self.vae = VaeLayer(self.config, hidden_size=self.hid_dim, latent_size=self.lat_dim, var_scale=0.1,
                                     eta_bn_prop=0.0)

            self.q_t0_cls = nn.Sequential(nn.Linear(self.lat_dim*4,self.lat_dim*4), #  z_c +z_y + c_sect + c_size
                                       nn.Sigmoid(),#
                                          nn.Linear(self.lat_dim * 4, self.lat_dim * 4),
                                          self.dropout,
                                      nn.LayerNorm(self.lat_dim * 4, eps=self.config.layer_norm_eps),
                                          self.dropout,
                                      nn.Linear(self.lat_dim * 4,1))

            self.q_t1_cls = nn.Sequential(nn.Linear(self.lat_dim*4,self.lat_dim*4), #  z_c +z_y + c_sect + c_size
                                       nn.Sigmoid(),#
                                          nn.Linear(self.lat_dim * 4, self.lat_dim * 4),
                                          self.dropout,
                                      nn.LayerNorm(self.lat_dim * 4, eps=self.config.layer_norm_eps),
                                          self.dropout,
                                      nn.Linear(self.lat_dim * 4,1))


        self.g_cls = nn.Sequential(nn.Linear(self.lat_dim*4,self.lat_dim*4), #  z_c +z_y + c_sect + c_size
                                       nn.ReLU(),#
                                          nn.Linear(self.lat_dim * 4, self.lat_dim * 4),
                                          self.dropout,
                                      nn.LayerNorm(self.lat_dim * 4, eps=self.config.layer_norm_eps),
                                          self.dropout,
                                      nn.Linear(self.lat_dim * 4,2))

        self.g_cls_y = nn.Sequential(nn.Linear(self.lat_dim,self.lat_dim), #  z_c +z_y + c_sect + c_size
                                       nn.ReLU(),#
                                          nn.Linear(self.lat_dim , self.lat_dim ),
                                          self.dropout,
                                      nn.LayerNorm(self.lat_dim , eps=self.config.layer_norm_eps),
                                          self.dropout,
                                      nn.Linear(self.lat_dim ,2))


    def __kld(self, posterior_mean, posterior_logvar):

            kld = torch.mean(
                -0.5 * torch.sum(1 + posterior_logvar - posterior_mean ** 2 - posterior_logvar.exp(), dim=1), dim=0)

            #### ver0
            # prior_mean = self.prior_mean.expand_as(posterior_mean)
            # prior_var = self.prior_var.expand_as(posterior_logvar)
            # prior_mean = self.prior_mean
            # prior_var = self.prior_var
            # prior_logvar = self.prior_logvar
            #
            # posterior_var = posterior_logvar.exp()
            #
            # var_division = posterior_var / prior_var
            # diff = posterior_mean - prior_mean
            # diff_term = diff * diff / prior_var
            # logvar_division = prior_logvar - posterior_logvar
            #
            # kld = torch.mean(0.5 * (var_division + diff_term + logvar_division).sum(1))
            # kld = 0.5 * ((var_division + diff_term + logvar_division).sum(1) - 100)
            # kld = -0.5 * torch.sum(1 + posterior_logvar - posterior_mean.pow(2) - posterior_logvar.exp())

            return kld

    def __loss(self, x, x_recon, posterior_mean, posterior_logvar):

            recons_loss = F.mse_loss(x_recon, x)
            # kld
            posterior_mean_c, posterior_logvar_c = posterior_mean[0], posterior_logvar[0]
            posterior_mean_t, posterior_logvar_t = posterior_mean[1], posterior_logvar[1]
            posterior_mean_y, posterior_logvar_y = posterior_mean[2], posterior_logvar[2]

            kld_c = self.__kld(posterior_mean_c, posterior_logvar_c)
            kld_t = self.__kld(posterior_mean_t, posterior_logvar_t)
            kld_y = self.__kld(posterior_mean_y, posterior_logvar_y)

            # print('recons_loss',recons_loss)
            # print('kld_c',kld_c)
            # print('kld_t',kld_t)
            # print('kld_y',kld_y)
            # vae_loss = recons_loss + 0.01 * kld_c + 0.01 *kld_t+ 0.01* kld_y
            # vae_loss = recons_loss + 0.001 * kld_c + 0.001 * kld_t + 0.001 * kld_y
            vae_loss = recons_loss + kld_c +  kld_t + kld_y

            return vae_loss

    def gaussian_kernel(self, a, b):
            dim1_1, dim1_2 = a.shape[0], b.shape[0]
            depth = a.shape[1]
            a = a.view(dim1_1, 1, depth)
            b = b.view(1, dim1_2, depth)
            a_core = a.expand(dim1_1, dim1_2, depth)
            b_core = b.expand(dim1_1, dim1_2, depth)
            numerator = (a_core - b_core).pow(2).mean(2) / depth
            return torch.exp(-numerator)

    def MMD(self, a, b):
            return self.gaussian_kernel(a, a).mean() + self.gaussian_kernel(b, b).mean() - 2 * self.gaussian_kernel(a,
                                                                                                                    b).mean()


    def __ort_loss(self, z_t, z_c, z_y):
            ort_loss_t = torch.dist(z_c @ z_t.T, torch.eye(z_c.size(0)).to(self.device))
            ort_loss_y = torch.dist(z_c @ z_y.T, torch.eye(z_c.size(0)).to(self.device))
            ort_loss_c = torch.dist(z_t @ z_y.T, torch.eye(z_t.size(0)).to(self.device))
            ort_loss = ort_loss_t + ort_loss_y + ort_loss_c
            return ort_loss


    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                labels=None,
                C_sect=None,
                C_size=None,
                outcome_real=None,
                confounder_id=None,
                treatment_id=None,
                outcome_sim=None,
                outcome_ctf=None,
                special_tokens_mask=None,):

        outputs = self.encoder_bert(input_ids,
                             attention_mask=attention_mask,
                             token_type_ids=token_type_ids,)

        c_sect = self.c_sect_embeddings(C_sect)
        c_size = self.c_size_embeddings(C_size)

        c_embeddings = torch.cat((c_sect,c_size),dim=1)

        sequence_output = outputs[0] # last hidden state B * L * H


        if self.use_mlm and labels is not None:#
            mlm_loss = None
            # sequence_output = outputs[0]  # last hidden state B * L * H
            prediction_scores = self.cls_mlm(sequence_output)
            loss_fct = CrossEntropyLoss()  # -100 index = padding token,
            mlm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
        else:
            mlm_loss = 0

        pooled_output = outputs[1] #

        (z_c,z_t,z_y), (mu_c,mu_t,mu_y),(log_var_c,log_var_t,log_var_y) = self.vae.encoder(input=pooled_output) # B * 32


        t0_indices = (treatment_id == 0).nonzero().squeeze()
        t1_indices = (treatment_id == 1).nonzero().squeeze()

        if self.use_ort:
            ort_loss = self.__ort_loss(z_t,z_c,z_y)
        else:
            ort_loss = 0

        ### mmd loss
        if self.use_mmd:
            if t0_indices.view(-1).shape[0] <= 1 or t1_indices.view(-1).shape[0] <= 1:
                mmd_loss = 0
            else:
                # mmd_loss = self.MMD(pooled_output[t0_indices], pooled_output[t1_indices])
                mmd_loss = self.MMD(z_c[t0_indices], z_c[t1_indices]) + self.MMD(z_t[t0_indices], z_t[t1_indices]) + self.MMD(z_y[t0_indices], z_y[t1_indices])
        else:
            mmd_loss = 0

        z_cy = torch.cat((z_c,z_y, c_embeddings), dim=1)
        z_cy = self.dropout(z_cy)

        q_t0 = self.q_t0_cls(z_cy)
        q_t1 = self.q_t1_cls(z_cy)


        z_tc = torch.cat((z_c,z_t,c_embeddings), dim=1)
        z_tc = self.dropout(z_tc)

        g_logit = self.g_cls(z_tc)


        g_logit_y =self.g_cls_y(z_y)



        outcome = outcome_sim

        if self.args.task =="mov":
            t0_outcomes = outcome.clone().scatter(0, t1_indices, -100)
            t1_outcomes = outcome.clone().scatter(0, t0_indices, -100)

            q_t0_loss = CrossEntropyLoss()(q_t0.view(-1,self.num_labels), t0_outcomes)
            q_t1_loss = CrossEntropyLoss()(q_t1.view(-1,self.num_labels), t1_outcomes)


        elif self.args.task =="vol":
            t0_outcomes = outcome.clone().scatter(0, t1_indices, 0)
            t1_outcomes = outcome.clone().scatter(0, t0_indices, 0)

            q0_outcomes = q_t0.view(-1).clone().scatter(0, t1_indices, 0)
            q1_outcomes = q_t1.view(-1).clone().scatter(0, t0_indices, 0)

            q_t0_loss = MSELoss()(q0_outcomes, t0_outcomes)
            q_t1_loss = MSELoss()(q1_outcomes, t1_outcomes)



        q_loss = q_t0_loss + q_t1_loss


        g_loss = CrossEntropyLoss()(g_logit.view(-1,self.num_labels), treatment_id.view(-1))
        g_loss_y = CrossEntropyLoss()(g_logit_y.view(-1,self.num_labels), treatment_id.view(-1))

        z = torch.cat((z_c,z_t,z_y),dim=1) # B * 64

        pooled_output_recon = self.vae.decoder(z=z) # B * 768

        vae_loss = self.__loss(pooled_output, pooled_output_recon,(mu_c,mu_t,mu_y),(log_var_c,log_var_t,log_var_y))


        return {'g_loss_y':g_loss_y,'q_loss':q_loss,'q_t0_loss':q_t0_loss,'q_t1_loss':q_t1_loss,'mlm_loss':mlm_loss,'vae_loss':vae_loss,'g_loss':g_loss,'ort_loss':ort_loss,'mmd_loss':mmd_loss},\
            {'g_logit_y':g_logit_y,'q_t0_logit':q_t0,'q_t1_logit':q_t1,'g_logit':g_logit}

