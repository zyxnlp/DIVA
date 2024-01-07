# -*- coding: utf-8 -*-
from argparse import ArgumentParser, Namespace


import pytorch_lightning as pl
import torch

from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss,KLDivLoss
from torch import nn
import logging
from transformers import (AdamW, get_linear_schedule_with_warmup)

from estimator import ITE
import torchmetrics

from model.interface import DivaModel



logger = logging.getLogger(__name__)



class DIVA(pl.LightningModule):
    def __init__(self, args: Namespace) -> None:
        super().__init__()
        self.args = args
        self.loss_weights = {
            'q': args.q_weight,
            'mlm': args.mlm_weight,
            'g': args.g_weight,
            'mmd': args.mmd_weight,
            'ort': args.ort_weight,

        }

        self.model = DivaModel(args)



    def forward(self, **inputs):

        loss_dict, logit_dict = self.model(**inputs)

        return loss_dict, logit_dict


    def training_step(self, batch, batch_idx):

        loss_dict, logit_dict = self.forward(**batch)


        q_loss = loss_dict['q_loss']
        mlm_loss = loss_dict['mlm_loss']
        g_loss = loss_dict['g_loss']
        vae_loss = loss_dict['vae_loss']
        # mi_loss = loss_dict['mi_loss']
        ort_loss = loss_dict['ort_loss']
        mmd_loss = loss_dict['mmd_loss']
        g_loss_y = loss_dict['g_loss_y']
        loss = self.loss_weights['q'] * q_loss  + self.loss_weights['mlm'] * mlm_loss+vae_loss +self.loss_weights['g'] * g_loss\
               + self.loss_weights['ort'] * ort_loss + self.loss_weights['mmd'] * mmd_loss-10*g_loss_y

        self.log('loss', loss*0.001, on_step=True, on_epoch=False, prog_bar=True)

        return {'loss': loss, 'logit_dict':logit_dict, 'batch': batch,'g_loss':g_loss,'g_loss_y':g_loss_y}

    def training_epoch_end(self, outputs):
        pass




    def validation_step(self, batch, batch_idx):

        loss_dict, logit_dict  = self.forward(**batch)

        q_loss = loss_dict['q_loss']
        g_loss = loss_dict['g_loss']
        mlm_loss = loss_dict['mlm_loss']
        vae_loss = loss_dict['vae_loss']
        # mi_loss = loss_dict['mi_loss']
        ort_loss = loss_dict['ort_loss']
        mmd_loss = loss_dict['mmd_loss']
        g_loss_y = loss_dict['g_loss_y']

        #   g_loss_y weight
        val_loss = self.loss_weights['q'] * q_loss  + self.loss_weights['mlm'] * mlm_loss+ vae_loss+ self.loss_weights['g'] * g_loss\
                    + self.loss_weights['ort'] * ort_loss + self.loss_weights['mmd'] * mmd_loss- 10*g_loss_y



        self.log('v_loss', val_loss*0.001, on_step=False, on_epoch=True, prog_bar=True)

        return {'logit_dict':logit_dict, 'batch': batch,'g_loss':g_loss,'g_loss_y':g_loss_y}



    def validation_epoch_end(self,outputs):

        q0_pre = torch.cat([x['logit_dict']['q_t0_logit'] for x in outputs])
        q1_pre = torch.cat([x['logit_dict']['q_t1_logit'] for x in outputs])


        if self.args.task=="mov":

            q0_pre = q0_pre.argmax(dim=-1)
            q1_pre = q1_pre.argmax(dim=-1)

        elif self.args.task=="vol":
            q0_pre = q0_pre.view(-1)
            q1_pre = q1_pre.view(-1)


        T = torch.cat([x['batch']['treatment_id'] for x in outputs])

        T0_ind = (T == 0).nonzero().squeeze()
        T1_ind = (T == 1).nonzero().squeeze()

        Y = torch.cat([x['batch']['outcome_sim'] for x in outputs])


        if self.args.task=="mov":

            Y_t0 = Y.clone().scatter(0, T1_ind, 100)
            Y_t1 = Y.clone().scatter(0, T0_ind, 100)

            Q_t0 = q0_pre.clone().scatter(0, T1_ind, 100)
            Q_t1 = q1_pre.clone().scatter(0, T0_ind, 100)

            Q0_acc = torchmetrics.Accuracy(ignore_index=100).to(self.device)(Q_t0.detach().cpu(), Y_t0.detach().cpu())
            Q1_acc = torchmetrics.Accuracy(ignore_index=100).to(self.device)(Q_t1.detach().cpu(), Y_t1.detach().cpu())

            Q_acc = (Q0_acc + Q1_acc) / 2
            self.log('v_q_acc', Q_acc, on_step=False, on_epoch=True, prog_bar=False)


        elif self.args.task=="vol":
            Y_t0 = Y.clone().scatter(0, T1_ind, 0)
            Y_t1 = Y.clone().scatter(0, T0_ind, 0)

            Q_t0 = q0_pre.clone().scatter(0, T1_ind, 0)
            Q_t1 = q1_pre.clone().scatter(0, T0_ind, 0)

            Q0_mse = MSELoss()(Q_t0, Y_t0)
            Q1_mse = MSELoss()(Q_t1, Y_t1)

            Q_mse = (Q0_mse + Q1_mse) / 2
            self.log('v_q_mse', Q_mse, on_step=False, on_epoch=True, prog_bar=False)



    def test_step(self, batch, batch_idx): #
        # start testing
        output_dict ={}
        loss_dict, logit_dict = self.forward(**batch)
        output_dict = {'logit_dict': logit_dict, 'batch': batch}

        return output_dict

    def test_epoch_end(self,outputs):

        q0_t = torch.cat([x['logit_dict']['q_t0_logit'] for x in outputs])
        q1_t = torch.cat([x['logit_dict']['q_t1_logit'] for x in outputs])


        if self.args.task=="mov":

            sm = nn.Softmax(dim=1)
            q0_pre = sm(q0_t) # t= 0, Y = 1 score t= Y =0
            q1_pre = sm(q1_t) # t= 1, Y = 1 score t= Y =0

            q0_pre = q0_pre[:,1]
            q1_pre = q1_pre[:,1]

        elif self.args.task=="vol":
            q0_pre = q0_t.view(-1)
            q1_pre = q1_t.view(-1)




        T = torch.cat([x['batch']['treatment_id'] for x in outputs])

        T0_ind = (T == 0).nonzero().squeeze()
        T1_ind = (T == 1).nonzero().squeeze()

        #  estimate the ATE on the simulation data
        Y_true = torch.cat([x['batch']['outcome_sim'] for x in outputs])
        #simulation ctf for calculating true ATE
        Y_true_ctf = torch.cat([x['batch']['outcome_ctf'] for x in outputs])
        tau = ITE(Y_true, Y_true_ctf, T)

        # calculate ate using predicted counterfactual and observational outcomes
        q0 = q0_pre.clone().scatter(0, T0_ind, 1)
        Y0_true = Y_true.clone().scatter(0, T1_ind, 1)
        Q0 = torch.mul(q0,Y0_true)

        q1 = q1_pre.clone().scatter(0, T1_ind, 1)
        Y1_true = Y_true.clone().scatter(0, T0_ind, 1)
        Q1 = torch.mul(q1,Y1_true)

        tau_hat = Q1 - Q0

        ATE = torch.mean(tau.type(torch.FloatTensor)).item()
        ATE_hat = torch.mean(tau_hat.type(torch.FloatTensor)).item()

        PEHE = torch.mean((tau - tau_hat).type(torch.FloatTensor) ** 2)

        sqrt_PEHE = torch.sqrt(PEHE)
        eps_ATE = abs(ATE - ATE_hat)

        print('\n')
        print('=' * 9)
        # print('pred_ate:%.6f' % ATE_pre)
        print('sqrt_pehe:%.6f' % sqrt_PEHE)
        print('eps_ate:%.6f' % eps_ATE)
        # print('true_ate:%.6f' % ATE_true)
        print('=' * 9)




    def configure_optimizers(self):
        self.train_len = self.args.train_len
        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            self.args.num_train_epochs = self.args.max_steps // self.train_len // self.args.accumulate_grad_batches + 1
        else:
            t_total = self.train_len // self.args.accumulate_grad_batches * self.args.num_train_epochs

        logger.info('{} training steps in total.. '.format(t_total))

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0},
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.lr, eps=self.args.adam_epsilon) # has a weight decay in this optimizer already,
        # scheduler is called only once per epoch by default
        warmup_steps = t_total * 0.1
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                    num_training_steps=t_total)
        scheduler_dict = {
            'scheduler': scheduler,
            'interval': 'step',
            'name': 'linear-schedule',
        }

        return [optimizer], [scheduler]









if __name__ == '__main__':

    print("Done")




