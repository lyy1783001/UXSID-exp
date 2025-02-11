# =========================================================================
# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================
import torch
from torch import nn
from torch import Tensor
from collections import OrderedDict, defaultdict
from typing import Optional, Any, Union, Callable
import numpy as np
import json
import os
import sys
import logging
from tqdm import tqdm
from pandas.core.common import flatten
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbeddingDict, MLP_Block, DIN_Attention
from fuxictr.pytorch.torch_utils import get_optimizer
from .loss import MixLoss
from .module import *
import polars as pl


class CFormer(BaseModel):
    def __init__(
            self, 
            feature_map, 
            model_id="Cluster", 
            gpu=-1, 
            learning_rate=1e-3, 
            embedding_dim=10, 
            embedding_regularizer=None, 
            net_regularizer=None,
            user_field="user_id",
            item_field="item_id",
            item_sequence_field="click_history",
            long_target_field=["item_id", "cate_id"],
            long_sequence_field=["click_history", "cate_history"],
            # short
            short_target_field=["item_id", "cate_id"],
            short_sequence_field=["click_history", "cate_history"],
            logging_steps=500,
            n_cluster: int = 12,
            d_model: int = 256, 
            nhead: int = 4, 
            num_encoder_layers: int = 4,
            num_decoder_layers: int = 2, 
            dim_feedforward: int = 256, 
            dropout: float = 0.1,
            #
            attention_dropout: float = 0.1,
            dec_dropout: float = 0.1,
            activation: str = 'relu',
            layer_norm_eps: float = 1e-5, 
            batch_first: bool = True, 
            norm_first: bool = False,
            encoder_norm: bool = False,
            decoder_norm: bool = False,
            recon_loss: bool = True, 
            orth_loss: bool = False,
            bce_loss: bool = False,
            recon_weight: float = 1.0,
            orth_weight: float = 1.0,
            bce_weight: float = 1.0,
            # dnn
            dnn_hidden_units=[512, 128, 64],
            dnn_activations="ReLU",
            net_dropout=0,
            batch_norm=False,
            #
            attention_hidden_units=[64],
            attention_hidden_activations="Dice",
            attention_output_activation=None,
            din_use_softmax=False,
            **kwargs):
        self._supervised = (bce_weight > 0) and bce_loss
        if self._supervised:
            kwargs['monitor'] = 'AUC'
            kwargs['monitor_mode'] = 'max'
        else:
            kwargs['monitor'] = 'eval_loss'
            kwargs['monitor_mode'] = 'min'

        super(CFormer, self).__init__(feature_map,
                                        model_id=model_id, 
                                        gpu=gpu, 
                                        embedding_regularizer=embedding_regularizer, 
                                        net_regularizer=net_regularizer,
                                        **kwargs)
        self._logging_steps = logging_steps


        if type(short_target_field) != list:
            short_target_field = [short_target_field]
        if type(short_sequence_field) != list:
            short_sequence_field = [short_sequence_field]
        if type(long_target_field) != list:
            long_target_field = [long_target_field]
        if type(long_sequence_field) != list:
            long_sequence_field = [long_sequence_field]
        self.user_field = user_field
        self.item_field = item_field
        self.item_sequence_field = item_sequence_field
        self.short_target_field = short_target_field
        self.short_sequence_field = short_sequence_field
        self.long_target_field = long_target_field
        self.long_sequence_field = long_sequence_field
        assert len(self.long_target_field) == len(self.long_sequence_field), \
            "Config error: target_field mismatches with sequence_field."

        self.feature_map = feature_map
        self.embedding_dim = embedding_dim

        # save path
        basename = f"dropout{dropout}_" + \
                    f"cluster{n_cluster}_" + \
                    f"lr{learning_rate}_" + \
                    f"reg{embedding_regularizer}_" + \
                    f"dmodel{d_model}_" + \
                    f"nhead{nhead}_" + \
                    f"enc{num_encoder_layers}_" + \
                    f"dec{num_decoder_layers}_" + \
                    f"ffn{dim_feedforward}_"
        if recon_loss and recon_weight > 0:
            basename += f"{recon_weight}recon_loss_"
        if orth_loss and orth_weight > 0:
            basename += f"{orth_weight}orth_loss_"
        if bce_loss and bce_weight > 0:
            basename += f"{bce_weight}bce_loss_"

        self.embedding_layer = FeatureEmbeddingDict(feature_map, embedding_dim)
        self.linear = nn.Linear(len(long_sequence_field) * embedding_dim, d_model)

        self.global_vecs = nn.Parameter(torch.empty((n_cluster, d_model)))
        # encoder
        self.pos = nn.Embedding(900 + 1, d_model)
        encoder_layer = AbridgedTransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout,
                                                        activation, layer_norm_eps, batch_first, norm_first)
        encoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps) if encoder_norm else None
        self.encoder = Encoder(encoder_layer, num_encoder_layers, encoder_norm)

        # decoder
        if num_decoder_layers > 0:
            decoder_layer = AbridgedTransformerDecoderLayer(d_model, nhead, dim_feedforward, dec_dropout,
                                                            activation, layer_norm_eps, batch_first, norm_first)
            decoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps) if decoder_norm else None
            self.decoder = Encoder(decoder_layer, num_decoder_layers, decoder_norm)
        else:
            self.decoder = NoParamLayer()

        if self._supervised:
            self.short_attention = DIN_Attention(embedding_dim * len(self.short_target_field),
                                                attention_units=attention_hidden_units,
                                                hidden_activations=attention_hidden_activations,
                                                output_activation=attention_output_activation,
                                                dropout_rate=attention_dropout,
                                                use_softmax=din_use_softmax)
            self.long_attention = DIN_Attention(embedding_dim * len(self.long_target_field),
                                                attention_units=attention_hidden_units,
                                                hidden_activations=attention_hidden_activations,
                                                output_activation=attention_output_activation,
                                                dropout_rate=attention_dropout,
                                                use_softmax=din_use_softmax)

            self.dnn = MLP_Block(input_dim=feature_map.sum_emb_out_dim() + len(long_sequence_field) * embedding_dim,
                             output_dim=1,
                             hidden_units=dnn_hidden_units,
                             hidden_activations=dnn_activations,
                             output_activation=self.output_activation,
                             dropout_rates=net_dropout,
                             batch_norm=batch_norm)
        
        # loss
        loss_kwargs = {
            'recon_loss': recon_loss,
            'orth_loss': orth_loss,
            'bce_loss': bce_loss,

            'recon_weight': recon_weight,
            'orth_weight': orth_weight,
            'bce_weight': bce_weight,
        }
        self.compile(kwargs["optimizer"], learning_rate, **loss_kwargs)
        self.reset_parameters()
        self.model_to_device()

    def compile(self, optimizer, lr, **kwargs):
        self.optimizer = get_optimizer(optimizer, self.parameters(), lr)
        self.loss_fn = MixLoss(**kwargs)

    def reset_parameters(self):
        super().reset_parameters()
        nn.init.xavier_normal_(self.global_vecs.data)

        if self.init_model is not None:
            self.load_weights(self.init_model)
            for param in self.embedding_layer.parameters():
                param.requires_grad = False

    def compute_loss(self, return_dict, y_true=None):
        loss, loss_dict = self.loss_fn(
                            personalized_vecs=return_dict["personalized_vecs"], 
                            ori_behaviors=return_dict["ori_behaviors"], 
                            recon_behaviors=return_dict["recon_behaviors"], 
                            y_pred=return_dict.get("y_pred", None),
                            y_true=y_true, 
                            enc_weights=return_dict.get("enc_weights", None),
                            dec_weights=return_dict.get("dec_weights", None),
                            mask=return_dict.get("mask", None),
                            reduction='mean')
        reg_loss = self.regularization_loss()
        loss += reg_loss
        loss_dict['reg_loss'] = reg_loss
        return loss, loss_dict
    

    def get_embedding(self, field, feature_emb_dict):
        if type(field) == tuple:
            emb_list = [feature_emb_dict[f] for f in field]
            return torch.cat(emb_list, dim=-1)
        else:
            return feature_emb_dict[field]

    def forward(self, inputs, construct_indices=False, assign=False):
        X = self.get_inputs(inputs)
        feature_emb_dict = self.embedding_layer(X)
        sequence_embs = self.get_embedding(tuple(self.long_sequence_field), feature_emb_dict)   # (B, L, 2*D)
        sequence_embs = self.linear(sequence_embs)  

        # pos
        pos_mask = sequence_embs.shape[1] - torch.arange(sequence_embs.shape[1], 
                                                        device=sequence_embs.device)
        pos = self.pos(pos_mask).unsqueeze(0).tile(sequence_embs.shape[0], 1, 1)
        sequence_embs += pos * 0.02

        seq_field = list(flatten(self.long_sequence_field))[0]                                            # (B, L, d_model)
        mask = X[seq_field].long() == 0                                                                   # true indicates padding
        personalized_vecs, enc_weights = self.encoder(
                                            query=self.global_vecs.unsqueeze(0).tile(sequence_embs.shape[0], 1, 1),     # (B, n_cluster, d_model)
                                            key_value=sequence_embs,                                                    # (B, L, d_model)
                                            key_padding_mask=mask)
        output, dec_weights = self.decoder(query=sequence_embs, key_value=personalized_vecs)



        centroids = self.get_centroids_embs(
                        X[self.user_field],
                        X[self.item_sequence_field], 
                        dec_weights.transpose(-1, -2),
                        enc_weights,
                        sequence_embs
                        )
            
        if self._supervised:
            # short interest attention
            seq_field = list(flatten([self.long_sequence_field]))[0]  # flatten nested list to pick the first field
            target_emb = self.get_embedding(tuple(self.short_target_field), feature_emb_dict)
            short_sequence_emb = self.get_embedding(tuple(self.short_sequence_field), feature_emb_dict)
            short_sequence_emb = short_sequence_emb[:, -100:, :]
            short_mask = X[seq_field][:, -100:].long() != 0
            short_interest_emb = self.short_attention(target_emb, short_sequence_emb, short_mask)
            for field, field_emb in zip(list(flatten(self.short_sequence_field)),
                                                    short_interest_emb.split(self.embedding_dim, dim=-1)):
                feature_emb_dict['short_'+field] = field_emb



            # long interest attention
            target_emb = self.get_embedding(tuple(self.long_target_field), feature_emb_dict)
            interest_embs = self.long_attention(target_emb, centroids)

            for idx, (sequence_field, interest) in enumerate(zip(self.long_sequence_field,
                                                                interest_embs.split(self.embedding_dim, dim=-1))):
                feature_emb_dict[sequence_field] = interest

            feature_emb = self.embedding_layer.dict2tensor(feature_emb_dict, flatten_emb=True)
            short_term_feature_emb = torch.cat([feature_emb_dict['short_'+field] 
                                            for field in list(flatten([self.short_sequence_field]))], dim=-1)
            feature_emb = torch.cat([feature_emb, short_term_feature_emb], dim=-1)
            y_pred = self.dnn(feature_emb)
        else:
            y_pred = None

        return {
            "personalized_vecs": personalized_vecs,
            "ori_behaviors": sequence_embs,
            "recon_behaviors": output,
            "y_pred": y_pred,
            "enc_weights": enc_weights,
            "dec_weights": dec_weights,
            "mask": mask,
        }

    def predict(self, data_generator, construct_indices=True, assign=True):
        self.eval()  # set to evaluation mode
        with torch.no_grad():
            self.cluster_indices = []
            self.cluster_items = []
            self.cluster_assign = {
                'user_id': [],
                'item_history': [],
                'cate_history': [],
                'item_history_cluster': [],
                'seq_len': [],
                'cate_uni': []
            }
            if self._verbose > 0:
                data_generator = tqdm(data_generator, disable=False, file=sys.stdout)
            for batch_data in data_generator:
                self.forward(batch_data, construct_indices=construct_indices, assign=assign)
            
            if construct_indices and hasattr(self, 'cluster_indices'):
                self.save_cluster_indices()

            if assign and hasattr(self, 'cluster_assign'):
                self.save_cluster_assigns()


    def get_centroids_embs(self, user_id, item_sequence_ids, dec_weights, enc_weights, sequence_embs, k=5):      
        '''
        weight: (bs, n_cluster, S)
        sequence_embs: (bs, S, D)
        '''        
        cluster_idx = torch.sort(dec_weights, dim=1, descending=True).indices[:, :k, :]     # (bs, k, S)
        tmp = torch.zeros_like(dec_weights)                                                 # (bs, n_cluster, S)
        tmp.scatter_(
            1, 
            cluster_idx,                                                                    # (bs, 1, S)
            torch.ones_like(cluster_idx).float()
        )                                                                               # (bs, n_cluster, S)
        cluster_weights = enc_weights * tmp                                             
        centroids = torch.bmm(cluster_weights, sequence_embs)                           # (bs, n_cluster, D)

        return centroids
    

    # ---------------- Modify Backbone -----------
         
    def fit(self, data_generator, epochs=1, validation_data=None,
            max_gradient_norm=10., **kwargs):
        self._train_loss = 0
        self._train_loss_dict = defaultdict(int)
        super().fit(data_generator, epochs, validation_data,
                    max_gradient_norm, **kwargs)
        self.save_weights(self.checkpoint)


    def train_epoch(self, data_generator):
        self._batch_index = 0
        
        self.train()
        if self._verbose == 0:
            batch_iterator = data_generator
        else:
            batch_iterator = tqdm(data_generator, disable=False, file=sys.stdout)
        for batch_index, batch_data in enumerate(batch_iterator):
            self._batch_index = batch_index
            self._total_steps += 1
            loss, loss_dict = self.train_step(batch_data)
            self._train_loss += loss.item()
            for k, v in loss_dict.items():
                self._train_loss_dict[k] += v
            if self._total_steps % self._logging_steps == 0:
                logging.info("Train loss: {:.6f} (Steps: {:d})".format(
                                                                    self._train_loss / self._logging_steps, 
                                                                    self._total_steps))
                self._train_loss = 0
                for k, v in self._train_loss_dict.items():
                    logging.info("{}: {:.6f} (Steps: {:d})".format(k,
                                                                    v / self._logging_steps, 
                                                                    self._total_steps))
                    self._train_loss_dict[k] = 0
            if self._total_steps % self._eval_steps == 0:
                self.eval_step()
            if self._stop_training:
                break

    def train_step(self, batch_data):
        self.optimizer.zero_grad()
        return_dict = self.forward(batch_data)
        y_true = self.get_labels(batch_data)
        loss, loss_dict = self.compute_loss(return_dict, y_true)
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), self._max_gradient_norm)
        self.optimizer.step()
        return loss, loss_dict
    

    def load_weights(self, checkpoint):
        if os.path.exists(checkpoint):
            super().load_weights(checkpoint)  


    def evaluate(self, data_generator, metrics=None):
        if self._supervised:
            return super().evaluate(data_generator, metrics)
        else:
            self.eval()  # set to evaluation mode
            with torch.no_grad():
                if self._verbose > 0:
                    batch_iterator = tqdm(data_generator, disable=False, file=sys.stdout)

                eval_loss = 0
                for batch_data in batch_iterator:
                    return_dict = self.forward(batch_data)
                    y_true = self.get_labels(batch_data)
                    loss, loss_dict = self.compute_loss(return_dict, y_true)
                    eval_loss += loss
                val_logs = {
                    "eval_loss": eval_loss
                }
                logging.info('[Metrics] ' + ' - '.join('{}: {:.6f}'.format(k, v) for k, v in val_logs.items()))
                return val_logs 
            

    def eval_step(self):
        if self.valid_gen is not None:
            super().eval_step()