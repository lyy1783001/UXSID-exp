import torch
from torch import nn
from collections import OrderedDict
import logging
import sys
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbeddingDict, MLP_Block
from fuxictr.pytorch.torch_utils import get_optimizer
from tqdm import tqdm
from .module import DIN_Attention
from .loss import DinLoss


class Din(BaseModel):
    def __init__(
        self,
        feature_map,
        model_id="Din",
        gpu=-1,
        task="binary_classification",
        learning_rate=1e-3,
        embedding_dim=16,
        embedding_regularizer=None,
        net_regularizer=None,
        target_field=None,
        sequence_field=None,
        item_field="item_id",
        cate_field="cate_id",
        item_sequence_field="item_history",
        cate_sequence_field="cate_history",
        short_seq_len=100,
        attention_hidden_units=[32, 32],
        attention_hidden_activations="Dice",
        attention_output_activation=None,
        attention_dropout=0.0,
        din_use_softmax=True,
        dnn_hidden_units=[200, 80],
        dnn_activations="ReLU",
        net_dropout=0.0,
        batch_norm=False,
        logging_steps=100,
        **kwargs,
    ):
        super().__init__(
            feature_map,
            model_id=model_id,
            gpu=gpu,
            task=task,
            embedding_regularizer=embedding_regularizer,
            net_regularizer=net_regularizer,
            **kwargs,
        )
        self._logging_steps = logging_steps
        self.short_seq_len = short_seq_len
        self.embedding_dim = embedding_dim
        self.embedding_layer = FeatureEmbeddingDict(feature_map, embedding_dim)

        if target_field is None:
            target_field = kwargs.get("short_target_field", None)
        if sequence_field is None:
            sequence_field = kwargs.get("short_sequence_field", None)
        if target_field is None:
            target_field = [item_field, cate_field]
        if sequence_field is None:
            sequence_field = [item_sequence_field, cate_sequence_field]

        if type(target_field) != list:
            target_field = [target_field]
        if type(sequence_field) != list:
            sequence_field = [sequence_field]

        self.target_field = target_field
        self.sequence_field = sequence_field

        def get_field_dim(field_name):
            spec = feature_map.features[field_name]
            return spec.get("emb_output_dim", spec.get("embedding_dim", embedding_dim))

        self.non_seq_features = [
            f
            for f, spec in feature_map.features.items()
            if spec["type"] not in ["sequence", "meta"]
        ]
        non_seq_dim = sum(get_field_dim(f) for f in self.non_seq_features)

        target_dim = sum(get_field_dim(f) for f in self.target_field)
        seq_dim = sum(get_field_dim(f) for f in self.sequence_field)
        if target_dim != seq_dim:
            raise ValueError(
                f"Din requires target_dim == seq_dim, but got target_dim={target_dim}, seq_dim={seq_dim}."
            )
        self.din_embedding_dim = seq_dim

        self.din_attention = DIN_Attention(
            embedding_dim=self.din_embedding_dim,
            attention_units=attention_hidden_units,
            hidden_activations=attention_hidden_activations,
            output_activation=attention_output_activation,
            dropout_rate=attention_dropout,
            batch_norm=batch_norm,
            use_softmax=din_use_softmax,
        )

        self.dnn = MLP_Block(
            input_dim=non_seq_dim + self.din_embedding_dim,
            output_dim=1,
            hidden_units=dnn_hidden_units,
            hidden_activations=dnn_activations,
            output_activation=self.output_activation,
            dropout_rates=net_dropout,
            batch_norm=batch_norm,
        )

        optimizer = kwargs.get("optimizer", "adam")
        loss = kwargs.get("loss", "binary_crossentropy")
        self.compile(optimizer, loss, learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def compile(self, optimizer, loss, lr):
        self.optimizer = get_optimizer(optimizer, self.parameters(), lr)
        self.loss_fn = DinLoss(loss)

    def compute_loss(self, return_dict, y_true):
        loss = self.loss_fn(return_dict["y_pred"], y_true, reduction="mean")
        loss += self.regularization_loss()
        return loss

    def train_epoch(self, data_generator):
        self._batch_index = 0
        train_loss = 0
        logging_loss = 0
        self.train()
        if self._verbose == 0:
            batch_iterator = data_generator
        else:
            batch_iterator = tqdm(data_generator, disable=False, file=sys.stdout)
        for batch_index, batch_data in enumerate(batch_iterator):
            self._batch_index = batch_index
            self._total_steps += 1
            loss = self.train_step(batch_data)
            train_loss += loss.item()
            logging_loss += loss.item()
            if self._logging_steps and self._total_steps % self._logging_steps == 0:
                logging.info(
                    "Train loss: {:.6f} (Steps: {:d})".format(
                        logging_loss / self._logging_steps, self._total_steps
                    )
                )
                logging_loss = 0
            if self._total_steps % self._eval_steps == 0:
                logging.info("Train loss: {:.6f}".format(train_loss / self._eval_steps))
                train_loss = 0
                self.eval_step()
            if self._stop_training:
                break

    def forward(self, inputs):
        X = self.get_inputs(inputs)
        feature_emb_dict = self.embedding_layer(X)

        target_emb = torch.cat([feature_emb_dict[f] for f in self.target_field], dim=-1)
        history_emb = torch.cat([feature_emb_dict[f] for f in self.sequence_field], dim=-1)
        seq_ids = X[self.sequence_field[0]].long()
        if self.short_seq_len and self.short_seq_len > 0 and seq_ids.dim() == 2:
            seq_ids = seq_ids[:, -self.short_seq_len :]
            history_emb = history_emb[:, -self.short_seq_len :, :]
        mask = seq_ids != 0
        din_emb = self.din_attention(target_emb, history_emb, mask)

        non_seq_emb_dict = OrderedDict()
        for f in self.non_seq_features:
            if f in feature_emb_dict:
                non_seq_emb_dict[f] = feature_emb_dict[f]
        non_seq_emb = self.embedding_layer.dict2tensor(non_seq_emb_dict, flatten_emb=True)

        dnn_input = torch.cat([non_seq_emb, din_emb], dim=-1)
        y_pred = self.dnn(dnn_input)
        return {"y_pred": y_pred}
