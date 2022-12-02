import typing

import torch
from torch import nn
from torch.nn import functional as F

from notes_generator.constants import *
from notes_generator.layers.base_layers import BiLSTM, get_conv_stack
from notes_generator.models.fuzzy_label import fuzzy_on_batch
from notes_generator.models.merge_labels import merge_labels
from notes_generator.models.util import batch_first


class OnsetsBase(nn.Module):
    """The base class for onset prediction model"""

    def predict(self, batch: typing.Dict[str, torch.Tensor]):
        """Predict an onset score.

        Parameters
        ----------
        batch : typing.Dict[str, torch.Tensor]
            The Dict containing tensors below:
            * audio
            * onset
            * conditions
            * beats

        Returns
        -------
        pred : torch.Tensor
            The tensor of shape (batch_size, seq_len, output_features)
            containing predicted onset score.
            `output_features` defaults to `1`.

        """
        pred, _ = self.predict_with_probs(batch)
        return pred

    def predict_with_probs(self, batch: typing.Dict[str, torch.Tensor]):
        """Predict an onset score with a probability

        Parameters
        ----------
        batch : typing.Dict[str, torch.Tensor]
            The Dict containing tensors below:
            * audio
            * onset
            * conditions
            * beats

        Returns
        -------
        pred : torch.Tensor
            The tensor of shape (batch_size, seq_len, output_features)
            containing predicted onset score.
        proba : torch.Tensor
            The tensor of shape (batch_size, seq_len, output_features)
            containing predicted probability of onset score on each frames.

        """
        device = next(self.parameters()).device
        mel = batch_first(batch["audio"]).to(device)
        condition = batch["condition"].expand(
            (
                mel.shape[0],
                mel.shape[1],
            )
        )
        condition = condition.reshape(-1, condition.shape[-1], 1).to(device)
        if self.enable_beats:
            beats = batch["beats"].reshape(mel.shape[0], mel.shape[1], -1).to(device)
        else:
            beats = None
        self.eval()
        with torch.no_grad():
            probs = self(mel, condition, beats)
            return probs > 0.5, probs

    def run_on_batch(
        self,
        batch: typing.Dict[str, torch.Tensor],
        fuzzy_width: int = 1,
        fuzzy_scale: float = 1.0,
        merge_scale: typing.Optional[float] = None,
        net: typing.Optional[nn.Module] = None,
    ):
        """Forward training on one minibatch

        Parameters
        ----------
        batch : typing.Dict[str, torch.Tensor]
            The Dict containing minibatch tensors below:
            * audio
            * onset
            * conditions
            * beats
        fuzzy_width : int
            The width of fuzzy labeling applied to notes_label.
            default: `1`
        fuzzy_scale : float
            The scale of fuzzy labeling applied to notes_label.
            The value should be within an interval `[0, 1]`.
            default: `1.0`
        merge_scale : typing.Optional[float] = None,
            If nonzero, mix the label of other conditions in specified scale.
            Formally, at each time step use the label calculated as below:
                max(onset_label, merge_scale * onset_label_in_other_conditions)
            default: `None`
        net : typing.Optional[nn.Module] = None,
            If not `None`, use specified model for forward propagation.
            default: `None`

        Returns
        -------
        g_loss : typing.Dict
            The Dict containing losses evaluated for current iteration.

        """
        device = next(self.parameters()).device
        audio_label = batch["audio"].to(device)
        onset_label = batch["onset"].to(device)
        # reshape batch first
        audio_label = batch_first(audio_label)
        if self.enable_condition:
            condition = batch["condition"].expand(
                (
                    audio_label.shape[0],
                    audio_label.shape[1],
                )
            )
            condition = condition.reshape(-1, condition.shape[-1], 1).to(device)
        else:
            condition = None
        if self.enable_beats:
            beats = (
                batch["beats"].reshape(audio_label.shape[0], audio_label.shape[1], -1).to(device)
            )
        else:
            beats = None
        if net is None:
            net = self
        if self.onset_weight:
            weight_onset = torch.tensor([self.onset_weight]).float().to(audio_label.device)
        else:
            weight_onset = None
        if fuzzy_width > 1 and self.training:
            onset_label = fuzzy_on_batch(onset_label, fuzzy_width, fuzzy_scale)
        if merge_scale and self.training:
            onset_label = merge_labels(onset_label, batch, merge_scale)
        onset_pred = net(audio_label, condition, beats)
        predictions = {
            "onset": onset_pred.reshape(*onset_label.shape),
        }
        losses = {
            "loss-onset": F.binary_cross_entropy(
                predictions["onset"], onset_label, weight=weight_onset
            ),
        }
        return predictions, losses


class SimpleOnsets(OnsetsBase):
    """Model for onset prediction

    Parameters
    ----------
    input_features : int
        Size of each input sample
    output_features : int
        Size of each output sample. In principle, set the value to `1`.
    inference_chunk_length : int
        Size of the chunk length used for inference, normally is sequence_length/FRAME
    model_complexity : int
        Number of channels defining convolution stack. default: `48`
    num_layers : int
        Number of recurrent layers. default: `2`
    enable_condition : bool
        If `True`, the game difficulty level will be provided to a model.
        default: `False`
    enable_beats : bool
        If `True`, beats information will be provided to a model.
        default: `False`
    dropout : float
        The rate of a Dropout layer before the linear layer.
        default: `0.5`
    onset_weight: typing.Optional[int]
        The scale factor multiplied to the loss calculated in training.
        default: `None`
    conv_stack_type: ConvStackType
        The type of ConvStack.
        default: `ConvStackType.v1`
    dropout_rnn: float
        The rate of Dropout layers of the RNN layer.
        default: 0
    """

    def __init__(
        self,
        input_features,
        output_features,
        inference_chunk_length: int = 640,
        model_complexity: int = 48,
        num_layers: int = 1,
        enable_condition: bool = False,
        enable_beats: bool = False,
        dropout: float = 0.5,
        onset_weight: typing.Optional[int] = None,
        conv_stack_type: ConvStackType = ConvStackType.v1,
        rnn_dropout: float = 0.0,
    ):
        super().__init__()
        model_size = model_complexity * 16
        self.enable_condition = enable_condition
        self.enable_beats = enable_beats
        self.onset_weight = onset_weight
        condition_length = 0
        beats_length = 0
        if self.enable_condition:
            condition_length = 1
        if self.enable_beats:
            beats_length = 1
        self.conv_stack_type = conv_stack_type
        self.onset_stack = get_conv_stack(conv_stack_type, input_features, model_size)
        self.onset_sequence = BiLSTM(
            model_size + condition_length + beats_length,
            model_size // 2,
            inference_chunk_length=inference_chunk_length,
            num_layers=num_layers,
            dropout=rnn_dropout,
        )
        self.drop = nn.Dropout(dropout)
        self.onset_linear = nn.Sequential(nn.Linear(model_size, output_features), nn.Sigmoid())

    def forward(self, mel, condition=None, beats=None):
        """

        Parameters
        ----------
        mel : torch.Tensor
            Tensor of shape (batch_size, seq_len, input_features)
            containing the log-scaled melspectrogram audio data.
        condition : torch.Tensor
            Tensor of shape (batch_size, seq_len, 1)
            containing the game difficulty level.
        beats : torch.Tensor
            Tensor of shape (batch_size, seq_len, 1)
            containing the beats information.

        Returns
        -------
        output: torch.Tensor
            Tensor of shape (batch_size, 1, output_features)

        """
        onset_pred = self.onset_stack(mel)
        if self.enable_condition:
            onset_pred = torch.cat([onset_pred, condition], dim=-1)
        if self.enable_beats:
            onset_pred = torch.cat([onset_pred, beats], dim=-1)
        onset_pred = self.onset_sequence(onset_pred)
        onset_pred = self.drop(onset_pred)
        onset_pred = self.onset_linear(onset_pred)
        return onset_pred
