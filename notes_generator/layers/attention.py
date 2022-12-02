import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1):
        super(Attention, self).__init__()
        self.d_model = d_model
        projection_inout = (self.d_model, self.d_model)
        self.query_projection = nn.Linear(*projection_inout)
        self.key_projection = nn.Linear(*projection_inout)
        self.value_projection = nn.Linear(*projection_inout)
        self.attention_projection = nn.Linear(*projection_inout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value):
        """
        Parameters
        ----------
        query (batch, q_length, model_size)
        key (batch, key_length, model_size)
        value (batch, key_length, model_size)

        Returns
        -------
        (batch, q_length, model_size)

        """
        # query: (batch, q_length, dim)
        # key: (batch, key_length, dim)
        # value: (batch, key_length, dim)
        queries = self.query_projection(query)
        keys = self.query_projection(key)
        values = self.query_projection(value)

        score = torch.matmul(queries, keys.transpose(-2, -1))
        # score: (batch, q_length, key_length)
        weights = self.dropout(F.softmax(score, dim=-1))
        output = torch.matmul(weights, values)
        # output: (batch, key_length, dim)
        output = self.attention_projection(output)
        return output, weights
