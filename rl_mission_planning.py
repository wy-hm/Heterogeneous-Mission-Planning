import torch
import torch.nn as nn
from torch.distributions import Categorical

from modules import Beta_Glimpse, GraphEmbedding, Beta_Pointer

from Environment.State import TA_State


class att_layer(nn.Module):
    def __init__(self, embed_dim, n_heads, feed_forward_hidden=512, bn=False):
        super(att_layer, self).__init__()
        self.mha = torch.nn.MultiheadAttention(embed_dim, n_heads)
        self.embed = nn.Sequential(nn.Linear(embed_dim, feed_forward_hidden), nn.ReLU(), nn.Linear(feed_forward_hidden, embed_dim))

    def forward(self, x):
        """
        [Input]
            x: batch X seq_len X embedding_size
        [Output]
            _2: batch X seq_len X embedding_size

            # Multiheadattention in pytorch starts with (target_seq_length, batch_size, embedding_size).
            # thus we permute order first. https://pytorch.org/docs/stable/nn.html#multiheadattention
        """
        x = x.permute(1, 0, 2)
        _1 = x + self.mha(x, x, x)[0]
        _1 = _1.permute(1, 0, 2)
        _2 = _1 + self.embed(_1)
        return _2


class AttentionModule(nn.Sequential):
    def __init__(self, embed_dim, n_heads, feed_forward_hidden=512, n_self_attentions=3, bn=False):
        super(AttentionModule, self).__init__(
            *(att_layer(embed_dim, n_heads, feed_forward_hidden, bn) for _ in range(n_self_attentions))
        )


class HetNet(nn.Module):
    def __init__(self,
                 embedding_size,
                 hidden_size,
                 n_head=8,
                 C=10):
        super(HetNet, self).__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        # self.seq_len = seq_len
        self.n_head = n_head
        self.C = C

        self.embedding = GraphEmbedding(8, embedding_size)
        self.mha = AttentionModule(embedding_size, n_head)

        self.init_w = nn.Parameter(torch.Tensor(2 * self.embedding_size))
        self.init_w.data.uniform_(-1, 1)
        self.glimpse = Beta_Glimpse(self.embedding_size+1, self.embedding_size, self.hidden_size, self.n_head)
        self.pointer = Beta_Pointer(self.embedding_size, self.hidden_size, 1, self.C)

        self.h_context_embed = nn.Linear(self.embedding_size, self.embedding_size)
        self.v_weight_embed = nn.Linear(self.embedding_size * 2, self.embedding_size)

    def forward(self, inputs):
        """
        [Input]
            inputs: batch_size x seq_len x feature_size
        [Return]
            logprobs:
            actions:
        """
        batch_size = inputs.shape[0]
        seq_len = inputs.shape[1]

        embedded = self.embedding(inputs)         # batch x task_num x embedding_size
        h = self.mha(embedded)                    # batch x task_num x embedding_size
        h_mean = h.mean(dim=1)                    # batch x embedding_size
        h_bar = self.h_context_embed(h_mean)      # batch x embedding_size
        h_rest = self.v_weight_embed(self.init_w) # batch x embedding_size
        query = h_bar + h_rest                    # batch x embedding_size

        #init query
        prev_chosen_indices = []
        prev_chosen_logprobs = []
        first_chosen_hs = None

        state = TA_State.initialize(inputs)

        # mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        # mask[:,0] = True # Except Depot at first time

        # depot = torch.zeros(batch_size,dtype=torch.int64 ,device=inputs.device)
        prev_chosen_indices.append(state.prev_task.squeeze(1))

        while not state.all_finished():
            mask = state.get_mask() # batch x task_num

            # Add remained time on decoding step
            remained_time = state.get_remained_time() # batch x task_num
            query = torch.cat((query,remained_time), dim=1)

            _, n_query = self.glimpse(query, h, mask)
            prob, _ = self.pointer(n_query, h, mask)
            cat = Categorical(prob)
            chosen = cat.sample() # batch
            logprobs = cat.log_prob(chosen)
            prev_chosen_indices.append(chosen)
            prev_chosen_logprobs.append(logprobs)

            # mask[[i for i in range(batch_size)], chosen] = True
            state = state.update(chosen)

            cc = chosen.unsqueeze(1).unsqueeze(2).repeat(1, 1, self.embedding_size)
            if first_chosen_hs is None:
                first_chosen_hs = h.gather(1, cc).squeeze(1)
            chosen_hs = h.gather(1, cc).squeeze(1)
            h_rest = self.v_weight_embed(torch.cat([first_chosen_hs, chosen_hs], dim=-1))
            query = h_bar + h_rest

        return torch.stack(prev_chosen_logprobs, 1), torch.stack(prev_chosen_indices, 1)


class HetNet_Partial(nn.Module):
    def __init__(self,
                 partial_type,
                 input_size,
                 embedding_size,
                 hidden_size,
                 n_head=8,
                 C=10):
        super(HetNet_Partial, self).__init__()

        self.partial_type = partial_type

        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        # self.seq_len = seq_len
        self.n_head = n_head
        self.C = C

        self.embedding = GraphEmbedding(input_size, embedding_size)
        self.mha = AttentionModule(embedding_size, n_head)

        self.init_w = nn.Parameter(torch.Tensor(2 * self.embedding_size))
        self.init_w.data.uniform_(-1, 1)
        self.glimpse = Beta_Glimpse(self.embedding_size+1, self.embedding_size, self.hidden_size, self.n_head)
        self.pointer = Beta_Pointer(self.embedding_size, self.hidden_size, 1, self.C)

        self.h_context_embed = nn.Linear(self.embedding_size, self.embedding_size)
        self.v_weight_embed = nn.Linear(self.embedding_size * 2, self.embedding_size)

    def forward(self, inputs):
        """
        [Input]
            inputs: batch_size x seq_len x feature_size
        [Return]
            logprobs:
            actions:
        """
        batch_size = inputs.shape[0]
        seq_len = inputs.shape[1]

        state = TA_State.initialize(inputs)

        if self.partial_type=="notype":
            embedded = self.embedding(inputs[:,:,:5])         # batch x task_num x embedding_size
        elif self.partial_type=="nogeo":
            inputs = torch.cat((inputs[:,:,:2],inputs[:,:,-3:]),axis=-1)
            embedded = self.embedding(inputs)
        elif self.partial_type=="nothing":
            embedded = self.embedding(inputs[:,:,:2])
        else:
            raise ValueError
        h = self.mha(embedded)                           # batch x task_num x embedding_size
        h_mean = h.mean(dim=1)                           # batch x embedding_size
        h_bar = self.h_context_embed(h_mean)             # batch x embedding_size
        h_rest = self.v_weight_embed(self.init_w)        # batch x embedding_size
        query = h_bar + h_rest                           # batch x embedding_size

        #init query
        prev_chosen_indices = []
        prev_chosen_logprobs = []
        first_chosen_hs = None

        # mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        # mask[:,0] = True # Except Depot at first time

        # depot = torch.zeros(batch_size,dtype=torch.int64 ,device=inputs.device)
        prev_chosen_indices.append(state.prev_task.squeeze(1))

        while not state.all_finished():
            mask = state.get_mask() # batch x task_num

            # Add remained time on decoding step
            remained_time = state.get_remained_time() # batch x task_num
            query = torch.cat((query,remained_time), dim=1)

            _, n_query = self.glimpse(query, h, mask)
            prob, _ = self.pointer(n_query, h, mask)
            cat = Categorical(prob)
            chosen = cat.sample() # batch
            logprobs = cat.log_prob(chosen)
            prev_chosen_indices.append(chosen)
            prev_chosen_logprobs.append(logprobs)

            # mask[[i for i in range(batch_size)], chosen] = True
            state = state.update(chosen)

            cc = chosen.unsqueeze(1).unsqueeze(2).repeat(1, 1, self.embedding_size)
            if first_chosen_hs is None:
                first_chosen_hs = h.gather(1, cc).squeeze(1)
            chosen_hs = h.gather(1, cc).squeeze(1)
            h_rest = self.v_weight_embed(torch.cat([first_chosen_hs, chosen_hs], dim=-1))
            query = h_bar + h_rest

        return torch.stack(prev_chosen_logprobs, 1), torch.stack(prev_chosen_indices, 1)

class HetNet_tSNE(nn.Module):
    def __init__(self,
                 embedding_size,
                 hidden_size,
                 n_head=8,
                 C=10):
        super(HetNet_tSNE, self).__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        # self.seq_len = seq_len
        self.n_head = n_head
        self.C = C

        self.embedding = GraphEmbedding(8, embedding_size)
        self.mha = AttentionModule(embedding_size, n_head)

        self.init_w = nn.Parameter(torch.Tensor(2 * self.embedding_size))
        self.init_w.data.uniform_(-1, 1)
        self.glimpse = Beta_Glimpse(self.embedding_size+1, self.embedding_size, self.hidden_size, self.n_head)
        self.pointer = Beta_Pointer(self.embedding_size, self.hidden_size, 1, self.C)

        self.h_context_embed = nn.Linear(self.embedding_size, self.embedding_size)
        self.v_weight_embed = nn.Linear(self.embedding_size * 2, self.embedding_size)

    def forward(self, inputs):
        """
        [Input]
            inputs: batch_size x seq_len x feature_size
        [Return]
            logprobs:
            actions:
        """
        batch_size = inputs.shape[0]
        seq_len = inputs.shape[1]

        embedded = self.embedding(inputs)         # batch x task_num x embedding_size
        h = self.mha(embedded)                    # batch x task_num x embedding_size
        h_mean = h.mean(dim=1)                    # batch x embedding_size
        h_bar = self.h_context_embed(h_mean)      # batch x embedding_size
        h_rest = self.v_weight_embed(self.init_w) # batch x embedding_size
        query = h_bar + h_rest                    # batch x embedding_size

        #init query
        prev_chosen_indices = []
        prev_chosen_logprobs = []
        first_chosen_hs = None

        state = TA_State.initialize(inputs)

        # mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        # mask[:,0] = True # Except Depot at first time

        # depot = torch.zeros(batch_size,dtype=torch.int64 ,device=inputs.device)
        prev_chosen_indices.append(state.prev_task.squeeze(1))

        while not state.all_finished():
            mask = state.get_mask() # batch x task_num

            # Add remained time on decoding step
            remained_time = state.get_remained_time() # batch x task_num
            query = torch.cat((query,remained_time), dim=1)

            _, n_query = self.glimpse(query, h, mask)
            prob, _ = self.pointer(n_query, h, mask)
            cat = Categorical(prob)
            chosen = cat.sample() # batch
            logprobs = cat.log_prob(chosen)
            prev_chosen_indices.append(chosen)
            prev_chosen_logprobs.append(logprobs)

            # mask[[i for i in range(batch_size)], chosen] = True
            state = state.update(chosen)

            cc = chosen.unsqueeze(1).unsqueeze(2).repeat(1, 1, self.embedding_size)
            if first_chosen_hs is None:
                first_chosen_hs = h.gather(1, cc).squeeze(1)
            chosen_hs = h.gather(1, cc).squeeze(1)
            h_rest = self.v_weight_embed(torch.cat([first_chosen_hs, chosen_hs], dim=-1))
            query = h_bar + h_rest

        return torch.stack(prev_chosen_logprobs, 1), torch.stack(prev_chosen_indices, 1), embedded.reshape(-1, self.embedding_size)

class HetNet_Partial_tSNE(nn.Module):
    def __init__(self,
                 partial_type,
                 input_size,
                 embedding_size,
                 hidden_size,
                 n_head=8,
                 C=10):
        super(HetNet_Partial_tSNE, self).__init__()

        self.partial_type = partial_type

        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        # self.seq_len = seq_len
        self.n_head = n_head
        self.C = C

        self.embedding = GraphEmbedding(input_size, embedding_size)
        self.mha = AttentionModule(embedding_size, n_head)

        self.init_w = nn.Parameter(torch.Tensor(2 * self.embedding_size))
        self.init_w.data.uniform_(-1, 1)
        self.glimpse = Beta_Glimpse(self.embedding_size+1, self.embedding_size, self.hidden_size, self.n_head)
        self.pointer = Beta_Pointer(self.embedding_size, self.hidden_size, 1, self.C)

        self.h_context_embed = nn.Linear(self.embedding_size, self.embedding_size)
        self.v_weight_embed = nn.Linear(self.embedding_size * 2, self.embedding_size)

    def forward(self, inputs):
        """
        [Input]
            inputs: batch_size x seq_len x feature_size
        [Return]
            logprobs:
            actions:
        """
        batch_size = inputs.shape[0]
        seq_len = inputs.shape[1]

        state = TA_State.initialize(inputs)

        if self.partial_type=="notype":
            embedded = self.embedding(inputs[:,:,:5])         # batch x task_num x embedding_size
        elif self.partial_type=="nogeo":
            inputs = torch.cat((inputs[:,:,:2],inputs[:,:,-3:]),axis=-1)
            embedded = self.embedding(inputs)
        elif self.partial_type=="nothing":
            embedded = self.embedding(inputs[:,:,:2])
        else:
            raise ValueError
        h = self.mha(embedded)                           # batch x task_num x embedding_size
        h_mean = h.mean(dim=1)                           # batch x embedding_size
        h_bar = self.h_context_embed(h_mean)             # batch x embedding_size
        h_rest = self.v_weight_embed(self.init_w)        # batch x embedding_size
        query = h_bar + h_rest                           # batch x embedding_size

        #init query
        prev_chosen_indices = []
        prev_chosen_logprobs = []
        first_chosen_hs = None

        # mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        # mask[:,0] = True # Except Depot at first time

        # depot = torch.zeros(batch_size,dtype=torch.int64 ,device=inputs.device)
        prev_chosen_indices.append(state.prev_task.squeeze(1))

        while not state.all_finished():
            mask = state.get_mask() # batch x task_num

            # Add remained time on decoding step
            remained_time = state.get_remained_time() # batch x task_num
            query = torch.cat((query,remained_time), dim=1)

            _, n_query = self.glimpse(query, h, mask)
            prob, _ = self.pointer(n_query, h, mask)
            cat = Categorical(prob)
            chosen = cat.sample() # batch
            logprobs = cat.log_prob(chosen)
            prev_chosen_indices.append(chosen)
            prev_chosen_logprobs.append(logprobs)

            # mask[[i for i in range(batch_size)], chosen] = True
            state = state.update(chosen)

            cc = chosen.unsqueeze(1).unsqueeze(2).repeat(1, 1, self.embedding_size)
            if first_chosen_hs is None:
                first_chosen_hs = h.gather(1, cc).squeeze(1)
            chosen_hs = h.gather(1, cc).squeeze(1)
            h_rest = self.v_weight_embed(torch.cat([first_chosen_hs, chosen_hs], dim=-1))
            query = h_bar + h_rest

        return torch.stack(prev_chosen_logprobs, 1), torch.stack(prev_chosen_indices, 1), embedded.reshape(-1, self.embedding_size)
