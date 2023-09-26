from numpy import sqrt as np_sqrt, tril as np_tril, triu as np_triu, ones as np_ones
from torch import (
    sum as torch_sum,
    max as torch_max,
    ones as torch_ones,
    ones_like as torch_ones_like,
    zeros as torch_zeros,
    zeros_like as torch_zeros_like,
    cat as torch_cat,
    sqrt as torch_sqrt,
    bmm as torch_bmm,
    matmul as torch_matmul,
    sigmoid as torch_sigmoid,
    tahn as torch_tanh,
    where as torch_where,
    from_numpy as torch_from_numpy,
)
from torch.nn import Module, ModuleList, Dropout, Linear, LayerNorm, Parameter, ParameterList, Sequential, ReLU, LSTM
from torch.nn.functional import relu as F_relu, dropout as F_dropout, softmax as F_softmax
from torch.nn.init import xavier_uniform_
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
from transformers.modeling_bert import gelu
from csr_mhqa.utils import get_weights, get_act


def tok_to_ent(tok2ent):
    if tok2ent == 'mean':
        return MeanPooling
    elif tok2ent == 'mean_max':
        return MeanMaxPooling
    else:
        raise NotImplementedError

class MLP(Module):
    def __init__(self, input_sizes, dropout_prob=0.2, bias=False):
        super(MLP, self).__init__()
        layers = ModuleList()
        for i in range(1, len(input_sizes)):
            layers.append(Linear(input_sizes[i - 1], input_sizes[i], bias=bias))
        self.layers = layers
        norm_layers = ModuleList()
        if len(input_sizes) > 2:
            for i in range(1, len(input_sizes) - 1):
                norm_layers.append(LayerNorm(input_sizes[i]))
        self.norm_layers = norm_layers
        self.drop_out = Dropout(p=dropout_prob)

    def forward(self, x):
        drop_out = self.drop_out
        layers = self.layers
        len_layers_1 = len(layers) - 1
        norm_layers = self.norm_layers
        len_norm_layers = len(norm_layers)
        for i, layer in enumerate(layers):
            x = layer(drop_out(x))
            if i < len_layers_1:
                x = gelu(x)
                if len_norm_layers:
                    x = norm_layers[i](x)
        return x


def mean_pooling(input, mask):
    mean_pooled = input.sum(dim=1) / mask.sum(dim=1, keepdim=True)
    return mean_pooled


class MeanPooling(Module):
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, doc_state, entity_mapping, entity_lens):
        entity_states = entity_mapping.unsqueeze(3) * doc_state.unsqueeze(1)  # N x E x L x d
        mean_pooled = torch_sum(entity_states, dim=2) / entity_lens.unsqueeze(2)
        return mean_pooled

class MeanMaxPooling(Module):
    def __init__(self):
        super(MeanMaxPooling, self).__init__()

    def forward(self, doc_state, entity_mapping, entity_lens):
        """
        :param doc_state:  N x L x d
        :param entity_mapping:  N x E x L
        :param entity_lens:  N x E
        :return: N x E x 2d
        """
        entity_states = entity_mapping.unsqueeze(3) * doc_state.unsqueeze(1)  # N x E x L x d
        max_pooled = torch_max(entity_states, dim=2)[0]
        mean_pooled = torch_sum(entity_states, dim=2) / entity_lens.unsqueeze(2)
        output = torch_cat([max_pooled, mean_pooled], dim=2)  # N x E x 2d
        return output

class LayerNorm(Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = Parameter(torch_ones(hidden_size))
        self.bias = Parameter(torch_zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        x_u = x - u
        s = x_u.pow(2).mean(-1, keepdim=True)
        x = x_u / torch_sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class GATSelfAttention(Module):
    def __init__(self, in_dim, out_dim, config, q_attn=False, head_id=0):
        """ One head GAT """
        super(GATSelfAttention, self).__init__()
        self.dropout = config.gnn_drop
        self.q_attn = q_attn
        self.n_type = n_type = config.num_edge_type
        input_dim = config.input_dim
        hidden_dim = config.hidden_dim
        self.step = 0
        q_update = config.q_update

        W_type = ParameterList()
        a_type = ParameterList()
        qattn_W1 = ParameterList()
        qattn_W2 = ParameterList()
        for _ in range(n_type):
            W_type.append(get_weights((in_dim, out_dim)))
            a_type.append(get_weights((out_dim * 2, 1)))

            if q_attn:
                q_dim = hidden_dim if q_update else input_dim
                qattn_W1.append(get_weights((q_dim, out_dim * 2)))
                qattn_W2.append(get_weights((out_dim * 2, out_dim * 2)))

        self.W_type = W_type
        self.a_type = a_type
        self.qattn_W1 = qattn_W1
        self.qattn_W2 = qattn_W2
        self.act = get_act('lrelu:0.2')

    def forward(self, input_state, adj, node_mask=None, query_vec=None):
        size = adj.size()
        dtype = adj.dtype
        device = adj.device
        zero_vec = torch_zeros_like(adj)
        scores = torch_zeros_like(adj)
        dropout = self.dropout
        training: bool = self.training
        q_attn: bool = self.q_attn
        W_type = self.W_type
        qattn_W1 = self.qattn_W1
        qattn_W2 = self.qattn_W2
        a_type = self.a_type
        act = self.act
        for i in range(self.n_type):
            h = torch_matmul(input_state, W_type[i])
            h = F_dropout(h, dropout, training)
            N, E, d = h.shape

            a_input = torch_cat([h.repeat(1, 1, E).view(N, E * E, -1), h.repeat(1, E, 1)], dim=-1)
            a_input = a_input.view(-1, E, E, 2*d)

            if q_attn:
                q_gate = F_relu(torch_matmul(query_vec, qattn_W1[i]))
                q_gate = torch_sigmoid(torch_matmul(q_gate, qattn_W2[i]))
                a_input = a_input * q_gate[:, None, None, :]
            score = act(torch_matmul(a_input, a_type[i]).squeeze(3))
            score += torch_where(adj == i+1, score, zero_vec.to(score.dtype))

        zero_vec = -1e30 * torch_ones_like(scores)
        scores = torch_where(adj > 0, scores, zero_vec.to(scores.dtype))

        # Ahead Alloc
        if node_mask is not None:
            h = h * node_mask

        coefs = F_softmax(scores, dim=2)  # N * E * E
        h = coefs.unsqueeze(3) * h.unsqueeze(2)  # N * E * E * d
        h = torch_sum(h, dim=1)
        return h


class AttentionLayer(Module):
    def __init__(self, in_dim, hid_dim, n_head, q_attn, config):
        super(AttentionLayer, self).__init__()
        assert hid_dim % n_head == 0
        self.dropout = config.gnn_drop

        self.attn_funcs = ModuleList()
        for i in range(n_head):
            self.attn_funcs.append(
                GATSelfAttention(in_dim=in_dim, out_dim=hid_dim // n_head, config=config, q_attn=q_attn, head_id=i))

        if in_dim != hid_dim:
            self.align_dim = Linear(in_dim, hid_dim)
            xavier_uniform_(self.align_dim.weight, gain=1.414)
        else:
            self.align_dim = lambda x: x

    def forward(self, input, adj, node_mask=None, query_vec=None):
        hidden_list = []
        for attn in self.attn_funcs:
            h = attn(input, adj, node_mask=node_mask, query_vec=query_vec)
            hidden_list.append(h)

        h = torch_cat(hidden_list, dim=-1)
        h = F_dropout(h, self.dropout, training=self.training)
        h = F_relu(h)
        return h


class BertLayerNorm(Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = Parameter(torch_ones(hidden_size))
        self.bias = Parameter(torch_zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        x_u = x - u
        s = x_u.pow(2).mean(-1, keepdim=True)
        x = x_u / torch_sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class OutputLayer(Module):
    def __init__(self, hidden_dim, config, num_answer=1):
        super(OutputLayer, self).__init__()

        self.output = Sequential(
            Linear(hidden_dim, hidden_dim*2),
            ReLU(),
            BertLayerNorm(hidden_dim*2, eps=1e-12),
            Dropout(config.trans_drop),
            Linear(hidden_dim*2, num_answer),
        )

    def forward(self, hidden_states):
        return self.output(hidden_states)

class GraphBlock(Module):
    def __init__(self, q_attn, config):
        super(GraphBlock, self).__init__()
        self.device = config.device
        self.hidden_dim = hidden_dim = config.hidden_dim
        self.q_update = q_update = config.q_updateq_update
        input_dim = config.input_dim

        if q_update:
            self.gat_linear = Linear(hidden_dim*2, hidden_dim)
            self.gat = AttentionLayer(hidden_dim, hidden_dim, config.num_gnn_heads, q_attn=q_attn, config=config)
            self.sent_mlp = OutputLayer(hidden_dim, config, num_answer=1)
            self.entity_mlp = OutputLayer(hidden_dim, config, num_answer=1)
        else:
            self.gat_linear = Linear(input_dim, hidden_dim*2)
            self.gat = AttentionLayer(hidden_dim*2, hidden_dim*2, config.num_gnn_heads, q_attn=q_attn, config=config)
            self.sent_mlp = OutputLayer(hidden_dim*2, config, num_answer=1)
            self.entity_mlp = OutputLayer(hidden_dim*2, config, num_answer=1)

    def forward(self, batch, input_state, query_vec):
        context_lens = batch['context_lens']
        context_mask = batch['context_mask']
        sent_mapping = batch['sent_mapping']
        sent_start_mapping = batch['sent_start_mapping']
        sent_end_mapping = batch['sent_end_mapping']
        para_mapping = batch['para_mapping']
        para_start_mapping = batch['para_start_mapping']
        para_end_mapping = batch['para_end_mapping']
        ent_mapping = batch['ent_mapping']
        ent_start_mapping = batch['ent_start_mapping']
        ent_end_mapping = batch['ent_end_mapping']

        hidden_dim = self.hidden_dim

        input_state_first = input_state[:, :, hidden_dim:]
        input_state_second = input_state[:, :, :hidden_dim]
        para_start_output = torch_bmm(para_start_mapping, input_state_first)   # N x max_para x d
        para_end_output = torch_bmm(para_end_mapping, input_state_second)       # N x max_para x d
        para_state = torch_cat([para_start_output, para_end_output], dim=-1)  # N x max_para x 2d

        sent_start_output = torch_bmm(sent_start_mapping, input_state_first)   # N x max_sent x d
        sent_end_output = torch_bmm(sent_end_mapping, input_state_second)       # N x max_sent x d
        sent_state = torch_cat([sent_start_output, sent_end_output], dim=-1)  # N x max_sent x 2d

        ent_start_output = torch_bmm(ent_start_mapping, input_state_first)   # N x max_ent x d
        ent_end_output = torch_bmm(ent_end_mapping, input_state_second)       # N x max_ent x d
        ent_state = torch_cat([ent_start_output, ent_end_output], dim=-1)  # N x max_ent x 2d

        N, max_para_num, _ = para_state.size()
        _, max_sent_num, _ = sent_state.size()
        _, max_ent_num, _ = ent_state.size()

        if self.q_update:
            graph_state = self.gat_linear(torch_cat([para_state, sent_state, ent_state], dim=1)) # N * (max_para + max_sent + max_ent) * d
            graph_state = torch_cat([query_vec.unsqueeze(1), graph_state], dim=1)
        else:
            graph_state = self.gat_linear(query_vec)
            graph_state = torch_cat([graph_state.unsqueeze(1), para_state, sent_state, ent_state], dim=1)
        node_mask = torch_cat([torch_ones(N, 1, device=self.device), batch['para_mask'], batch['sent_mask'], batch['ent_mask']], dim=-1).unsqueeze(-1)

        graph_adj = batch['graphs']
        assert graph_adj.size(1) == node_mask.size(1)

        graph_state = self.gat(graph_state, graph_adj, node_mask=node_mask, query_vec=query_vec) # N x (1+max_para+max_sent) x d
        ent_state = graph_state[:, 1+max_para_num+max_sent_num:, :]

        gat_logit = self.sent_mlp(graph_state[:, :1+max_para_num+max_sent_num, :]) # N x max_sent x 1
        para_logit = gat_logit[:, 1:1+max_para_num, :].contiguous()
        sent_logit = gat_logit[:, 1+max_para_num:, :].contiguous()

        query_vec = graph_state[:, 0, :].squeeze(1)

        ent_logit = self.entity_mlp(ent_state).view(N, -1)
        ent_logit = ent_logit - 1e30 * (1 - batch['ans_cand_mask'])

        para_logits_aux = Variable(para_logit.data.new(para_logit.size(0), para_logit.size(1), 1).zero_())
        para_prediction = torch_cat([para_logits_aux, para_logit], dim=-1).contiguous()

        sent_logits_aux = Variable(sent_logit.data.new(sent_logit.size(0), sent_logit.size(1), 1).zero_())
        sent_prediction = torch_cat([sent_logits_aux, sent_logit], dim=-1).contiguous()

        return input_state, graph_state, node_mask, sent_state, query_vec, para_logit, para_prediction, \
            sent_logit, sent_prediction, ent_logit

class GatedAttention(Module):
    def __init__(self, input_dim, memory_dim, hid_dim, dropout, gate_method='gate_att_up'):
        super(GatedAttention, self).__init__()
        self.gate_method = gate_method
        self.dropout = dropout
        self.input_linear_1 = Linear(input_dim, hid_dim, bias=True)
        self.memory_linear_1 = Linear(memory_dim, hid_dim, bias=True)

        self.input_linear_2 = Linear(input_dim + memory_dim, hid_dim, bias=True)

        self.dot_scale = np_sqrt(input_dim)

    def forward(self, input, memory, mask):
        """
        :param input: context_encoding N * Ld * d
        :param memory: query_encoding N * Lm * d
        :param mask: query_mask N * Lm
        :return:
        """
        bsz, input_len, memory_len = input.size(0), input.size(1), memory.size(1)

        input_dot = F_relu(self.input_linear_1(input))  # N x Ld x d
        memory_dot = F_relu(self.memory_linear_1(memory))  # N x Lm x d

        # N * Ld * Lm
        att = torch_bmm(input_dot, memory_dot.permute(0, 2, 1).contiguous()) / self.dot_scale

        att = att - 1e30 * (1 - mask[:, None])
        weight_one = F_softmax(att, dim=-1)
        output_one = torch_bmm(weight_one, memory)

        gate_method = self.gate_method
        input_linear_2 = self.input_linear_2
        if gate_method == 'no_gate':
            output = torch_cat( [input, output_one], dim=-1 )
            output = F_relu(input_linear_2(output))
        elif gate_method == 'gate_att_or':
            output = torch_cat( [input, input - output_one], dim=-1)
            output = F_relu(input_linear_2(output))
        elif gate_method == 'gate_att_up':
            output = torch_cat([input, output_one], dim=-1 )
            gate_sg = torch_sigmoid(input_linear_2(output))
            gate_th = torch_tanh(input_linear_2(output))
            output = gate_sg * gate_th
        else:
            raise ValueError("Not support gate method: {}".format(gate_method))


        return output, memory


class BiAttention(Module):
    def __init__(self, input_dim, memory_dim, hid_dim, dropout):
        super(BiAttention, self).__init__()
        self.dropout = dropout
        self.input_linear_1 = Linear(input_dim, 1, bias=False)
        self.memory_linear_1 = Linear(memory_dim, 1, bias=False)

        self.input_linear_2 = Linear(input_dim, hid_dim, bias=True)
        self.memory_linear_2 = Linear(memory_dim, hid_dim, bias=True)

        self.dot_scale = np_sqrt(input_dim)

    def forward(self, input, memory, mask):
        """
        :param input: context_encoding N * Ld * d
        :param memory: query_encoding N * Lm * d
        :param mask: query_mask N * Lm
        :return:
        """
        bsz, input_len, memory_len = input.size(0), input.size(1), memory.size(1)

        training = self.training
        dropout = self.dropout
        input = F_dropout(input, dropout, training=training)  # N x Ld x d
        memory = F_dropout(memory, dropout, training=training)  # N x Lm x d

        input_dot = self.input_linear_1(input)  # N x Ld x 1
        memory_dot = self.memory_linear_1(memory).view(bsz, 1, memory_len)  # N x 1 x Lm
        # N * Ld * Lm
        cross_dot = torch_bmm(input, memory.permute(0, 2, 1).contiguous()) / self.dot_scale
        # [f1, f2]^T [w1, w2] + <f1 * w3, f2>
        # (N * Ld * 1) + (N * 1 * Lm) + (N * Ld * Lm)
        att = input_dot + memory_dot + cross_dot  # N x Ld x Lm
        # N * Ld * Lm
        att = att - 1e30 * (1 - mask[:, None])

        input = self.input_linear_2(input)
        memory = self.memory_linear_2(memory)

        weight_one = F_softmax(att, dim=-1)
        output_one = torch_bmm(weight_one, memory)
        weight_two = F_softmax(att.max(dim=-1)[0], dim=-1).view(bsz, 1, input_len)
        output_two = torch_bmm(weight_two, input)

        return torch_cat([input, output_one, input*output_one, output_two*output_one], dim=-1), memory


class LSTMWrapper(Module):
    def __init__(self, input_dim, hidden_dim, n_layer, concat=False, bidir=True, dropout=0.3, return_last=True):
        super(LSTMWrapper, self).__init__()
        self.rnns = ModuleList()
        for i in range(n_layer):
            if i == 0:
                input_dim_ = input_dim
                output_dim_ = hidden_dim
            else:
                input_dim_ = hidden_dim if not bidir else hidden_dim * 2
                output_dim_ = hidden_dim
            self.rnns.append(LSTM(input_dim_, output_dim_, 1, bidirectional=bidir, batch_first=True))
        self.dropout = dropout
        self.concat = concat
        self.n_layer = n_layer
        self.return_last = return_last

    def forward(self, input, input_lengths=None):
        # input_length must be in decreasing order
        bsz, slen = input.size(0), input.size(1)
        output = input
        outputs = []

        if input_lengths is not None:
            lens = input_lengths.data.cpu().numpy()

        training: bool = self.training
        rnns = self.rnns
        dropout: float = self.dropout
        for i in range(self.n_layer):
            output: Tensor = F_dropout(output, p=dropout, training=training)

            if input_lengths is not None:
                output = pack_padded_sequence(output, lens, batch_first=True)

            output, _ = rnns[i](output)

            if input_lengths is not None:
                output, _ = pad_packed_sequence(output, batch_first=True)
                if output.size(1) < slen:  # used for parallel
                    padding = Variable(output.data.new(1, 1, 1).zero_())
                    output = torch_cat([output, padding.expand(output.size(0), slen-output.size(1), output.size(2))], dim=1)

            outputs.append(output)
        if self.concat:
            return torch_cat(outputs, dim=2)
        return outputs[-1]


class PredictionLayer(Module):
    """
    Identical to baseline prediction layer
    """
    def __init__(self, config, q_dim):
        super(PredictionLayer, self).__init__()
        self.config = config
        input_dim = config.ctx_attn_hidden_dim
        h_dim = config.hidden_dim

        self.hidden = h_dim

        self.start_linear = OutputLayer(input_dim, config, num_answer=1)
        self.end_linear = OutputLayer(input_dim, config, num_answer=1)
        self.type_linear = OutputLayer(input_dim, config, num_answer=4)

        self.cache_S = 0
        self.cache_mask = None

    def get_output_mask(self, outer):
        S = outer.size(1)
        if S <= self.cache_S:
            return Variable(self.cache_mask[:S, :S], requires_grad=False)
        self.cache_S = S
        np_mask = np_tril(np_triu(np_ones((S, S)), 0), 15)
        self.cache_mask = outer.data.new(S, S).copy_(torch_from_numpy(np_mask))
        return Variable(self.cache_mask, requires_grad=False)

    def forward(self, batch, context_input, sent_logits, ing_mask=None, return_yp=False):
        context_mask = batch['context_mask']
        inverse_context_mask = 1e30 * (1 - context_mask)
        start_prediction = self.start_linear(context_input).squeeze(2) - inverse_context_mask  # N x L
        end_prediction = self.end_linear(context_input).squeeze(2) - inverse_context_mask  # N x L
        type_prediction = self.type_linear(context_input[:, 0, :])

        if not return_yp:
            return start_prediction, end_prediction, type_prediction

        outer = start_prediction[:, :, None] + end_prediction[:, None]
        outer_mask = self.get_output_mask(outer)
        outer = outer - 1e30 * (1 - outer_mask[None].expand_as(outer))
        if ing_mask is not None:
            outer = outer - 1e30 * ing_mask[:, :, None]
        # yp1: start
        # yp2: end
        yp1 = outer.max(dim=2)[0].max(dim=1)[1]
        yp2 = outer.max(dim=1)[0].max(dim=1)[1]
        return start_prediction, end_prediction, type_prediction, yp1, yp2
