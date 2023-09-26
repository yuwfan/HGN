from torch.nn import Module, ModuleList, Linear
from models.layers import BiAttention, LSTMWrapper, GatedAttention, PredictionLayer, GraphBlock, mean_pooling


class HierarchicalGraphNetwork(Module):
    """
    Packing Query Version
    """
    def __init__(self, config):
        super(HierarchicalGraphNetwork, self).__init__()
        # Caching
        input_dim: int = config.input_dim
        hidden_dim: int = config.hidden_dim
        bi_attn_drop: float = config.bi_attn_drop
        self.max_query_length: int = config.max_query_length
        self.num_gnn_layers = num_gnn_layers = config.num_gnn_layers
        self.q_update = q_update = config.q_update

        self.bi_attention = BiAttention(input_dim=input_dim,
                                        memory_dim=input_dim,
                                        hid_dim=hidden_dim,
                                        dropout=bi_attn_drop)
        self.bi_attn_linear = Linear(hidden_dim * 4, hidden_dim)
        self.sent_lstm = LSTMWrapper(input_dim=hidden_dim,
                                     hidden_dim=hidden_dim,
                                     n_layer=1,
                                     dropout=config.lstm_drop)

        graph_blocks = ModuleList()
        q_attn: bool = config.q_attn
        for _ in range(num_gnn_layers):
            graph_blocks.append(GraphBlock(q_attn, config))
        self.graph_blocks = graph_blocks
        self.ctx_attention = GatedAttention(input_dim=hidden_dim*2,
                                            memory_dim=hidden_dim if q_update else hidden_dim*2,
                                            hid_dim=config.ctx_attn_hidden_dim,
                                            dropout=bi_attn_drop,
                                            gate_method=config.ctx_attn)

        q_dim: int = hidden_dim if q_update else input_dim
        self.predict_layer = PredictionLayer(config, q_dim)
    def forward(self, batch, return_yp):
        query_mapping = batch['query_mapping']
        context_encoding = batch['context_encoding']

        # extract query encoding
        max_query_length: int = self.max_query_length
        trunc_query_mapping = query_mapping[:, :max_query_length].contiguous()
        trunc_query_state = (context_encoding * query_mapping.unsqueeze(2))[:, :max_query_length, :].contiguous()
        # bert encoding query vec
        query_vec = mean_pooling(trunc_query_state, trunc_query_mapping)

        attn_output, trunc_query_state = self.bi_attention(context_encoding,
                                                           trunc_query_state,
                                                           trunc_query_mapping)

        input_state = self.bi_attn_linear(attn_output) # N x L x d
        input_state = self.sent_lstm(input_state, batch['context_lens'])

        if self.q_update:
            query_vec = mean_pooling(trunc_query_state, trunc_query_mapping)

        para_logits, sent_logits = [], []
        para_predictions, sent_predictions, ent_predictions = [], [], []

        graph_blocks = self.graph_blocks
        for l in range(self.num_gnn_layers):
            new_input_state, graph_state, graph_mask, sent_state, query_vec, para_logit, para_prediction, \
            sent_logit, sent_prediction, ent_logit = graph_blocks[l](batch, input_state, query_vec)

            para_logits.append(para_logit)
            sent_logits.append(sent_logit)
            para_predictions.append(para_prediction)
            sent_predictions.append(sent_prediction)
            ent_predictions.append(ent_logit)

        input_state, _ = self.ctx_attention(input_state, graph_state, graph_mask.squeeze(-1))
        predictions = self.predict_layer(batch, input_state, sent_logits[-1], packing_mask=query_mapping, return_yp=return_yp)

        if return_yp:
            start, end, q_type, yp1, yp2 = predictions
            return start, end, q_type, para_predictions[-1], sent_predictions[-1], ent_predictions[-1], yp1, yp2
        else:
            start, end, q_type = predictions
            return start, end, q_type, para_predictions[-1], sent_predictions[-1], ent_predictions[-1]
