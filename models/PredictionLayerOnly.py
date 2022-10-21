
import torch.nn
from models.layers import *
import logging
logger = logging.getLogger(__name__)


class PredictionLayer(nn.Module):
    """
    Identical to baseline prediction layer
    """
    def __init__(self, config):
        super(PredictionLayer, self).__init__()
        self.config = config
        self.input_dim = config.input_dim
        h_dim = config.hidden_dim

        self.hidden = h_dim

        self.start_linear = OutputLayer(self.input_dim, config, num_answer=1)
        self.end_linear = OutputLayer(self.input_dim, config, num_answer=1)
        self.type_linear = OutputLayer(self.input_dim, config, num_answer=4)
        self.sent_mlp = OutputLayer(self.input_dim*2, config, num_answer=2)

        self.cache_S = 0
        self.cache_mask = None

    def get_output_mask(self, outer):
        S = outer.size(1)
        if S <= self.cache_S:
            return Variable(self.cache_mask[:S, :S], requires_grad=False)
        self.cache_S = S
        np_mask = np.tril(np.triu(np.ones((S, S)), 0), 15)
        self.cache_mask = outer.data.new(S, S).copy_(torch.from_numpy(np_mask))
        return Variable(self.cache_mask, requires_grad=False)

    def forward(self, batch, context_input, packing_mask=None, return_yp=False):
        context_mask = batch['context_mask']
        
        # sent pred
        sent_mapping = batch['sent_mapping']
        sent_start_mapping = batch['sent_start_mapping']
        sent_end_mapping = batch['sent_end_mapping']
        sent_start_output = torch.bmm(sent_start_mapping, context_input)   # N x max_sent x d
        sent_end_output = torch.bmm(sent_end_mapping, context_input)       # N x max_sent x d
        sent_state = torch.cat([sent_start_output, sent_end_output], dim=-1)  # N x max_sent x 2d       
        sent_logit = self.sent_mlp(sent_state) # N x max_sent x 1

        # span pred
        start_prediction = self.start_linear(context_input).squeeze(2) - 1e30 * (1 - context_mask)  # N x L
        end_prediction = self.end_linear(context_input).squeeze(2) - 1e30 * (1 - context_mask)  # N x L
        type_prediction = self.type_linear(context_input[:, 0, :])

        if not return_yp:
            return start_prediction, end_prediction, type_prediction, sent_logit

        outer = start_prediction[:, :, None] + end_prediction[:, None]
        outer_mask = self.get_output_mask(outer)
        outer = outer - 1e30 * (1 - outer_mask[None].expand_as(outer))
        if packing_mask is not None:
            outer = outer - 1e30 * packing_mask[:, :, None]
        # yp1: start
        # yp2: end
        yp1 = outer.max(dim=2)[0].max(dim=1)[1]
        yp2 = outer.max(dim=1)[0].max(dim=1)[1]
        return start_prediction, end_prediction, type_prediction, sent_logit, yp1, yp2