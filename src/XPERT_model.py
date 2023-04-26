import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import scipy.sparse as sp

def expand(offsets):
    expanded = np.empty(offsets[-1], dtype=int)
    sp._sparsetools.expandptr(len(offsets) - 1, offsets, expanded)
    return expanded.tolist()

class segment1(nn.Module):
    """
        Takes in History items and applies a transformer model to get the user embeddings.
        Returns a (B, D) tensor
    """

    def __init__(self, config, dropout, emb_dim, hid_dim, device):
        super().__init__()
        self.dropout = dropout
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.device = device
        self.type = config['type']
        
        self.nlayers = config['nlayer']
        self.nhead = config['nhead']
        self.pos_encoding = config['pos_encoding']

        encoder_layer = nn.TransformerEncoderLayer(emb_dim, self.nhead, config['t_hid_dim'], dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, self.nlayers)
        self.linear = nn.Linear(self.emb_dim, self.hid_dim)

    def forward(self, batch, item_emb):
        items = torch.tensor(batch['history_items'], dtype=torch.int64).to(self.device)  # [B, H]

        mask = (items < 0)
        his_len_mask = (batch['history_lens'] > 0)

        items = items[his_len_mask]
        mask = mask[his_len_mask]

        his_item_emb = (item_emb[items.transpose(0, 1)]*(~mask).transpose(0, 1).unsqueeze(2))  # [H, B', E]
        # where B' = # of users with more than zero history items in the current batch

        # Do masking again
        x = self.transformer_encoder(his_item_emb, src_key_padding_mask=mask)  # [H, B', E]
        x = self.linear(x)  # [H, B', E_hidden]
        x = x * (~mask).transpose(0, 1).unsqueeze(2)
        user_emb_masked = x.sum(0)  # [B', E_hidden]
        user_emb_masked = F.normalize(user_emb_masked, p=2, dim=1)

        user_emb = torch.zeros(batch['history_items'].shape[0], self.hid_dim, dtype=torch.float32, device=self.device)

        user_emb[his_len_mask] = user_emb_masked
        return user_emb


class segment2(nn.Module):
    """
        Takes in a (B, D) dense tensor and returns a (B, D', D') tensor
        Parameters:
            in_dim: Hidden user embedding dimension after Segment1
            out_dim: Hidden item embedding dimension in Segment3
            num_heads: # of FC heads
    """

    def __init__(self, config, in_dim, out_dim, device):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.type = config['type']
        
        self.combine_heads = config['combine_heads']
        self.num_heads = config['num_heads']
        self.FC = nn.Linear(self.in_dim, self.out_dim * self.out_dim * self.num_heads)

    def mean_combine_heads(self, FC_heads_out):
        return torch.mean(FC_heads_out, dim=-1).reshape(-1, self.out_dim, self.out_dim)

    def forward(self, x):
        fc_out = F.relu(self.FC(x).reshape(-1, self.out_dim * self.out_dim, self.num_heads))
        if self.combine_heads == 'mean':
            return self.mean_combine_heads(fc_out)


class segment3(nn.Module):
    """
        Takes in Seg2_output,batch and 
        returns  { (B* summation(len(N_i), D) Tensor denoting the personalized embeddings of items in N
    """

    def __init__(self, config, device):
        super().__init__()
        self.type = config['type']
        self.device = device

    def forward(self, batch, item_emb, seg2_out):
        items = torch.tensor(batch['novel_items'], dtype=torch.int64).to(self.device)
        emb = item_emb[items]
        if 'novel_userids' in batch:
            exp_ind = torch.tensor(batch['novel_userids'], dtype=torch.int64).to(self.device)
        else:
            exp_ind = torch.tensor(expand(np.append(batch['novel_offsets'], items.shape[0]))).to(self.device)
        exp_morph = seg2_out[exp_ind]

        # EOE
        emb_morph = emb + torch.bmm(exp_morph.transpose(1, 2), emb.unsqueeze(2)).squeeze(2)

        return F.normalize(emb_morph, p=2, dim=1)


class XPERT(nn.Module):
    """
        Takes in the Batch data and outputs a (B, D') tensor representing the personalized
        embeddings of the novel behaviour items in the batch.
    """

    def __init__(self, config, device):
        super().__init__()
        self.emb_dim = config['emb_dim']
        self.hid_dim = config['hid_dim']
        self.dropout = config['dropout']
        self.device = device

        self.segment1 = segment1(config['seg1'], self.dropout, self.emb_dim, self.hid_dim, self.device)
        self.segment2 = segment2(config['seg2'], self.hid_dim, self.emb_dim, self.device)
        self.segment3 = segment3(config['seg3'], device)

    def forward(self, batch):
        # Extracting data from batch

        item_emb = F.normalize(torch.from_numpy(batch['item_features']).float().to(self.device), p=2, dim=1)
        user_emb = self.segment1(batch, item_emb)

        # XPERT
        seg2_out = self.segment2(user_emb)
        personalized_novel = self.segment3(batch, item_emb, seg2_out)
        
        return personalized_novel, item_emb