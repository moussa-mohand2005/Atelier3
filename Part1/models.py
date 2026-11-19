import torch
import torch.nn as nn

ARCHS = {
    "RNN": (nn.RNN, False),
    "BiRNN": (nn.RNN, True),
    "GRU": (nn.GRU, False),
    "LSTM": (nn.LSTM, False),
}


class SeqRegressor(nn.Module):

    def __init__(self, arch, vocab_size, emb_dim=128, hidden=128, layers=1, dropout=0.3):
        super().__init__()
        cell, bidirectional = ARCHS[arch]
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        kwargs = dict(batch_first=True, num_layers=layers, bidirectional=bidirectional,
                      dropout=dropout if layers > 1 else 0.0)
        if cell is nn.RNN:
            kwargs["nonlinearity"] = "tanh"
        self.rnn = cell(emb_dim, hidden, **kwargs)
        out_dim = hidden * (2 if bidirectional else 1)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(2 * out_dim, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1))

    def forward(self, x, lengths):
        emb = self.drop(self.emb(x))
        packed = nn.utils.rnn.pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.rnn(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=x.size(1))
        mask = (x != 0).unsqueeze(-1).float()
        mean = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
        mx = out.masked_fill(mask == 0, -1e4).max(1).values
        h = self.drop(torch.cat([mean, mx], dim=1))
        return 10 * torch.sigmoid(self.head(h)).squeeze(-1)
