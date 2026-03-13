import torch.nn as nn
import torch.nn.functional as F


class ECG_LSTM(nn.Module):
    def __init__(self, input_size=16, hidden_size=64, num_layers=2, num_classes=5, dropout=0.3):
        super(ECG_LSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: [batch_size, seq_len=1, input_size=16]
        lstm_out, _ = self.lstm(x)  # lstm_out: [batch, seq_len, hidden_size]
        out = self.fc(lstm_out[:, -1, :])
        return out

class ECG_BiLSTM(nn.Module):
    def __init__(self, input_size=16, hidden_size=64, num_layers=2, num_classes=5, dropout=0.3):
        super(ECG_BiLSTM, self).__init__()
        self.bilstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        # x: [batch_size, seq_len=1, input_size=16]
        bilstm_out, _ = self.bilstm(x)
        out = self.fc(bilstm_out[:, -1, :])
        return out



class TemporalMultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.layernorm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, D)
        attn_out, attn_weights = self.attn(x, x, x, need_weights=True)
        out = self.layernorm(x + self.dropout(attn_out))
        return out, attn_weights


class AMHBilstmModel(nn.Module):
    def __init__(
        self,
        num_features: int,
        num_classes: int,
        cnn_channels: int = 64,
        lstm_hidden: int = 128,
        lstm_layers: int = 1,
        num_heads: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()

        # Conv1d over time; input will be (B, D, T)
        self.conv1 = nn.Conv1d(num_features, cnn_channels, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(cnn_channels)

        self.conv2 = nn.Conv1d(cnn_channels, cnn_channels, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(cnn_channels)

        self.relu = nn.ReLU()

        self.bilstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
        )

        d_model = 2 * lstm_hidden
        self.temporal_attn = TemporalMultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.fc1 = nn.Linear(d_model, d_model)
        self.fc2 = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, D) -> (B, D, T) for Conv1d
        B, T, D = x.shape

        x = x.permute(0, 2, 1)        # (B, D, T)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)              # (B, C, T)

        x = x.permute(0, 2, 1)        # (B, T, C) for LSTM

        lstm_out, _ = self.bilstm(x)  # (B, T, 2*hidden)

        attn_out, attn_weights = self.temporal_attn(lstm_out)  # (B, T, D_model)

        # Global average pool over time
        pooled = attn_out.mean(dim=1)  # (B, D_model)

        z = self.dropout(F.relu(self.fc1(pooled)))
        logits = self.fc2(z)           # (B, num_classes)

        return logits, attn_weights