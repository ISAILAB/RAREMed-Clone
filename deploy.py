import torch
import torch.nn as nn


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, kernel_size=3, dropout=0, max_len=1000):
        super(LearnablePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout1d(p=dropout)

        self.conv1d = nn.Conv1d(
            in_channels=1,
            out_channels=d_model,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        self.embeddings = nn.Embedding(max_len, d_model)

        initrange = 0.1
        self.conv1d.weight.data.uniform_(-initrange, initrange)
        self.embeddings.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        batch_size, seq_length, d_model = x.shape
        print("Input x:\n", x)

        pos = torch.arange(0, seq_length, device=x.device).int().unsqueeze(0)
        print("Position indices:\n", pos)

        pos_embed = self.embeddings(pos).expand(batch_size, seq_length, d_model)
        print("Positional embeddings:\n", pos_embed)

        print("Conv1D Kernel:\n", self.conv1d.weight)

        x_reshaped = x.view(batch_size * seq_length, 1, d_model)
        print("Reshaped x for Conv1D:\n", x_reshaped)

        conv_out = self.conv1d(x_reshaped)
        print("Output after Conv1D:\n", conv_out)

        conv_out = self.global_avg_pool(conv_out)
        print("Output after Global Avg Pooling:\n", conv_out)

        conv_out = conv_out.view(batch_size, seq_length, d_model)
        print("Reshaped output after pooling:\n", conv_out)

        out = conv_out + pos_embed
        print("Final output before dropout:\n", out)

        return self.dropout(out)


# Test
d_model = 5
seq_length = 3
batch_size = 1

x = torch.randn(batch_size, seq_length, d_model)
model = LearnablePositionalEncoding(d_model)
output = model(x)
print("Final output after dropout:\n", output)
