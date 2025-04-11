import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.nn import LayerNorm
from torch.autograd import Variable
import math


# RareMed
class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0, max_len=1000):
        super(LearnablePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.embeddings = nn.Embedding(max_len, d_model)

        initrange = 0.1
        self.embeddings.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        pos = torch.arange(0, x.size(1), device=x.device).int().unsqueeze(0)
        x = x + self.embeddings(pos).expand_as(x)
        return x

# Method 1
# class LearnablePositionalEncoding(nn.Module):
#     def __init__(self, d_model, kernel_size=3, dropout=0, max_len=1000):
#         super(LearnablePositionalEncoding, self).__init__()
#         self.dropout = nn.Dropout(p=dropout)

#         self.conv1d = nn.Conv1d(
#             in_channels=1,
#             out_channels=d_model,
#             kernel_size=kernel_size,
#             padding=kernel_size // 2,
#         )

#         self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

#         self.embeddings = nn.Embedding(max_len, d_model)

#         initrange = 0.1
#         self.conv1d.weight.data.uniform_(-initrange, initrange)
#         self.embeddings.weight.data.uniform_(-initrange, initrange)

#     def forward(self, x):
#         batch_size, seq_length, d_model = x.shape

#         pos = (
#             torch.arange(0, seq_length, device=x.device).int().unsqueeze(0)
#         )
#         pos_embed = self.embeddings(pos).expand(
#             batch_size, seq_length, d_model
#         )

#         x_reshaped = x.view(batch_size * seq_length, 1, d_model)
#         conv_out = self.conv1d(x_reshaped)
#         conv_out = self.global_avg_pool(conv_out)
#         conv_out = conv_out.view(batch_size, seq_length, d_model)

#         out = conv_out + pos_embed

#         return self.dropout(out)

# Method 2
# class LearnablePositionalEncoding(nn.Module):
#     def __init__(self, d_model, kernel_size=3, dropout=0, max_len=1000):
#         super(LearnablePositionalEncoding, self).__init__()
#         self.dropout = nn.Dropout(p=dropout)

#         self.conv1d = nn.Conv1d(
#             in_channels=1,
#             out_channels=d_model,
#             kernel_size=kernel_size,
#             padding=kernel_size // 2,
#             # bias=False,
#         )

#         self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

#         # self.embeddings = nn.Embedding(max_len, d_model)

#         initrange = 0.1
#         self.conv1d.weight.data.uniform_(-initrange, initrange)
#         # self.embeddings.weight.data.uniform_(-initrange, initrange)

#     def forward(self, x):
#         batch_size, seq_length, d_model = x.shape

#         # pos = torch.arange(0, seq_length, device=x.device).int().unsqueeze(0)
#         # pos_embed = self.embeddings(pos).expand(batch_size, seq_length, d_model)

#         x_reshaped = x.view(batch_size * seq_length, 1, d_model)
#         conv_out = self.conv1d(x_reshaped)
#         conv_out = self.global_avg_pool(conv_out)
#         conv_out = conv_out.view(batch_size, seq_length, d_model)

#         x_reshaped_2 = conv_out.view(batch_size * seq_length, 1, d_model)
#         conv_out_2 = self.conv1d(x_reshaped_2)
#         conv_out_2 = self.global_avg_pool(conv_out_2)
#         conv_out_2 = conv_out_2.view(batch_size, seq_length, d_model)

#         out = conv_out + conv_out_2

#         return self.dropout(out)

# Method 3
# Thêm Relu sau phần biến đổi
# class LearnablePositionalEncoding(nn.Module):
#     def __init__(self, d_model, dropout=0.2, max_len=1000):
#         super().__init__()
#         self.dropout = nn.Dropout(p=dropout)
#         self.embeddings = nn.Embedding(max_len, d_model)
#         self.d_model = d_model

#         # initrange = 0.1
#         # self.embeddings.weight.data.uniform_(-initrange, initrange)
#         nn.init.kaiming_normal_(
#             self.embeddings.weight, a=0, mode="fan_in", nonlinearity="relu"
#         )

#         self.linear = nn.Linear(d_model, d_model)

#     def forward(self, x):

#         pos = torch.arange(0, x.size(1), device=x.device).unsqueeze(0)
#         position_embedding = self.embeddings(pos)
#         position_embedding = position_embedding.expand_as(x)

#         x_proj = F.relu(self.linear(x))

#         out = x_proj + position_embedding
#         return self.dropout(out)

# Method 4
# class LearnablePositionalEncoding(nn.Module):
#     def __init__(self, d_model, dropout=0.1, max_len=1000):
#         super(LearnablePositionalEncoding, self).__init__()
#         self.dropout = nn.Dropout(p=dropout)
#         self.embeddings = nn.Embedding(max_len, d_model)
#         self.d_model = d_model
#         # self.norm = nn.LayerNorm(d_model)
#         initrange = 0.1
#         self.embeddings.weight.data.uniform_(-initrange, initrange)
#         self.W = nn.Parameter(torch.empty(d_model, d_model))
#         self.W2 = nn.Parameter(torch.empty(d_model, d_model))
#         nn.init.xavier_uniform_(self.W)
#         nn.init.xavier_uniform_(self.W2)

#     def forward(self, x):
#         # pos = torch.arange(0, x.size(1), device=x.device).int().unsqueeze(0)
#         # weighted_matrix = torch.randn(self.d_model, self.d_model).to(x.device)
#         multiplied_matrix = torch.matmul(x, self.W)
#         # position_embedding = self.embeddings(pos).expand_as(x)
#         multiplied_matrix2 = torch.matmul(multiplied_matrix, self.W2)
#         x = multiplied_matrix + multiplied_matrix2
#         # x = self.norm(x)
#         return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.2, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.embeddings = nn.Embedding(max_len, d_model)
        self.d_model = d_model

        # He (Kaiming) initialization
        nn.init.kaiming_uniform_(
            self.embeddings.weight, a=0, mode="fan_in", nonlinearity="relu"
        )

        self.linear = nn.Linear(d_model, d_model)

    def forward(self, x):
        pos = torch.arange(0, x.size(1), device=x.device).unsqueeze(0)
        position_embedding = self.embeddings(pos)
        position_embedding = position_embedding.expand_as(x)

        x_proj = self.linear(x)
        x_proj = F.relu(x_proj)  # Add ReLU here

        out = x_proj + position_embedding
        return self.dropout(out)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        pe *= 0.1
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, : x.size(1)], requires_grad=False)
        return self.dropout(x)


class PatientEncoder(nn.Module):  # 把patient_encoder抽象出来，方便后续的模型调用
    def __init__(self, args, voc_size):
        super(PatientEncoder, self).__init__()
        self.args = args
        self.voc_size = voc_size
        self.emb_dim = args.embed_dim  # 如果有预训练参数，保持与预训练的维度一致
        self.device = torch.device("cuda:{}".format(args.cuda))

        # self.embeddings = nn.ModuleList(
        #     [nn.Embedding(voc_size[i], self.emb_dim) for i in range(2)])  # 疾病 手术

        # self.special_embeddings = nn.Embedding(2, self.emb_dim)                # 增加两个token：[CLS]， [SEP]
        self.special_tokens = {
            "CLS": torch.LongTensor(
                [
                    0,
                ]
            ).to(self.device),
            "SEP": torch.LongTensor(
                [
                    1,
                ]
            ).to(self.device),
        }

        self.segment_embedding = nn.Embedding(2, self.emb_dim)

        # self.positional_embedding_layer = PositionalEncoding(d_model=args.embed_dim)
        # self.positional_embedding_layer = LearnablePositionalEncoding(d_model=args.embed_dim)

        if args.patient_seperate == False:
            self.embeddings = nn.ModuleList(
                [nn.Embedding(voc_size[i], self.emb_dim) for i in range(2)]
            )  # 疾病 手术
            self.special_embeddings = nn.Embedding(
                2, self.emb_dim
            )  # 增加两个token：[CLS]， [SEP]
            self.transformer_visit = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=self.emb_dim, nhead=args.nhead, dropout=args.dropout
                ),
                num_layers=args.encoder_layers,
            )
            self.positional_embedding_layer_disease = LearnablePositionalEncoding(
                d_model=args.embed_dim
            )
            self.positional_embedding_layer_procedure = LearnablePositionalEncoding(
                d_model=args.embed_dim
            )
            self.patient_encoder = self.patient_encoder_unified
        else:
            self.embeddings = nn.ModuleList(
                [nn.Embedding(voc_size[i], self.emb_dim // 2) for i in range(2)]
            )  # 疾病 手术
            self.special_embeddings = nn.Embedding(
                2, self.emb_dim // 2
            )  # 增加两个token：[CLS]， [SEP]
            self.transformer_disease = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=self.emb_dim // 2, nhead=args.nhead, dropout=args.dropout
                ),
                num_layers=args.encoder_layers,
            )
            self.transformer_procedure = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=self.emb_dim // 2, nhead=args.nhead, dropout=args.dropout
                ),
                num_layers=args.encoder_layers,
            )

            self.patient_layer = nn.Sequential(
                nn.Linear(self.emb_dim, self.emb_dim),
                nn.ReLU(),
                nn.Linear(self.emb_dim, self.emb_dim),
            )

            self.positional_embedding_layer_disease = LearnablePositionalEncoding(
                d_model=args.embed_dim // 2
            )
            self.positional_embedding_layer_procedure = LearnablePositionalEncoding(
                d_model=args.embed_dim // 2
            )

            self.patient_encoder = self.patient_encoder_seperate

    def patient_encoder_seperate(self, batch_visits):
        device = self.device

        batch_disease_repr, batch_procedure_repr = [], []
        for adm in batch_visits:
            # 对每次访问：
            diseases = adm[0]
            procedures = adm[1]

            disease_embedding = self.embeddings[0](
                torch.LongTensor(diseases).unsqueeze(dim=1).to(self.device)
            )  # (n, 1, dim/2)
            procedure_embedding = self.embeddings[1](
                torch.LongTensor(procedures).unsqueeze(dim=1).to(self.device)
            )  # (m, 1, dim/2)

            # 加入CLS token
            cls_embedding_dis = self.special_embeddings(
                self.special_tokens["CLS"]
            ).unsqueeze(
                dim=1
            )  # (1, 1, dim/2)
            cls_embedding_pro = self.special_embeddings(
                self.special_tokens["SEP"]
            ).unsqueeze(
                dim=1
            )  # (1, 1, dim/2)
            disease_embedding = torch.cat(
                (cls_embedding_dis, disease_embedding), dim=0
            )  # (n+1, 1, dim/2)
            procedure_embedding = torch.cat(
                (cls_embedding_pro, procedure_embedding), dim=0
            )  # (m+1, 1, dim/2)

            disease_embedding = self.positional_embedding_layer_disease(
                disease_embedding
            )
            procedure_embedding = self.positional_embedding_layer_procedure(
                procedure_embedding
            )

            disease_representation = self.transformer_disease(disease_embedding)[
                0
            ]  # (n+1,1,dim/2)
            procedure_representation = self.transformer_procedure(procedure_embedding)[
                0
            ]  # (m+1,1,dim/2)

            disease_representation = disease_representation.mean(dim=0)  # (1,1,dim/2)
            procedure_representation = procedure_representation.mean(
                dim=0
            )  # (1,1,dim/2)

            disease_representation = torch.reshape(
                disease_representation, (1, 1, -1)
            )  # (1,1,dim/2)
            procedure_representation = torch.reshape(
                procedure_representation, (1, 1, -1)
            )  # (1,1,dim/2)

            batch_disease_repr.append(disease_representation)
            batch_procedure_repr.append(procedure_representation)

        batch_disease_repr = torch.cat(batch_disease_repr, dim=1).to(
            device
        )  # (1, B, dim/2)
        batch_procedure_repr = torch.cat(batch_procedure_repr, dim=1).to(
            device
        )  #  (1, B, dim/2)

        batch_repr = torch.cat(
            (batch_disease_repr, batch_procedure_repr), dim=-1
        )  # (1, B, dim)
        batch_repr = batch_repr.squeeze(dim=0)  # (B, dim)
        # batch_repr = self.patient_layer(batch_repr)  # (B, dim)

        return batch_repr

    def patient_encoder_unified(self, batch_visits):
        batch_repr = []
        for adm in batch_visits:
            # 对每次访问：
            diseases = adm[0]
            procedures = adm[1]
            disease_embedding = self.embeddings[0](
                torch.LongTensor(diseases).unsqueeze(dim=1).to(self.device)
            )  # (n, 1, dim)
            procedure_embedding = self.embeddings[1](
                torch.LongTensor(procedures).unsqueeze(dim=1).to(self.device)
            )  # (m, 1, dim)

            cls_embedding = self.special_embeddings(
                self.special_tokens["CLS"]
            ).unsqueeze(dim=1)
            sep_embedding = self.special_embeddings(
                self.special_tokens["SEP"]
            ).unsqueeze(dim=1)

            disease_embedding = torch.cat(
                (cls_embedding, disease_embedding), dim=0
            )  # (n+1, 1, dim)
            procedure_embedding = torch.cat(
                (sep_embedding, procedure_embedding), dim=0
            )  # (m+1, 1, dim)

            disease_embedding = self.positional_embedding_layer_disease(
                disease_embedding
            )
            procedure_embedding = self.positional_embedding_layer_procedure(
                procedure_embedding
            )

            # disease_embedding = self.positional_embedding_layer(disease_embedding)
            # procedure_embedding = self.positional_embedding_layer(procedure_embedding)

            combined_embedding = torch.cat(
                (disease_embedding, procedure_embedding), dim=0
            )  # (n+m+2, 1, dim)
            # combined_embedding = torch.cat((cls_embedding, disease_embedding, sep_embedding, procedure_embedding), dim=0)  # (n+m+2, 1, dim)

            # 加入segment embedding
            segments = torch.tensor(
                [0] * (len(diseases) + 2) + [1] * len(procedures)
            ).to(self.device)
            segment_embedding = self.segment_embedding(segments).unsqueeze(dim=1)
            input_embedding = combined_embedding + segment_embedding

            visit_representation = self.transformer_visit(input_embedding)[0]
            visit_representation = torch.reshape(
                visit_representation, (1, 1, -1)
            )  # (1,1,dim)
            batch_repr.append(visit_representation)
        batch_repr = torch.cat(batch_repr, dim=1).to(self.device)  # (1,B,dim)
        batch_repr = batch_repr.squeeze(dim=0)  # (B,dim)
        return batch_repr


class RAREMed(PatientEncoder):
    def __init__(self, args, voc_size, ddi_adj):
        super(RAREMed, self).__init__(args, voc_size)
        self.tensor_ddi_adj = torch.FloatTensor(ddi_adj).to(self.device)

        # self.patient_layer = Adapter(self.emb_dim, args.adapter_dim)

        self.init_weights()

        # self.mask_adapter = Adapter(self.emb_dim, args.adapter_dim)
        self.cls_mask = nn.Linear(self.emb_dim, self.voc_size[0] + self.voc_size[1])

        # self.nsp_adapter = Adapter(self.emb_dim, args.adapter_dim)
        self.cls_nsp = nn.Linear(self.emb_dim, 1)

        self.cls_final = nn.Linear(self.emb_dim, self.voc_size[2])

    def forward_finetune(self, input):
        # B: batch size, D: drug num, dim: embedding dimension
        patient_repr = self.patient_encoder(input)  # (B,dim)
        # patient_repr = self.patient_layer(patient_repr)  # (B,dim)

        result = self.cls_final(patient_repr)  # (B,D)

        neg_pred_prob = F.sigmoid(result)  # (B,D)
        neg_pred_prob = torch.matmul(
            neg_pred_prob.t(), neg_pred_prob
        )  # (voc_size, voc_size)

        batch_neg = 0.0005 * neg_pred_prob.mul(self.tensor_ddi_adj).sum()

        # prompt the probability of indication medications to be higher
        # indication_medication = input[-1][4]
        # indication_medication = torch.LongTensor(indication_medication).to(self.device)
        # contraindication_medication = input[-1][5]
        # contraindication_medication = torch.LongTensor(contraindication_medication).to(self.device)

        # output_mean_0 = result.mean(dim=1)
        # result[0, indication_medication] += 2
        # result = result - result.mean(dim=1) + output_mean_0

        # output_mean_0 = result.mean(dim=1)
        # result[0, contraindication_medication] -= 0.1
        # result = result - result.mean(dim=1) + output_mean_0

        return result, batch_neg

    def forward(self, input, mode="fine-tune"):
        assert mode in ["fine-tune", "pretrain_mask", "pretrain_nsp"]
        if mode == "fine-tune":
            result, batch_neg = self.forward_finetune(input)
            return result, batch_neg

        elif mode == "pretrain_mask":
            patient_repr = self.patient_encoder(input)  # (B, dim)
            # patient_repr = self.mask_adapter(patient_repr)  # (B, dim)
            result = self.cls_mask(patient_repr)  # (B, voc_size[0]+voc_size[1])
            return result

        elif mode == "pretrain_nsp":
            patient_repr = self.patient_encoder(input)  # (B, dim)
            # patient_repr = self.nsp_adapter(patient_repr)   # (B, dim)
            result = self.cls_nsp(patient_repr)  # (B, 1)
            result = result.squeeze(dim=1)  # (B,)
            logit = F.sigmoid(result)  # logistic regression
            return logit

    def init_weights(self):
        """Initialize embedding weights."""
        initrange = 0.1
        self.embeddings[0].weight.data.uniform_(-initrange, initrange)  # disease
        self.embeddings[1].weight.data.uniform_(-initrange, initrange)  # procedure

        self.segment_embedding.weight.data.uniform_(-initrange, initrange)
        self.special_embeddings.weight.data.uniform_(-initrange, initrange)
