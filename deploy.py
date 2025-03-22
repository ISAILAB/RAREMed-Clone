# from fastapi import FastAPI
# import torch
# import dill
# import numpy as np
# import argparse
# from pydantic import BaseModel

# # from models.RAREMed import RAREMed
# import warnings

# warnings.filterwarnings(
#     "ignore", category=UserWarning, module="torch.nn.modules.transformer"
# )

# # Khởi tạo FastAPI
# app = FastAPI()

# # Đường dẫn model và vocab cho MIMIC-III và MIMIC-IV
# DATASET_PATHS = {
#     "mimic-iii": {
#         "model": r"/home/nguyenkhanh/Documents/AILAB/RAREMed/RAREMed-Clone/src/log/mimic-iii/RAREMed/log17_/Epoch_9_JA_0.5329_DDI_0.07024.model",
#         "vocab": r"/home/nguyenkhanh/Documents/AILAB/RAREMed/RAREMed-Clone/data/output/mimic-iii/voc_final.pkl",
#     },
#     # "mimic-iv": {
#     #     "model": r"C:\Users\CHuy\RAREMed\src\log\mimic-iv\RAREMed\log0_\Epoch_9_JA_0.4551_DDI_0.07102.model",
#     #     "vocab": r"/home/nguyenkhanh/Documents/AILAB/RAREMed/RAREMed-Clone/data/output/mimic-iv/voc_final.pkl",
#     # },
# }

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Tạo args với các giá trị mặc định
# args = argparse.Namespace(
#     embed_dim=512,  # Kích thước embedding
#     encoder_layers=3,
#     nhead=4,
#     dropout=0.3,
#     adapter_dim=128,
#     patient_seperate=False,  # Thêm thuộc tính này để tránh lỗi
# )


# # Hàm thêm từ mới vào vocab
# def add_word(word, voc):
#     if word not in voc.word2idx:
#         voc.word2idx[word] = len(voc.word2idx)
#         voc.idx2word[len(voc.idx2word)] = word
#     return voc


# # Định nghĩa input dữ liệu
# class InputData(BaseModel):
#     dataset: str  # "mimic-iii" hoặc "mimic-iv"
#     diseases: list[str]
#     procedures: list[str]


# @app.post("/recommend")
# def recommend_medications(data: InputData):
#     """
#     Nhận dataset, bệnh và thủ thuật, trả về danh sách thuốc được đề xuất.
#     """
#     if data.dataset not in DATASET_PATHS:
#         return {"error": "Invalid dataset. Choose 'mimic-iii' or 'mimic-iv'."}

import torch
import dill
from src.models.RAREMed import RAREMed
# import sys
# import os

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# Load trained model
model_path = "/home/nguyenkhanh/Documents/AILAB/RAREMed/RAREMed-Clone/src/log/mimic-iii/RAREMed/log17_/Epoch_9_JA_0.5329_DDI_0.07024.model"
model = RAREMed.load_from_checkpoint(model_path)

# Load patient data (example input)
patient_data = [{"ICD_CODE": ["401.9", "250.00"], "PRO_CODE": ["99.04"], "ATC3": []}]

# Make a prediction
predictions = model.predict(patient_data)

print("Recommended Medications:", predictions)
