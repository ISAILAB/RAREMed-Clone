import torch

print(torch.cuda.device_count())  # Số GPU khả dụng
print(torch.cuda.current_device())  # ID GPU hiện tại
print(torch.cuda.get_device_name(0))  # Tên GPU đầu tiên
