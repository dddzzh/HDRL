import torch
print(torch.cuda.is_available()) # 输出True即GPU生效
print(torch.version.cuda)