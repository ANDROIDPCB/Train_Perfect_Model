import torch
from DeepFIR.DeepFIR import LstmWin
from torch.serialization import add_safe_globals
# 加载检查点
Tar_Path = 'DeepFIR\\best_model.tar'
model_checkpoint = torch.load(Tar_Path, weights_only=False)  # 现在不会报错
model_static_dict = model_checkpoint["model"]
# 实例化模型（需确保参数与训练时一致）
model = LstmWin()  # 替换为你的实际参数
model.load_state_dict(model_static_dict)  # 加载参数
torch.save(model.state_dict(), 'DeepFIR\\DeepFir.pth')