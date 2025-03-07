import torch
from DeepFIR_RKNN.DeepFIR_RKNN import LstmWin

# 实例化模型（参数必须与训练时一致）
model = LstmWin()  # 替换为实际参数



# 加载模型参数
model.load_state_dict(torch.load("DeepFIR_RKNN\DeepFIR_RKNN.pth"))
model.eval()  # 切换到推理模式（关闭Dropout/BatchNorm等训练层）

# 假设输入是(batch_size, sequence_length, input_size)
dummy_input = torch.randn(1, 1, 129, 16)  # 固定长度16

torch.onnx.export(
    model,
    dummy_input,
    "DeepFIR_RKNN\\DeepFIR_RKNN.onnx",
    input_names=["input"],    # 输入名称（自定义，用于部署时识别）
    output_names=["output"],  # 输出名称（自定义）
    opset_version=13  # 根据需求选择ONNX算子版本（推荐≥11）
)



import onnx
import onnxruntime as ort

# 检查模型格式是否正确
onnx_model = onnx.load("DeepFIR_RKNN\\DeepFIR_RKNN.onnx")
onnx.checker.check_model(onnx_model)

# 使用ONNX Runtime推理测试
ort_session = ort.InferenceSession("DeepFIR_RKNN\\DeepFIR_RKNN.onnx")
output = ort_session.run(
    None, 
    {"input": dummy_input.numpy()}  # 输入需转为numpy数组
)
print(output[0].shape)  # 检查输出形状是否符合预期