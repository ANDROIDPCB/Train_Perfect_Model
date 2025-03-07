import torch
from DeepFIR.DeepFIR import LstmWin

# 实例化模型（参数必须与训练时一致）
model = LstmWin()  # 替换为实际参数

# 加载模型参数
model.load_state_dict(torch.load("DeepFIR\DeepFir.pth"))
model.eval()  # 切换到推理模式（关闭Dropout/BatchNorm等训练层）

# 假设输入是(batch_size, sequence_length, input_size)
dummy_input = torch.randn(1, 129, 100)  # 100是占位长度

torch.onnx.export(
    model,
    dummy_input,
    "DeepFIR\\DeepFIR.onnx",
    input_names=["input"],    # 输入名称（自定义，用于部署时识别）
    output_names=["output"],  # 输出名称（自定义）
    dynamic_axes={
        "input": {2: "sequence_length"},  # 动态轴（支持可变长度）
        "output": {2: "sequence_length"}
    },
    opset_version=13  # 根据需求选择ONNX算子版本（推荐≥11）
)



import onnx
import onnxruntime as ort

# 检查模型格式是否正确
onnx_model = onnx.load("DeepFIR\\DeepFIR.onnx")
onnx.checker.check_model(onnx_model)

# 使用ONNX Runtime推理测试
ort_session = ort.InferenceSession("DeepFIR\\DeepFIR.onnx")
output = ort_session.run(
    None, 
    {"input": dummy_input.numpy()}  # 输入需转为numpy数组
)
print(output[0].shape)  # 检查输出形状是否符合预期