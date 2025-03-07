import torch
import torch.nn as nn


class LstmWin(nn.Module):
    """
    Input: [batch size, channels=1, T, n_fft]
    Output: [batch size, T, n_fft]
    """

    def __init__(self):
        super(LstmWin, self).__init__()
        # LSTM
        self.lstm_layer_1 = nn.LSTM(input_size=129, hidden_size=200, num_layers=1, batch_first=True)
        self.lstm_layer_2 = nn.LSTM(input_size=200, hidden_size=200, num_layers=1, batch_first=True)

        # FC
        self.fc_layer = nn.Sequential(
            nn.Linear(200, 129),  # 第一个全连接层
            nn.ReLU(),           # 激活函数
            nn.Linear(129, 129),   # 第二个全连接层
            nn.Sigmoid()
        )

    def forward(self, x):
        self.lstm_layer_1.flatten_parameters()
        self.lstm_layer_2.flatten_parameters() # 展平参数进行训练优化

        # [2, 256, 4, 200] = [2, 1024, 200] => [2, 200, 1024]
        lstm_in = x.permute(0, 2, 1) # [B,F,T] => [B,T,F]
        lstm_out_1, _ = self.lstm_layer_1(lstm_in)  # [2, 200, 1024]
        lstm_out_2, _ = self.lstm_layer_2(lstm_out_1)

        fc_out = self.fc_layer(lstm_out_2) 
        fc_out = fc_out.permute(0, 2, 1) # [B,T,F] => [B,F,T]
        return fc_out


if __name__ == '__main__':
    layer = LstmWin()
    a = torch.rand(32, 129, 200)
    print(layer(a).shape)
