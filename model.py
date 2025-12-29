# Neural Network module
import torch.nn as nn


class HousePriceModel(nn.Module):
    def __init__(self, input_dim):
        """
        Hàm khởi tạo
            input_dim: Số feature đầu vào
        """
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)  # Neural Network với 1 layer (Linear Regression)

    def forward(self, x):
        """
        Luồng dữ liệu đi qua model
        """
        return self.linear(x)
