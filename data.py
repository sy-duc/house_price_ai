# data.py
import torch
from torch.utils.data import Dataset
import numpy as np


class HousePriceDataset(Dataset):
    """
    Dataset
    """

    def __init__(self, X, y):
        """
        Hàm khởi tạo
        """
        self.X = torch.tensor(X, dtype=torch.float32)              # Chuyển numpy → torch.Tensor
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)  # view(-1, 1) để đảm bảo shape label khớp output, tránh lỗi 

    def __len__(self):
        """
        Số lượng mẫu trong tập dữ liệu
        PyTorch cần biết dataset có bao nhiêu mẫu
        """
        return len(self.X)

    def __getitem__(self, idx):
        """
        Mỗi lần training, PyTorch gọi hàm này để lấy từng sample
        """
        return self.X[idx], self.y[idx]


def generate_house_data(num_samples=1000):
    """
    Tạo dữ liệu giả lập cho bài toán dự đoán giá nhà
    Gồm 1000 mẫu dữ liệu
    Trong thực tế, hàm này sẽ là load CSV / query DB / call API
    """

    # Giữ cho dữ liệu random luôn giống nhau mỗi lần chạy code
    np.random.seed(42)

    # Diện tích (m²): số thực từ 30 -> 200
    area = np.random.uniform(30, 200, num_samples)
    # Số phòng ngủ: số nguyên từ 1 -> 5
    bedrooms = np.random.randint(1, 6, num_samples)
    # Khoảng cách tới trung tâm (km): số thực từ 1 -> 20
    distance = np.random.uniform(1, 20, num_samples)

    # Công thức giá (chỉ giả định để tạo dữ liệu, AI không biết công thức này)
    price = (
        area * 5000                               # Nhà to → giá cao
        + bedrooms * 100000                       # Nhiều phòng → đắt
        - distance * 30000                        # Xa trung tâm → rẻ
        + np.random.normal(0, 50000, num_samples) # Noise (thực tế không hoàn hảo). Không có noise → model học quá dễ, không thực tế
    )

    # Gom feature thành ma trận X - dạng AI có thể dùng do AI không hiểu biến rời rạc
    X = np.column_stack((area, bedrooms, distance))  # Mỗi dòng = 1 căn nhà
    y = price # Nhãn (giá nhà)

    return X, y


def normalize_features(X):
    """
    Chuẩn hóa dữ liệu, đưa về cùng một thang đo
    Tương đương bước preprocessing
    """
    mean = X.mean(axis=0) # Tính trung bình
    std = X.std(axis=0)   # Tính độ lệch chuẩn (các giá trị cách xa trung bình bao nhiêu)

    X_norm = (X - mean) / std
    return X_norm, mean, std


def normalize_target(y):
    mean = y.mean()
    std = y.std()
    y_norm = (y - mean) / std
    return y_norm, mean, std
