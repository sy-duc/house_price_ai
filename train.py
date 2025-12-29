# train.py
import torch
from torch.utils.data import DataLoader

from data import generate_house_data, normalize_features, normalize_target, HousePriceDataset
from model import HousePriceModel


def train_model():
    # Chuẩn bị dữ liệu
    X, y = generate_house_data(1000)
    X_norm, mean, std = normalize_features(X)
    y_norm, y_mean, y_std = normalize_target(y)

    # Tạo dataset
    dataset = HousePriceDataset(X_norm, y_norm)
    # Tạo DataLoader để dễ dàng lấy batch dữ liệu
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True) # Không học 1000 mẫu cùng lúc mà học từng mini-batch (shuffle=True tránh model học theo thứ tự)

    # Khởi tạo model với 3 feature
    model = HousePriceModel(input_dim=3)

    # Thiết lập hàm mất mát (Loss Function) và optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Số epoch để huấn luyện (1 epoch = học hết dataset 1 lần)
    epochs = 100

    # Training loop
    for epoch in range(epochs):
        total_loss = 0

        # Lặp qua từng batch
        for X_batch, y_batch in dataloader:
            # Forward pass (dự đoán)
            preds = model(X_batch)
            # Tính loss (so sánh với giá thật)
            loss = criterion(preds, y_batch)

            # Zero gradients trước khi backward
            optimizer.zero_grad()
            # Backward pass (tính gradient)
            loss.backward()
            # Weight thay đổi → model thông minh hơn
            optimizer.step()

            total_loss += loss.item()

        # In kết quả: Loss giảm = model học được
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

    # Lưu model và thông tin chuẩn hóa
    torch.save({
        "model_state": model.state_dict(),
        "X_mean": mean,
        "X_std": std,
        "y_mean": y_mean,
        "y_std": y_std
    }, "model.pth")


if __name__ == "__main__":
    train_model()
