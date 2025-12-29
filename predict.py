import torch
from model import HousePriceModel


# Load checkpoint an toàn
checkpoint = torch.load("model.pth", weights_only=False)

# Khởi tạo model và load trọng số đã học
model = HousePriceModel(input_dim=3)
# Load model weights
model.load_state_dict(checkpoint["model_state"])
# Chuyển model về chế độ đánh giá
model.eval()

# Lấy thông tin chuẩn hóa
X_mean = checkpoint["X_mean"]
X_std = checkpoint["X_std"]
y_mean = checkpoint["y_mean"]
y_std = checkpoint["y_std"]

# Tạo một mẫu dữ liệu để dự đoán (1 căn nhà: 120 m² + 3 phòng + cách trung tâm 5 km)
sample = torch.tensor([[120, 3, 5]], dtype=torch.float32)

# Normalize input giống lúc train
sample_norm = (sample - X_mean) / X_std
sample_tensor = torch.tensor(sample_norm, dtype=torch.float32)

with torch.no_grad():  # Tắt gradient khi predict
    # Forward – AI dự đoán
    y_pred_norm = model(sample_tensor)

# Denormalize output → giá thật
price = y_pred_norm.item() * y_std + y_mean

# In kết quả ra dạng số
print(f"Predicted price: {price:,.0f} VND")
