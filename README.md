# Demo AI dự đoán giá nhà

## 1️⃣ Phân tích bài toán

- Input: đặc trưng ngôi nhà
- Output: giá nhà (số thực)

- 👉 AI sẽ phải học xu hướng từ dữ liệu:
  - **“Với các đặc trưng này, giá thường nằm ở khoảng nào?”**

## 2️⃣ Dữ liệu giả lập

- Dataset đơn giản nhưng thực tế:

  | Feature  | Ý nghĩa                        |
  | -------- | ------------------------------ |
  | area     | Diện tích (m²)                 |
  | bedrooms | Số phòng                       |
  | distance | Khoảng cách tới trung tâm (km) |

  - 👉 3 feature → 1 giá

## 3️⃣ Structure project

- ```
  house_price/
  │
  ├─ venv/
  ├─ data.py          # Dữ liệu
  ├─ model.py         # Định nghĩa Model
  ├─ train.py         # Học
  ├─ predict.py       # Dự đoán
  └─ requirements.txt
  ```

## 4️⃣ Installation

- ❶ Cài Python chính thức

- ❷ Vào folder project

- ❸ Tạo môi trường ảo

  ```bash
  python -m venv venv
  ```

- ❹ Active môi trường ảo

  ```bash
  venv\Scripts\activate
  ```

- ❺ Cài PyTorch

  ```bash
  pip install torch torchvision torchaudio
  ```

  - Dùng CPU là đủ, chưa cần GPU.

## 5️⃣ Run

- ❶ Chạy TRAIN tạo model dự đoán giá nhà

- ```bash
  python train.py
  ```

  - 👉 Sinh ra file `model.pth`

- ❷ Chạy PREDICT (dự đoán)

- ```bash
  python predict.py
  ```

  - Ở những lần chạy dự đoán giá nhà sau sẽ không cần chạy train nữa (trừ khi muốn train lại).
