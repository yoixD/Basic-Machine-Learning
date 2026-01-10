# BASIC MACHINE LEARNING

## Mô tả
Dự án triển khai các thuật toán Machine Learning cơ bản sử dụng Python để giải quyết bài toán dự đoán lương dựa trên số năm kinh nghiệm.

Chương trình hỗ trợ 2 bài toán:
1. **Linear Regression**: Hồi quy tuyến tính (Dự đoán giá trị thực).
2. **Logistic Regression**: Hồi quy Logistic (Phân loại).

## Cấu trúc thư mục
```text
├── BSM.py                # Mã nguồn chính (Chạy dự đoán)
├── dataset.csv           # Dữ liệu đầu vào
└── history_training.csv  # Lưu trữ lịch sử kết quả train
```
## Tính năng
- Preprocessing: Tiền xử lý dữ liệu đầu vào.
-Standard Scaler: Chuẩn hoá dữ liệu để tối ưu hoá quá trình huấn luyện mô hình.
-Tracking: Tự động lưu log kết quả vào file CSV.

## Yêu cầu
-Python 3.x
-NumPy
-Pandas
-Matplotlib
-Scikit_learn
-Os

## Chạy chương trình
```bash
python BSM.py
```
## Kết quả
<img width="632" height="478" alt="Screenshot 2026-01-09 at 21 49 44" src="https://github.com/user-attachments/assets/82600937-7f81-4a2e-b8ca-a40e01ec9a2f" />
<img width="630" height="468" alt="Screenshot 2026-01-09 at 21 50 01" src="https://github.com/user-attachments/assets/c9516257-de9f-4788-8955-335a792087be" />
