import torch
from src.utils.model_utils import load_all_models

# Biến global để lưu trữ các model đã được tải
models = None

def get_models():
    """
    Dependency function để cung cấp các model đã được tải.
    Tải model nếu chúng chưa được tải.
    """
    global models
    if models is None:
        print("Lần đầu khởi động: Đang tải model vào bộ nhớ...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Sử dụng thiết bị: {device}")
        models = load_all_models(device=device)
        print("Model đã sẵn sàng.")
    return models