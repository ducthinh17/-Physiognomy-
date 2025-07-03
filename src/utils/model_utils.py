import torch
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
from collections import defaultdict
from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

from src import config # Sử dụng import tuyệt đối

def load_all_models(device):
    """Tải tất cả các mô hình cần thiết và trả về dưới dạng một dict."""
    print("Đang tải các mô hình...")
    try:
        # Kiểm tra xem các tệp model có tồn tại không
        required_models = {
            "face": config.FACE_MODEL_PATH,
            "feature": config.FEATURE_MODEL_PATH,
            "classifier": config.CLASSIFICATION_MODEL_PATH
        }
        for name, path in required_models.items():
            if not os.path.exists(path):
                raise FileNotFoundError(f"LỖI: Không tìm thấy tệp model '{name}' tại đường dẫn: {path}. Hãy chắc chắn rằng bạn đã tải chúng lên hoặc sử dụng Git LFS.")

        face_detector = YOLO(config.FACE_MODEL_PATH)
        feature_detector = AutoDetectionModel.from_pretrained(
            model_type='yolov8',
            model_path=config.FEATURE_MODEL_PATH,
            confidence_threshold=config.FEATURE_CONF_THRESHOLD,
            device=device
        )
        face_shape_classifier = torch.load(config.CLASSIFICATION_MODEL_PATH, map_location=device)
        face_shape_classifier.to(device).eval()
        print("Tất cả mô hình đã được tải thành công.")
        return {
            "face": face_detector,
            "feature": feature_detector,
            "classifier": face_shape_classifier
        }
    except Exception as e:
        print(f"LỖI NGHIÊM TRỌNG khi tải mô hình: {e}")
        raise e

def preprocess_for_classification(image_pil):
    """Chuẩn bị ảnh khuôn mặt đã cắt cho mô hình phân loại."""
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image_pil).unsqueeze(0)

def get_probabilities(logits):
    """Tính toán xác suất softmax từ logits của mô hình."""
    return F.softmax(logits, dim=1) * 100

def detect_features_in_face(cropped_face_pil, feature_detector):
    """Phát hiện các đặc điểm khuôn mặt bằng suy luận cắt lớp (sliced inference)."""
    result = get_sliced_prediction(
        image=np.array(cropped_face_pil),
        detection_model=feature_detector,
        slice_height=max(256, int(cropped_face_pil.height * 0.6)),
        slice_width=max(256, int(cropped_face_pil.width * 0.6)),
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2,
        verbose=0
    )
    grouped_preds = defaultdict(list)
    for pred in result.object_prediction_list:
        grouped_preds[pred.category.id].append(pred)
    
    final_preds = []
    for class_id, preds in grouped_preds.items():
        preds.sort(key=lambda p: p.score.value, reverse=True)
        limit = 2 if class_id in config.EYEBROW_CLASS_IDS else 1
        final_preds.extend(preds[:limit])
    return final_preds