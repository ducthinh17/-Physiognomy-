import os
import io
import json
import datetime
import torch
from PIL import Image, ImageDraw

from . import config
from .utils.model_utils import (
    preprocess_for_classification, get_probabilities, detect_features_in_face
)
from .utils.metrics_utils import calculate_all_metrics, calculate_harmony_scores
from .utils.drawing_utils import (
    draw_shape_border, draw_metric_on_image, create_professional_report
)

def run_analysis_pipeline(original_image, face_detector, feature_detector, face_shape_classifier, filename="uploaded_image"):
    """
    Hàm pipeline chính để xử lý ảnh, được điều chỉnh cho API.
    Nhận: Đối tượng ảnh PIL và các model đã được tải.
    Trả về: (annotated_img, report_img, master_json_data)
    """
    # Bước 1: Phát hiện khuôn mặt lớn nhất
    face_results = face_detector(original_image, verbose=False, conf=config.FACE_CONF_THRESHOLD)
    if not face_results or not face_results[0].boxes:
        print("Không có khuôn mặt nào được phát hiện.")
        return None, None, None

    all_boxes = sorted(face_results[0].boxes, key=lambda b: (b.xyxy[0][2]-b.xyxy[0][0])*(b.xyxy[0][3]-b.xyxy[0][1]), reverse=True)
    box = all_boxes[0]
    x1_f, y1_f, x2_f, y2_f = map(int, box.xyxy[0])
    
    if (x2_f - x1_f) <= 0 or (y2_f - y1_f) <= 0:
        print("Phát hiện bounding box không hợp lệ.")
        return None, None, None
        
    face_crop_pil = original_image.crop((x1_f, y1_f, x2_f, y2_f))

    # Bước 2: Phân loại hình dạng khuôn mặt
    device = next(face_shape_classifier.parameters()).device
    tensor = preprocess_for_classification(face_crop_pil).to(device)
    with torch.no_grad():
        outputs = face_shape_classifier(tensor)
        probs = get_probabilities(outputs)
        shape_probs = {config.SHAPE_LABELS[i]: probs[0, i].item() for i in range(len(config.SHAPE_LABELS))}
        primary_shape = config.SHAPE_LABELS[torch.max(outputs, 1)[1].item()]

    # Bước 3: Phát hiện đặc điểm khuôn mặt
    best_features = detect_features_in_face(face_crop_pil, feature_detector)
    
    # ... Sao chép và điều chỉnh toàn bộ phần còn lại của hàm `process_image_pipeline` từ câu trả lời trước
    # Logic tính toán, vẽ và tạo báo cáo không thay đổi.
    # Chỉ cần đảm bảo nó trả về 3 giá trị: annotated_image.convert("RGB"), report_image, master_data
    
    # ... (Phần code tính toán và vẽ vời giống hệt như trước) ...
    
    # (Ví dụ phần cuối của hàm)
    margin = int((x2_f - x1_f) * config.UI_CONFIG['crop']['margin_percentage'])
    crop_coords = (max(0, x1_f - margin), max(0, y1_f - margin), min(original_image.width, x2_f + margin), min(original_image.height, y2_f + margin))
    focused_image = original_image.crop(crop_coords)
    annotated_image = focused_image.copy().convert('RGBA')
    draw = ImageDraw.Draw(annotated_image, 'RGBA')
    face_bbox_trans = (x1_f - crop_coords[0], y1_f - crop_coords[1], x2_f - crop_coords[0], y2_f - crop_coords[1])
    face_bbox_local = (0, 0, face_crop_pil.width, face_crop_pil.height)
    all_metrics = calculate_all_metrics(best_features, face_bbox_local, face_bbox_trans)
    harmony_scores = calculate_harmony_scores(all_metrics)

    # (Vẽ vời lên ảnh...)
    features_for_report = [] # populate this
    # ...
    report_image = create_professional_report(shape_probs, features_for_report, all_metrics, harmony_scores)

    # Tạo JSON cuối cùng
    master_data = {
        "metadata": {"reportTitle": "Facial Analysis Report", "version": "v1.0-API", "timestampUTC": datetime.datetime.utcnow().isoformat() + "Z"},
        "sourceImage": {"filename": filename, "resolution": {"width": original_image.width, "height": original_image.height}},
        "analysisResult": {
            "face": {
                "confidence": box.conf.item(),
                "boundingBoxOriginal": {"x1": x1_f, "y1": y1_f, "x2": x2_f, "y2": y2_f},
                "shape": {"primary": primary_shape, "probabilities": {k: round(v, 2) for k, v in sorted(shape_probs.items(), key=lambda i:i[1], reverse=True)}},
                "features": [{"label": config.FEATURE_CLASS_NAMES.get(p.category.id), "confidence": round(p.score.value, 4), "boundingBoxCropped": {"minX": p.bbox.minx, "minY": p.bbox.miny, "maxX": p.bbox.maxx, "maxY": p.bbox.maxy}} for p in best_features],
                "proportionality": {
                    "harmonyScores": {k: round(v, 2) for k, v in harmony_scores.items()},
                    "verticalThirds": {m['label']: f"{m['percentage']:.2f}%" for m in all_metrics if m['orientation']=='vertical'},
                    "horizontalFifths": {m['label']: f"{m['percentage']:.2f}%" for m in all_metrics if m['orientation']=='horizontal'},
                    "rawMetrics": [{"label": m['label'], "pixels": round(m['pixels'],2), "percentage": round(m['percentage'],2), "orientation": m['orientation']} for m in all_metrics]
                }
            }
        }
        # "visualElements" is removed from API response for brevity, URLs are used instead
    }
    
    return annotated_image.convert("RGB"), report_image, master_data