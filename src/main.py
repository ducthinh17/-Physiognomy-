
import os
import io
import json
import traceback
import datetime
from zipfile import ZipFile, ZIP_DEFLATED

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
import torch
from PIL import Image

# Import các thành phần từ file face_analyzer trong cùng thư mục src
from .face_analyzer import (
    process_image_pipeline,
    YOLO, AutoDetectionModel,
    FEATURE_CONF_THRESHOLD
)

# --- CONFIG & MODEL LOADING ---

# Lấy đường dẫn tuyệt đối đến thư mục `src` nơi file này đang chạy
current_dir = os.path.dirname(os.path.abspath(__file__))
# Đi ngược lên một cấp (tới thư mục gốc của project), sau đó vào thư mục `models`
MODEL_DIR = os.path.join(current_dir, '..', 'models')

# Tạo đường dẫn đầy đủ tới các file model
FACE_MODEL_PATH = os.path.join(MODEL_DIR, 'only_face.pt')
FEATURE_MODEL_PATH = os.path.join(MODEL_DIR, 'Face_detection.pt')
CLASSIFICATION_MODEL_PATH = os.path.join(MODEL_DIR, 'model_85_face_types.pth')

app = FastAPI(title="Face Analysis API", description="API for facial feature and proportion analysis.")
models = {} # Dùng dictionary để lưu các model đã load

@app.on_event("startup")
def load_models():
    """Tải tất cả các model vào bộ nhớ khi server khởi động."""
    print("--- 🚀 Server starting up, loading models... ---")

    # Kiểm tra xem các file model có tồn tại không
    if not all(os.path.exists(p) for p in [FACE_MODEL_PATH, FEATURE_MODEL_PATH, CLASSIFICATION_MODEL_PATH]):
        print(f"FATAL: One or more model files not found in {MODEL_DIR}. Please check your project structure.")
        return

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        # Load các model và lưu vào dictionary
        models['face_detector'] = YOLO(FACE_MODEL_PATH)
        models['feature_detector'] = AutoDetectionModel.from_pretrained(
            model_type='yolov8', model_path=FEATURE_MODEL_PATH,
            confidence_threshold=FEATURE_CONF_THRESHOLD, device=device
        )
        models['face_shape_classifier'] = torch.load(CLASSIFICATION_MODEL_PATH, map_location=device, weights_only=False)
        models['face_shape_classifier'].to(device).eval()

        print("--- ✅ All models loaded successfully! Server is ready. ---")
    except Exception as e:
        print(f"FATAL ERROR during model loading: {e}")
        traceback.print_exc()

# --- API ENDPOINTS ---

@app.get("/", summary="Health Check")
def read_root():
    """Endpoint để kiểm tra server có đang hoạt động không."""
    return {
        "status": "online",
        "timestamp": datetime.datetime.now().isoformat(),
        "models_loaded": "face_detector" in models
    }

@app.post("/analyze-face/", summary="Analyze a single face image")
async def analyze_face(file: UploadFile = File(..., description="Image file (jpg, png) to analyze.")):
    """
    Nhận một file ảnh, xử lý và trả về một file ZIP chứa 3 file kết quả:
    - `annotated_image.jpg`: Ảnh gốc với các đường kẻ phân tích.
    - `report.jpg`: Ảnh report tổng hợp.
    - `analysis.json`: Dữ liệu phân tích chi tiết dạng JSON.
    """
    if not models:
        raise HTTPException(status_code=503, detail="Models are not loaded. Server is not ready.")

    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image file.")

    image_bytes = await file.read()

    print(f"Processing image: {file.filename}, size: {len(image_bytes)} bytes")

    annotated_img, report_img, master_json_data = process_image_pipeline(
        image_bytes=image_bytes,
        filename=file.filename,
        face_detector=models['face_detector'],
        feature_detector=models['feature_detector'],
        face_shape_classifier=models['face_shape_classifier']
    )

    if annotated_img is None:
        error_detail = master_json_data.get("error", "Failed to process image.")
        print(f"Analysis failed for {file.filename}: {error_detail}")
        raise HTTPException(status_code=422, detail=error_detail)

    # Tạo file ZIP trong bộ nhớ để trả về cho người dùng
    zip_buffer = io.BytesIO()
    with ZipFile(zip_buffer, 'a', ZIP_DEFLATED, False) as zip_file:
        # Lưu ảnh chú thích
        img_byte_arr = io.BytesIO()
        annotated_img.save(img_byte_arr, format='JPEG', quality=95)
        zip_file.writestr('annotated_image.jpg', img_byte_arr.getvalue())

        # Lưu ảnh report
        report_byte_arr = io.BytesIO()
        report_img.save(report_byte_arr, format='JPEG', quality=95)
        zip_file.writestr('report.jpg', report_byte_arr.getvalue())

        # Lưu file JSON
        json_str = json.dumps(master_json_data, indent=4, ensure_ascii=False)
        zip_file.writestr('analysis.json', json_str.encode('utf-8'))

    zip_buffer.seek(0)

    print(f"Successfully processed {file.filename}. Returning zip file.")

    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename=result_{file.filename}.zip"}
    )