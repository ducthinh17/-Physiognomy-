import io
import uuid
import datetime
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Depends, HTTPException
from fastapi.responses import StreamingResponse
from typing import Dict

from .schemas import AnalysisResponse
from .dependencies import get_models
from src.processing import run_analysis_pipeline

app = FastAPI(
    title="Physiognomy Analysis API",
    description="Tải ảnh lên để phân tích tỷ lệ và đặc điểm khuôn mặt.",
    version="1.0.0"
)

# Một bộ nhớ đệm đơn giản để lưu trữ ảnh kết quả theo ID request
# Trong môi trường production, bạn có thể dùng Redis hoặc một giải pháp lưu trữ tạm thời khác.
results_cache = {}

@app.get("/", tags=["Health Check"])
def read_root():
    """Endpoint cơ bản để kiểm tra server có đang hoạt động không."""
    return {"status": "ok", "message": "Welcome to the Physiognomy Analysis API!"}

@app.post("/analyze/", response_model=AnalysisResponse, tags=["Analysis"])
async def analyze_face_image(
    file: UploadFile = File(..., description="Tệp ảnh cần phân tích (JPG, PNG)."),
    models: Dict = Depends(get_models)
):
    """
    Endpoint chính để phân tích ảnh:
    1. Nhận một tệp ảnh.
    2. Chạy pipeline phân tích.
    3. Trả về kết quả JSON cùng với URL để truy cập các ảnh kết quả.
    """
    # Đọc nội dung file ảnh
    image_bytes = await file.read()

    # Kiểm tra định dạng file ảnh
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Định dạng tệp không hợp lệ. Chỉ chấp nhận JPG hoặc PNG.")

    try:
        original_image = Image.open(io.BytesIO(image_bytes))
    except Exception:
        raise HTTPException(status_code=400, detail="Không thể xử lý tệp ảnh. Tệp có thể bị hỏng.")

    # Chạy pipeline xử lý
    annotated_img, report_img, master_json_data = run_analysis_pipeline(
        original_image, models["face"], models["feature"], models["classifier"], file.filename
    )

    if not master_json_data:
        raise HTTPException(status_code=500, detail="Phân tích thất bại. Không phát hiện được khuôn mặt hoặc có lỗi xảy ra.")

    # Tạo ID duy nhất cho request này
    request_id = str(uuid.uuid4())

    # Lưu ảnh kết quả vào bộ nhớ đệm
    annotated_buffer = io.BytesIO()
    annotated_img.save(annotated_buffer, format="JPEG")
    annotated_buffer.seek(0)

    report_buffer = io.BytesIO()
    report_img.save(report_buffer, format="JPEG")
    report_buffer.seek(0)
    
    results_cache[request_id] = {
        "annotated": annotated_buffer,
        "report": report_buffer,
        "timestamp": datetime.datetime.now()
    }
    
    # Bổ sung URL truy cập ảnh vào JSON response
    import os
    base_url = os.getenv("BASE_URL", "http://localhost:8000")  # Sử dụng environment variable cho Render
    master_json_data["visuals"] = {
        "annotated_image_url": f"{base_url}/results/{request_id}/annotated",
        "report_image_url": f"{base_url}/results/{request_id}/report"
    }
    
    return master_json_data


@app.get("/results/{request_id}/{image_type}", tags=["Results"])
def get_result_image(request_id: str, image_type: str):
    """
    Endpoint để trả về ảnh đã xử lý (chú thích hoặc báo cáo).
    """
    if request_id not in results_cache:
        raise HTTPException(status_code=404, detail="Không tìm thấy kết quả cho ID này.")
    
    if image_type not in ["annotated", "report"]:
        raise HTTPException(status_code=404, detail="Loại ảnh không hợp lệ. Chỉ chấp nhận 'annotated' hoặc 'report'.")

    image_buffer = results_cache[request_id][image_type]
    image_buffer.seek(0) # Đảm bảo con trỏ ở đầu buffer
    
    return StreamingResponse(io.BytesIO(image_buffer.read()), media_type="image/jpeg")