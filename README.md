# Physiognomy Analysis API

API phân tích tướng mặt sử dụng FastAPI và các mô hình AI.

## Cài đặt và Chạy Local

1. Cài đặt dependencies:
```bash
pip install -r requirements.txt
```

2. Chạy ứng dụng:
```bash
uvicorn api.main:app --reload
```

Ứng dụng sẽ chạy tại: http://localhost:8000

## Deploy lên Render

### Cách 1: Sử dụng render.yaml (Recommended)
1. Push code lên GitHub
2. Kết nối repository với Render
3. Render sẽ tự động đọc file `render.yaml` và deploy

### Cách 2: Manual Configuration
1. Tạo Web Service mới trên Render
2. Kết nối GitHub repository
3. Cấu hình:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn api.main:app --host 0.0.0.0 --port $PORT`
   - **Environment Variables**:
     - `PYTHON_VERSION`: `3.11.9`
     - `BASE_URL`: `https://your-app-name.onrender.com`

## Environment Variables cần thiết cho Production

- `PORT`: Port mà ứng dụng sẽ chạy (Render tự động cung cấp)
- `BASE_URL`: URL cơ sở của ứng dụng trên production
- `PYTHON_VERSION`: `3.11.9`

## API Endpoints

- `GET /`: Health check
- `POST /analyze/`: Upload và phân tích ảnh
- `GET /results/{request_id}/{image_type}`: Lấy ảnh kết quả

## Start Commands

### Local Development:
```bash
uvicorn api.main:app --reload
```

### Production (Render):
```bash
uvicorn api.main:app --host 0.0.0.0 --port $PORT
```