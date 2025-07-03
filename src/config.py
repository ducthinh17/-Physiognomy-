import os

# --- Cấu trúc thư mục ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# --- Cấu hình Mô hình ---
FACE_MODEL_PATH = os.path.join(MODELS_DIR, 'only_face.pt')
FEATURE_MODEL_PATH = os.path.join(MODELS_DIR, 'Face_detection.pt')
CLASSIFICATION_MODEL_PATH = os.path.join(MODELS_DIR, 'model_85_face_types.pth')

FACE_CONF_THRESHOLD = 0.5
FEATURE_CONF_THRESHOLD = 0.3

# --- Định nghĩa Lớp ---
FEATURE_CLASS_NAMES = {
    0: "Heart Shape", 1: "Oblong Shape", 2: "Oval Shape", 3: "Round Shape", 4: "Square Shape",
    5: "Upper Lip", 6: "Lower Lip", 7: "Inner Mouth", 8: "Pointed Nose",
    9: "Round Nose", 10: "Curved Eyebrow", 11: "Straight Eyebrow"
}
SHAPE_LABELS = ['Heart', 'Oblong', 'Oval', 'Round', 'Square']
EYEBROW_CLASS_IDS = {10, 11}
NOSE_CLASS_IDS = {8, 9}

# --- Cấu hình Giao diện & Báo cáo ---
UI_CONFIG = {
    'colors': {
        'face_bbox': (230, 230, 230, 255),
        'face_shapes': {
            "Heart": (255, 105, 180, 255), "Oblong": (135, 206, 250, 255),
            "Oval": (60, 179, 113, 255), "Round": (255, 165, 0, 255), "Square": (75, 0, 130, 255),
        },
        'feature_default': (0, 255, 255, 255), 'eyebrows': (0, 220, 220, 255), 'nose': (100, 200, 255, 255),
        'text_on_image': (0, 0, 0, 255), 'text_bg_on_image': (255, 255, 255, 210),
        'metrics': {
            'eyebrow_inner_gap': (0, 200, 150), 'eyebrow_width': (255, 105, 180),
            'eyebrow_outer_space': (218, 112, 214), 'nose_width': (230, 100, 100),
            'nose_side_space': (255, 140, 0), 'height_forehead': (255, 165, 0),
            'height_brow_to_nose': (50, 205, 50), 'height_lower_face': (0, 255, 255),
            'eyebrow_to_axis': (60, 179, 113),
        },
        'grid': (204, 153, 255, 220),
        'report_bg': (40, 42, 54, 255), 'report_text': (255, 255, 255, 255),
        'report_header': (200, 200, 255, 255), 'report_divider': (80, 82, 94, 255)
    },
    'line_widths': {'face': 4, 'face_shape': 5, 'feature': 3, 'metric': 4, 'grid': 2},
    'geometry': {'corner_radius': 15, 'label_corner_radius': 5},
    'crop': {'margin_percentage': 0.25}
}