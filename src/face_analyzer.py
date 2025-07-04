# src/face_analyzer.py

import os
import io
import json
import datetime
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import numpy as np
import cv2
import traceback

try:
    from ultralytics import YOLO
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction
    import torch
    import torch.nn.functional as F
    import torchvision.transforms as T
except ImportError as e:
    print(f"Import Error: {e}")

# --- CONFIGS & CONSTANTS ---

FACE_CONF_THRESHOLD = 0.5
FEATURE_CONF_THRESHOLD = 0.3

FEATURE_CLASS_NAMES = {
    0: "Heart Shape", 1: "Oblong Shape", 2: "Oval Shape", 3: "Round Shape", 4: "Square Shape",
    5: "Upper Lip", 6: "Lower Lip", 7: "Inner Mouth", 8: "Pointed Nose",
    9: "Round Nose", 10: "Curved Eyebrow", 11: "Straight Eyebrow"
}
SHAPE_LABELS = ['Heart', 'Oblong', 'Oval', 'Round', 'Square']
EYEBROW_CLASS_IDS = {10, 11}
NOSE_CLASS_IDS = {8, 9}

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

# --- HELPER FUNCTIONS (FROM YOUR NOTEBOOK) ---

def normalize_color(color_tuple):
    return tuple(c/255.0 for c in color_tuple)

def get_font():
    try:
        # Font 'arial.ttf' thường không có sẵn trên server Linux.
        # Pillow sẽ tự động dùng font mặc định nếu không tìm thấy.
        return ImageFont.truetype("arial.ttf", 15)
    except IOError:
        print("Arial font not found. Using default font.")
        return ImageFont.load_default()

def preprocess_for_classification(image_pil):
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image_pil).unsqueeze(0)

def get_probabilities(logits):
    return F.softmax(logits, dim=1) * 100

def draw_shape_border(image_obj, box, label):
    x1, y1, x2, y2 = map(int, box)
    color = UI_CONFIG['colors']['face_shapes'].get(label, (255, 255, 255, 255))
    width = UI_CONFIG['line_widths']['face_shape']
    overlay = Image.new('RGBA', image_obj.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    drawing_data = {'label': label, 'color_rgba': color, 'line_width': width}

    if label in ["Oval", "Round"]:
        draw.ellipse([x1, y1, x2, y2], outline=color, width=width)
        drawing_data.update({'type': 'ellipse', 'bounding_box': [x1, y1, x2, y2]})
    elif label in ["Square", "Oblong"]:
        radius = 30
        draw.rounded_rectangle([x1, y1, x2, y2], radius=radius, outline=color, width=width)
        drawing_data.update({'type': 'rounded_rectangle', 'bounding_box': [x1, y1, x2, y2], 'radius': radius})
    elif label == "Heart":
        overlay_np = np.array(overlay)
        pts = np.array([
            (x1, y1 + (y2 - y1) // 4), ((x1 + x2) // 2, y2), (x2, y1 + (y2 - y1) // 4),
            (x1 + (x2 - x1) * 3 // 4, y1), ((x1 + x2) // 2, y1 + (y2 - y1) // 5), (x1 + (x2 - x1) // 4, y1)
        ], np.int32)
        cv2.polylines(overlay_np, [pts], isClosed=True, color=color, thickness=width, lineType=cv2.LINE_AA)
        overlay = Image.fromarray(overlay_np)
        drawing_data.update({'type': 'polyline', 'points': pts.tolist()})

    image_obj.paste(overlay, (0, 0), overlay)
    return drawing_data

def draw_metric_on_image(draw, metric):
    p1, p2, color, label_text = metric['points'][0], metric['points'][1], metric['color'], metric['label_text']
    line_width = UI_CONFIG['line_widths']['metric']
    draw.line([p1, p2], fill=color, width=line_width)

    font = get_font()
    try:
        text_bbox = font.getbbox(label_text)
        text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
    except AttributeError:
        text_width, text_height = font.getsize(label_text)

    padding = 5
    if metric['orientation'] == 'horizontal':
        pos_y = p1[1] - (text_height + 15) if metric.get('position', 'above') == 'above' else p1[1] + 10
        pos = ((p1[0] + p2[0] - text_width) / 2, pos_y)
    else:
        pos = (p1[0] + metric.get('text_offset_x', 15), (p1[1] + p2[1] - text_height) / 2)

    bg_coords = [(pos[0] - padding, pos[1] - padding), (pos[0] + text_width + padding, pos[1] + text_height + padding)]
    radius = UI_CONFIG['geometry']['label_corner_radius']
    draw.rounded_rectangle(bg_coords, radius=radius, fill=UI_CONFIG['colors']['text_bg_on_image'])
    draw.text(pos, label_text, font=font, fill=UI_CONFIG['colors']['text_on_image'])

    return {
        'metric_label': metric['label'],
        'line': {'start': p1, 'end': p2, 'color_rgba': color, 'width': line_width},
        'label': {
            'text': label_text, 'position': pos,
            'background_box': {'coords': bg_coords, 'radius': radius, 'color_rgba': UI_CONFIG['colors']['text_bg_on_image']},
            'text_color_rgba': UI_CONFIG['colors']['text_on_image']
        }
    }

def detect_features_in_face(cropped_face_pil, feature_detector):
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
        limit = 2 if class_id in EYEBROW_CLASS_IDS else 1
        final_preds.extend(preds[:limit])
    return final_preds

def calculate_all_metrics(best_features, face_box_orig, face_box_trans):
    x1_o, y1_o, x2_o, y2_o = face_box_orig
    x1_t, y1_t, x2_t, y2_t = face_box_trans
    face_w, face_h = float(x2_o - x1_o), float(y2_o - y1_o)

    if face_w == 0 or face_h == 0:
        return []

    metrics, m_colors = [], UI_CONFIG['colors']['metrics']

    all_eyebrows = sorted([p for p in best_features if p.category.id in EYEBROW_CLASS_IDS], key=lambda p: p.bbox.minx)
    nose = next((p for p in best_features if p.category.id in NOSE_CLASS_IDS), None)

    if all_eyebrows:
        face_center_x = face_w / 2.0
        left_eyebrows = sorted([p for p in all_eyebrows if (p.bbox.minx + p.bbox.maxx) / 2 < face_center_x], key=lambda p: p.bbox.minx)
        right_eyebrows = sorted([p for p in all_eyebrows if (p.bbox.minx + p.bbox.maxx) / 2 >= face_center_x], key=lambda p: p.bbox.minx)
        avg_y_center = sum([(p.bbox.miny + p.bbox.maxy) for p in all_eyebrows]) / (2 * len(all_eyebrows))
        draw_y_eb = avg_y_center + y1_t

        if left_eyebrows:
            leftmost_eb_box = left_eyebrows[0].bbox
            px = leftmost_eb_box.minx
            p1, p2 = (x1_t, draw_y_eb), (leftmost_eb_box.minx + x1_t, draw_y_eb)
            metrics.append({'label': "L. Outer Face", 'pixels': px, 'percentage': (px/face_w)*100, 'points': [p1,p2], 'color': m_colors['eyebrow_outer_space'], 'orientation': 'horizontal', 'position': 'above', 'label_text': f"{(px/face_w)*100:.1f}%"})
            px_w = leftmost_eb_box.maxx - leftmost_eb_box.minx
            p1, p2 = (leftmost_eb_box.minx + x1_t, draw_y_eb), (leftmost_eb_box.maxx + x1_t, draw_y_eb)
            metrics.append({'label': "L. Eyebrow Width", 'pixels': px_w, 'percentage': (px_w/face_w)*100, 'points': [p1,p2], 'color': m_colors['eyebrow_width'], 'orientation': 'horizontal', 'position': 'below', 'label_text': f"{(px_w/face_w)*100:.1f}%"})

        if right_eyebrows:
            rightmost_eb_box = right_eyebrows[-1].bbox
            px = face_w - rightmost_eb_box.maxx
            p1, p2 = (rightmost_eb_box.maxx + x1_t, draw_y_eb), (x2_t, draw_y_eb)
            metrics.append({'label': "R. Outer Face", 'pixels': px, 'percentage': (px/face_w)*100, 'points': [p1,p2], 'color': m_colors['eyebrow_outer_space'], 'orientation': 'horizontal', 'position': 'above', 'label_text': f"{(px/face_w)*100:.1f}%"})
            px_w = rightmost_eb_box.maxx - rightmost_eb_box.minx
            p1, p2 = (rightmost_eb_box.minx + x1_t, draw_y_eb), (rightmost_eb_box.maxx + x1_t, draw_y_eb)
            metrics.append({'label': "R. Eyebrow Width", 'pixels': px_w, 'percentage': (px_w/face_w)*100, 'points': [p1,p2], 'color': m_colors['eyebrow_width'], 'orientation': 'horizontal', 'position': 'below', 'label_text': f"{(px_w/face_w)*100:.1f}%"})

        if left_eyebrows and right_eyebrows:
            inner_left_box, inner_right_box = left_eyebrows[-1].bbox, right_eyebrows[0].bbox
            px_inter = inner_right_box.minx - inner_left_box.maxx
            if px_inter > 0:
                p1, p2 = (inner_left_box.maxx + x1_t, draw_y_eb), (inner_right_box.minx + x1_t, draw_y_eb)
                metrics.append({'label': "Interocular Space", 'pixels': px_inter, 'percentage': (px_inter/face_w)*100, 'points': [p1,p2], 'color': m_colors['eyebrow_inner_gap'], 'orientation': 'horizontal', 'position': 'above', 'label_text': f"{(px_inter/face_w)*100:.1f}%"})

    if nose:
        y_nose = ((nose.bbox.miny + nose.bbox.maxy) / 2) + y1_t
        nose_data = [
            ("L. Nose-to-Cheek", nose.bbox.minx, (x1_t, y_nose), (nose.bbox.minx + x1_t, y_nose), 'nose_side_space', 'above'),
            ("Nose Width", nose.bbox.maxx - nose.bbox.minx, (nose.bbox.minx + x1_t, y_nose), (nose.bbox.maxx + x1_t, y_nose), 'nose_width', 'below'),
            ("R. Nose-to-Cheek", face_w - nose.bbox.maxx, (nose.bbox.maxx + x1_t, y_nose), (x2_t, y_nose), 'nose_side_space', 'above')
        ]
        for lbl, px, p1, p2, c, pos in nose_data:
            if px > 0:
                metrics.append({'label': lbl, 'pixels': px, 'percentage': (px/face_w)*100, 'points': [p1,p2], 'color': m_colors[c], 'orientation': 'horizontal', 'position': pos, 'label_text': f"{(px/face_w)*100:.1f}%"})

    if len(all_eyebrows) >= 2 and nose:
        l_eb, r_eb = all_eyebrows[0].bbox, all_eyebrows[-1].bbox
        eb_y_avg = (l_eb.miny + l_eb.maxy + r_eb.miny + r_eb.maxy) / 4
        nose_bottom_y = nose.bbox.maxy
        nose_center_x = (nose.bbox.minx + nose.bbox.maxx) / 2
        vertical_anchor_x = nose_center_x + x1_t
        p_y1, p_y2, p_y3 = y1_t, eb_y_avg + y1_t, nose_bottom_y + y1_t

        v_data = [
            ("Forehead Height", eb_y_avg, (vertical_anchor_x, p_y1), (vertical_anchor_x, p_y2), 'height_forehead'),
            ("Midface Height", nose_bottom_y - eb_y_avg, (vertical_anchor_x, p_y2), (vertical_anchor_x, p_y3), 'height_brow_to_nose'),
            ("Lower Face Height", face_h - nose_bottom_y, (vertical_anchor_x, p_y3), (vertical_anchor_x, y2_t), 'height_lower_face')
        ]
        for lbl, px, p1, p2, c in v_data:
            if px > 0:
                metrics.append({'label': lbl, 'pixels': px, 'percentage': (px/face_h)*100, 'points': [p1,p2], 'color': m_colors[c], 'orientation': 'vertical', 'label_text': f"{(px/face_h)*100:.1f}%"})

        eb_line_y = eb_y_avg + y1_t
        left_eb_inner_x, right_eb_inner_x = l_eb.maxx + x1_t, r_eb.minx + x1_t
        dist_left = vertical_anchor_x - left_eb_inner_x
        if dist_left > 0:
            metrics.append({'label': 'L. Eyebrow to Axis', 'pixels': dist_left, 'percentage': (dist_left/face_w)*100, 'points': [(left_eb_inner_x, eb_line_y), (vertical_anchor_x, eb_line_y)], 'color': m_colors['eyebrow_to_axis'], 'orientation': 'horizontal', 'position': 'below', 'label_text': f"{(dist_left/face_w)*100:.1f}%"})
        dist_right = right_eb_inner_x - vertical_anchor_x
        if dist_right > 0:
            metrics.append({'label': 'R. Eyebrow to Axis', 'pixels': dist_right, 'percentage': (dist_right/face_w)*100, 'points': [(vertical_anchor_x, eb_line_y), (right_eb_inner_x, eb_line_y)], 'color': m_colors['eyebrow_to_axis'], 'orientation': 'horizontal', 'position': 'below', 'label_text': f"{(dist_right/face_w)*100:.1f}%"})

    return metrics

def calculate_harmony_scores(all_metrics):
    axes = {
        'Outer Face Balance': {'ideal': 20.0, 'labels': ['L. Outer Face', 'R. Outer Face']},
        'Eyebrow Balance': {'ideal': 20.0, 'labels': ['L. Eyebrow Width', 'R. Eyebrow Width']},
        'Interocular Width': {'ideal': 20.0, 'labels': ['Interocular Space']},
        'Nose Width': {'ideal': 20.0, 'labels': ['Nose Width']},
        'Forehead Height': {'ideal': 33.3, 'labels': ['Forehead Height']},
        'Midface Height': {'ideal': 33.3, 'labels': ['Midface Height']},
        'Lower Face Height': {'ideal': 33.3, 'labels': ['Lower Face Height']}
    }
    scores = {}
    for key, data in axes.items():
        relevant_metrics = [m['percentage'] for m in all_metrics if m['label'] in data['labels']]
        if not relevant_metrics:
            scores[key] = 0
            continue
        avg_perc = sum(relevant_metrics) / len(relevant_metrics)
        deviation = abs(avg_perc - data['ideal']) / data['ideal']
        scores[key] = max(0, 100 * (1 - deviation))
    return scores

def generate_shape_bar_chart(shape_probs, width, height):
    sorted_probs = dict(sorted(shape_probs.items(), key=lambda item: item[1], reverse=False))
    labels, values = list(sorted_probs.keys()), list(sorted_probs.values())
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(width / 100, height / 100))
    bar_colors = [normalize_color(UI_CONFIG['colors']['face_shapes'].get(l, (200,200,200,255))) for l in labels]
    bars = ax.barh(labels, values, color=bar_colors, height=0.6)
    bg_color = normalize_color(UI_CONFIG['colors']['report_bg'])
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False); ax.spines['bottom'].set_color(normalize_color(UI_CONFIG['colors']['report_divider']))
    ax.tick_params(axis='x', colors='white'); ax.tick_params(axis='y', which='both', length=0)
    ax.set_yticklabels(labels, color='white'); ax.set_xlim(0, 100)
    ax.set_xlabel('Confidence Score (%)', color='white')
    for bar in bars:
        bar_width = bar.get_width()
        ax.text(bar_width + 1, bar.get_y() + bar.get_height() / 2, f'{bar_width:.1f}%', va='center', ha='left', color='white')
    plt.tight_layout(pad=1.5)
    buf = io.BytesIO()
    plt.savefig(buf, format='PNG', facecolor=bg_color); buf.seek(0)
    chart_image = Image.open(buf); plt.close(fig)
    return chart_image

def generate_radar_chart(harmony_scores):
    labels, scores = list(harmony_scores.keys()), list(harmony_scores.values())
    if not labels:
        return Image.new('RGBA', (500, 500), UI_CONFIG['colors']['report_bg'])

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    bg_color = normalize_color(UI_CONFIG['colors']['report_bg'])
    ax.set_facecolor(bg_color); fig.patch.set_facecolor(bg_color)
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    scores += scores[:1]; angles += angles[:1]
    ax.plot(angles, scores, color='#00ffff', linewidth=2, linestyle='solid')
    ax.fill(angles, scores, color='#00ffff', alpha=0.25)
    ax.set_yticklabels([]); ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, color='white', size='small')
    ax.spines['polar'].set_color('gray'); ax.grid(color='gray', linestyle='--', linewidth=0.5)
    buf = io.BytesIO()
    plt.savefig(buf, format='PNG', bbox_inches='tight', pad_inches=0.1, facecolor=bg_color)
    buf.seek(0); chart_image = Image.open(buf); plt.close(fig)
    return chart_image

def create_professional_report(shape_probs, features, all_metrics, harmony_scores):
    W, H = 1200, 1800
    image = Image.new('RGBA', (W, H), UI_CONFIG['colors']['report_bg'])
    draw = ImageDraw.Draw(image)
    font_h1, font_h2 = get_font(), get_font()

    p, y, ls = 40, 40, 25
    c1, c2 = p + 20, p + 300

    draw.text((p,y), "Facial Analysis & Proportionality Report", font=font_h1, fill=UI_CONFIG['colors']['report_text']); y += 40
    draw.line([(p,y-10),(W-p,y-10)], fill=UI_CONFIG['colors']['report_divider'], width=2); y += 30
    draw.text((p,y), "1. Primary Face Shape Assessment", font=font_h2, fill=UI_CONFIG['colors']['report_header']); y += 30
    if shape_probs:
        chart_width, chart_height = W - (2 * p), 350
        shape_chart = generate_shape_bar_chart(shape_probs, chart_width, chart_height)
        image.paste(shape_chart, (p, y), shape_chart); y += shape_chart.height + 20
    else:
        draw.text((p+10, y), "Shape assessment not available.", font=font_h1, fill=UI_CONFIG['colors']['report_text']); y += ls

    draw.text((p,y), "2. Key Feature Detection", font=font_h2, fill=UI_CONFIG['colors']['report_header']); y+=30
    draw.text((c1,y), "Feature", font=font_h1, fill=UI_CONFIG['colors']['report_text']); draw.text((c2,y), "Confidence", font=font_h1, fill=UI_CONFIG['colors']['report_text']); y+=20
    draw.line([(p+10, y), (W-p-200, y)], fill=UI_CONFIG['colors']['report_divider'], width=1); y += 15
    if features:
        for f in features:
            draw.rectangle([p + 15, y, p + 25, y + 10], fill=f['color'])
            draw.text((c1,y), f"- {f['label']}", font=font_h1, fill=UI_CONFIG['colors']['report_text'])
            draw.text((c2,y), f"{f['confidence']:.2f}", font=font_h1, fill=UI_CONFIG['colors']['report_text']); y += ls
    else:
        draw.text((p+10, y), "No specific features detected.", font=font_h1, fill=UI_CONFIG['colors']['report_text']); y += ls
    y += 30

    chart_y_start = y
    draw.text((p,y), "3. Proportionality Analysis", font=font_h2, fill=UI_CONFIG['colors']['report_header']); y += 30
    metrics_h = sorted([m for m in all_metrics if m['orientation']=='horizontal'], key=lambda x: x['label'])
    metrics_v = sorted([m for m in all_metrics if m['orientation']=='vertical'], key=lambda x: x['label'])
    draw.text((p,y), "Horizontal Proportions (Rule of Fifths)", font=font_h1, fill=(200,200,255)); y += ls
    if metrics_h:
        for m in metrics_h: draw.text((p+10,y),f"  - {m['label']}: {m['percentage']:.1f}%", font=font_h1, fill=UI_CONFIG['colors']['report_text']); y+=ls
    else:
        draw.text((p+10, y), "Not available (requires eyebrows/nose).", font=font_h1, fill=UI_CONFIG['colors']['report_text']); y += ls
    y += 15
    draw.text((p,y), "Vertical Proportions (Rule of Thirds)", font=font_h1, fill=(200,200,255)); y+=ls
    if metrics_v:
        for m in metrics_v: draw.text((p+10,y),f"  - {m['label']}: {m['percentage']:.1f}%", font=font_h1, fill=UI_CONFIG['colors']['report_text']); y+=ls
    else:
        draw.text((p+10, y), "Not available (requires eyebrows and nose).", font=font_h1, fill=UI_CONFIG['colors']['report_text']); y += ls

    radar_chart = generate_radar_chart(harmony_scores); chart_size = (550, 550)
    radar_chart.thumbnail(chart_size, Image.Resampling.LANCZOS)
    chart_x, chart_y = 600, chart_y_start + 40
    draw.text((chart_x + 120, chart_y_start + 10), "Facial Harmony Score", font=font_h2, fill=(200,200,255))
    image.paste(radar_chart, (chart_x, chart_y), radar_chart)
    return image.convert("RGB")

def generate_master_json_output(analysis_data):
    return json.dumps(analysis_data, indent=4)

# --- MAIN PROCESSING PIPELINE ---

def process_image_pipeline(image_bytes: bytes, filename: str, face_detector, feature_detector, face_shape_classifier):
    try:
        original_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        print(f"Error reading image bytes: {e}")
        return None, None, {"error": "Could not read image file."}

    # Step 1: Detect face
    face_results = face_detector(original_image, verbose=False, conf=FACE_CONF_THRESHOLD)
    if not face_results or not face_results[0].boxes:
        return None, None, {"error": "No faces detected in the image."}
    box = sorted(face_results[0].boxes, key=lambda b: (b.xyxy[0][2]-b.xyxy[0][0])*(b.xyxy[0][3]-b.xyxy[0][1]), reverse=True)[0]
    x1_f, y1_f, x2_f, y2_f = map(int, box.xyxy[0])

    if (x2_f - x1_f) <= 0 or (y2_f - y1_f) <= 0:
        return None, None, {"error": "Invalid face bounding box detected."}
    face_crop_pil = original_image.crop((x1_f, y1_f, x2_f, y2_f))

    # Step 2: Classify shape
    device = next(face_shape_classifier.parameters()).device
    tensor = preprocess_for_classification(face_crop_pil).to(device)
    with torch.no_grad():
        outputs = face_shape_classifier(tensor)
        probs = get_probabilities(outputs)
        shape_probs = {SHAPE_LABELS[i]: probs[0, i].item() for i in range(len(SHAPE_LABELS))}
        primary_shape = SHAPE_LABELS[torch.max(outputs, 1)[1].item()]

    # Step 3: Detect features
    best_features = detect_features_in_face(face_crop_pil, feature_detector)

    # Step 4: Prepare for drawing
    margin = int((x2_f - x1_f) * UI_CONFIG['crop']['margin_percentage'])
    crop_coords = (max(0, x1_f - margin), max(0, y1_f - margin), min(original_image.width, x2_f + margin), min(original_image.height, y2_f + margin))
    focused_image = original_image.crop(crop_coords)
    annotated_image = focused_image.copy().convert('RGBA')
    draw = ImageDraw.Draw(annotated_image, 'RGBA')

    face_bbox_trans = (x1_f - crop_coords[0], y1_f - crop_coords[1], x2_f - crop_coords[0], y2_f - crop_coords[1])
    face_bbox_local = (0, 0, face_crop_pil.width, face_crop_pil.height)

    # Step 5: Calculate metrics
    all_metrics = calculate_all_metrics(best_features, face_bbox_local, face_bbox_trans)
    harmony_scores = calculate_harmony_scores(all_metrics)

    # Step 6: Draw visuals
    visual_elements = {'shapes': [], 'features': [], 'metrics': [], 'gridlines': []}
    visual_elements['shapes'].append(draw_shape_border(annotated_image, face_bbox_trans, primary_shape))
    draw.rounded_rectangle(face_bbox_trans, radius=UI_CONFIG['geometry']['corner_radius'], outline=UI_CONFIG['colors']['face_bbox'], width=UI_CONFIG['line_widths']['face'])
    visual_elements['shapes'].append({'label': 'face_detection_box', 'type': 'rounded_rectangle', 'bounding_box': face_bbox_trans, 'radius': UI_CONFIG['geometry']['corner_radius'], 'color_rgba': UI_CONFIG['colors']['face_bbox'], 'line_width': UI_CONFIG['line_widths']['face']})

    features_for_report = []
    for p in best_features:
        cid = p.category.id
        color_key = 'eyebrows' if cid in EYEBROW_CLASS_IDS else 'nose' if cid in NOSE_CLASS_IDS else 'feature_default'
        color = UI_CONFIG['colors'][color_key]
        features_for_report.append({'label': FEATURE_CLASS_NAMES.get(cid, "Unknown"), 'confidence': p.score.value, 'color': color})
        trans_bbox = (p.bbox.minx + face_bbox_trans[0], p.bbox.miny + face_bbox_trans[1], p.bbox.maxx + face_bbox_trans[0], p.bbox.maxy + face_bbox_trans[1])
        draw.rounded_rectangle(trans_bbox, radius=8, outline=color, width=UI_CONFIG['line_widths']['feature'])
        visual_elements['features'].append({'label': FEATURE_CLASS_NAMES.get(cid), 'type': 'rounded_rectangle', 'bounding_box': trans_bbox, 'radius': 8, 'color_rgba': color, 'line_width': UI_CONFIG['line_widths']['feature']})

    for metric in all_metrics:
        visual_elements['metrics'].append(draw_metric_on_image(draw, metric))

    fifth_width = (face_bbox_trans[2] - face_bbox_trans[0]) / 5.0
    for i in range(1, 5):
        line_x = face_bbox_trans[0] + i * fifth_width
        p1, p2 = (line_x, face_bbox_trans[1]), (line_x, face_bbox_trans[3])
        draw.line([p1, p2], fill=UI_CONFIG['colors']['grid'], width=UI_CONFIG['line_widths']['grid'])
        visual_elements['gridlines'].append({'label': f'fifth_line_{i}', 'start': p1, 'end': p2, 'color_rgba': UI_CONFIG['colors']['grid'], 'width': UI_CONFIG['line_widths']['grid']})

    # Step 7: Generate final outputs
    report_image = create_professional_report(shape_probs, features_for_report, all_metrics, harmony_scores)
    master_data = {
        "metadata": {"reportTitle": "Facial Analysis Report", "version": "v23.2", "timestampUTC": datetime.datetime.utcnow().isoformat() + "Z"},
        "sourceImage": {"filename": filename, "resolution": {"width": original_image.width, "height": original_image.height}},
        "analysisResult": {
            "face": {
                "confidence": box.conf.item(),
                "boundingBoxOriginal": {"x1": x1_f, "y1": y1_f, "x2": x2_f, "y2": y2_f},
                "shape": {"primary": primary_shape, "probabilities": {k: round(v, 2) for k, v in sorted(shape_probs.items(), key=lambda i:i[1], reverse=True)}},
                "features": [{"label": FEATURE_CLASS_NAMES.get(p.category.id), "confidence": round(p.score.value, 4), "boundingBoxCropped": {"minX": p.bbox.minx, "minY": p.bbox.miny, "maxX": p.bbox.maxx, "maxY": p.bbox.maxy}} for p in best_features],
                "proportionality": {
                    "harmonyScores": {k: round(v, 2) for k, v in harmony_scores.items()},
                    "verticalThirds": {m['label']: f"{m['percentage']:.2f}%" for m in all_metrics if m['orientation']=='vertical'},
                    "horizontalFifths": {m['label']: f"{m['percentage']:.2f}%" for m in all_metrics if m['orientation']=='horizontal'},
                    "rawMetrics": [{"label": m['label'], "pixels": round(m['pixels'],2), "percentage": round(m['percentage'],2), "orientation": m['orientation']} for m in all_metrics]
                }
            }
        },
        "visualElements": visual_elements
    }

    return annotated_image.convert("RGB"), report_image, master_datas