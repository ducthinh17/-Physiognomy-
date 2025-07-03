import io
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from src import config

def get_font(size=15):
    """Tải font chữ. Ưu tiên Arial nếu có, nếu không thì dùng font mặc định."""
    try:
        # Font Arial cho hiển thị đẹp hơn
        return ImageFont.truetype("arial.ttf", size)
    except IOError:
        # Fallback về font mặc định của PIL nếu không tìm thấy Arial
        return ImageFont.load_default()

def normalize_color(color_tuple):
    """Chuyển đổi màu RGBA 8-bit (0-255) sang định dạng float (0-1) cho Matplotlib."""
    return tuple(c/255.0 for c in color_tuple)

def draw_shape_border(image_obj, box, label):
    """
    Vẽ một đường viền cách điệu quanh khuôn mặt để thể hiện hình dạng đã phân loại.

    Returns:
        dict: Dữ liệu hình học của đường viền đã vẽ để dùng trong báo cáo JSON.
    """
    x1, y1, x2, y2 = map(int, box)
    color = config.UI_CONFIG['colors']['face_shapes'].get(label, (255, 255, 255, 255))
    width = config.UI_CONFIG['line_widths']['face_shape']
    
    # Tạo một lớp phủ (overlay) trong suốt để vẽ lên, tránh ảnh hưởng ảnh gốc
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
        # Hình trái tim phức tạp hơn, dùng OpenCV để vẽ đa giác
        overlay_np = np.array(overlay)
        pts = np.array([
            (x1, y1 + (y2 - y1) // 4), ((x1 + x2) // 2, y2), (x2, y1 + (y2 - y1) // 4),
            (x1 + (x2 - x1) * 3 // 4, y1), ((x1 + x2) // 2, y1 + (y2 - y1) // 5), (x1 + (x2 - x1) // 4, y1)
        ], np.int32)
        cv2.polylines(overlay_np, [pts], isClosed=True, color=color, thickness=width, lineType=cv2.LINE_AA)
        overlay = Image.fromarray(overlay_np)
        drawing_data.update({'type': 'polyline', 'points': pts.tolist()})

    # Dán lớp phủ lên ảnh chính
    image_obj.paste(overlay, (0, 0), overlay)
    return drawing_data

def draw_metric_on_image(draw, metric):
    """Vẽ một đường kẻ và nhãn cho một số liệu cụ thể lên ảnh."""
    p1, p2, color, label_text = metric['points'][0], metric['points'][1], metric['color'], metric['label_text']
    line_width = config.UI_CONFIG['line_widths']['metric']
    draw.line([p1, p2], fill=color, width=line_width)

    font = get_font()
    try:
        # Cách hiện đại để lấy kích thước bounding box của text
        text_bbox = font.getbbox(label_text)
        text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
    except AttributeError: 
        # Fallback cho các phiên bản PIL cũ hơn
        text_width, text_height = draw.textsize(label_text, font=font)
    
    padding = 5
    # Logic để đặt nhãn ở vị trí hợp lý
    if metric['orientation'] == 'horizontal':
        pos_y = p1[1] - (text_height + 15) if metric.get('position', 'above') == 'above' else p1[1] + 10
        pos = ((p1[0] + p2[0] - text_width) / 2, pos_y)
    else:  # vertical
        pos = (p1[0] + metric.get('text_offset_x', 15), (p1[1] + p2[1] - text_height) / 2)

    bg_coords = [(pos[0] - padding, pos[1] - padding), (pos[0] + text_width + padding, pos[1] + text_height + padding)]
    radius = config.UI_CONFIG['geometry']['label_corner_radius']
    
    # Vẽ nền cho text để dễ đọc
    draw.rounded_rectangle(bg_coords, radius=radius, fill=config.UI_CONFIG['colors']['text_bg_on_image'])
    draw.text(pos, label_text, font=font, fill=config.UI_CONFIG['colors']['text_on_image'])

    return {
        'metric_label': metric['label'],
        'line': {'start': p1, 'end': p2, 'color_rgba': color, 'width': line_width},
        'label': {
            'text': label_text, 'position': pos,
            'background_box': {'coords': bg_coords, 'radius': radius, 'color_rgba': config.UI_CONFIG['colors']['text_bg_on_image']},
            'text_color_rgba': config.UI_CONFIG['colors']['text_on_image']
        }
    }

def generate_shape_bar_chart(shape_probs, width, height):
    """Tạo biểu đồ cột ngang cho xác suất các hình dạng khuôn mặt."""
    if not shape_probs: return Image.new('RGBA', (width, height), config.UI_CONFIG['colors']['report_bg'])

    sorted_probs = dict(sorted(shape_probs.items(), key=lambda item: item[1], reverse=False))
    labels, values = list(sorted_probs.keys()), list(sorted_probs.values())
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(width / 100, height / 100))
    
    bar_colors = [normalize_color(config.UI_CONFIG['colors']['face_shapes'].get(l, (200,200,200,255))) for l in labels]
    bars = ax.barh(labels, values, color=bar_colors, height=0.6)
    
    bg_color = normalize_color(config.UI_CONFIG['colors']['report_bg'])
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color(normalize_color(config.UI_CONFIG['colors']['report_divider']))
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', which='both', length=0)
    ax.set_yticklabels(labels, color='white')
    ax.set_xlim(0, 100)
    ax.set_xlabel('Confidence Score (%)', color='white')
    
    for bar in bars:
        bar_width = bar.get_width()
        ax.text(bar_width + 1, bar.get_y() + bar.get_height() / 2, f'{bar_width:.1f}%', va='center', ha='left', color='white')
    
    plt.tight_layout(pad=1.5)
    # Lưu biểu đồ vào bộ nhớ đệm (in-memory buffer) thay vì file
    buf = io.BytesIO()
    plt.savefig(buf, format='PNG', facecolor=bg_color)
    buf.seek(0)
    chart_image = Image.open(buf)
    plt.close(fig)
    return chart_image

def generate_radar_chart(harmony_scores):
    """Tạo biểu đồ radar cho điểm số hài hòa của khuôn mặt."""
    if not harmony_scores: return Image.new('RGBA', (500, 500), config.UI_CONFIG['colors']['report_bg'])

    labels, scores = list(harmony_scores.keys()), list(harmony_scores.values())
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    
    bg_color = normalize_color(config.UI_CONFIG['colors']['report_bg'])
    ax.set_facecolor(bg_color)
    fig.patch.set_facecolor(bg_color)
    
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    
    # "Đóng" biểu đồ radar lại
    scores += scores[:1]
    angles += angles[:1]
    
    ax.plot(angles, scores, color='#00ffff', linewidth=2, linestyle='solid')
    ax.fill(angles, scores, color='#00ffff', alpha=0.25)
    
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, color='white', size='small')
    ax.spines['polar'].set_color('gray')
    ax.grid(color='gray', linestyle='--', linewidth=0.5)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='PNG', bbox_inches='tight', pad_inches=0.1, facecolor=bg_color)
    buf.seek(0)
    chart_image = Image.open(buf)
    plt.close(fig)
    return chart_image

def create_professional_report(shape_probs, features, all_metrics, harmony_scores):
    """Tạo ảnh báo cáo chuyên nghiệp tổng hợp tất cả các kết quả."""
    W, H = 1200, 1800
    image = Image.new('RGBA', (W, H), config.UI_CONFIG['colors']['report_bg'])
    draw = ImageDraw.Draw(image)
    font_h1 = get_font(22)
    font_h2 = get_font(28)
    font_title = get_font(36)
    
    p, y, ls = 40, 40, 30  # padding, y-cursor, line-spacing

    draw.text((p, y), "Facial Analysis & Proportionality Report", font=font_title, fill=config.UI_CONFIG['colors']['report_text']); y += 60
    draw.line([(p, y - 10), (W - p, y - 10)], fill=config.UI_CONFIG['colors']['report_divider'], width=2); y += 30

    # Phần 1: Đánh giá hình dạng khuôn mặt
    draw.text((p, y), "1. Primary Face Shape Assessment", font=font_h2, fill=config.UI_CONFIG['colors']['report_header']); y += 45
    if shape_probs:
        chart_width, chart_height = W - (2 * p), 350
        shape_chart = generate_shape_bar_chart(shape_probs, chart_width, chart_height)
        image.paste(shape_chart, (p, y), shape_chart)
        y += shape_chart.height + 30
    else:
        draw.text((p + 10, y), "Shape assessment not available.", font=font_h1, fill=config.UI_CONFIG['colors']['report_text']); y += ls

    draw.line([(p, y - 10), (W - p, y - 10)], fill=config.UI_CONFIG['colors']['report_divider'], width=1); y += 20
    
    chart_y_start = y  # Lưu vị trí y để vẽ biểu đồ radar sau

    # Cột bên trái: Các đặc điểm và số liệu
    col1_x = p
    
    draw.text((col1_x, y), "2. Key Feature Detection", font=font_h2, fill=config.UI_CONFIG['colors']['report_header']); y += 45
    draw.text((col1_x + 20, y), "Feature", font=font_h1, fill=config.UI_CONFIG['colors']['report_text'])
    draw.text((col1_x + 300, y), "Confidence", font=font_h1, fill=config.UI_CONFIG['colors']['report_text']); y += 25
    draw.line([(col1_x + 10, y), (col1_x + 550, y)], fill=config.UI_CONFIG['colors']['report_divider'], width=1); y += 15
    if features:
        for f in features:
            draw.text((col1_x + 20, y), f"- {f['label']}", font=font_h1, fill=config.UI_CONFIG['colors']['report_text'])
            draw.text((col1_x + 300, y), f"{f['confidence']:.2f}", font=font_h1, fill=config.UI_CONFIG['colors']['report_text'])
            y += ls
    else:
        draw.text((col1_x + 20, y), "No specific features detected.", font=font_h1, fill=config.UI_CONFIG['colors']['report_text']); y += ls
    y += 30

    draw.text((col1_x, y), "3. Proportionality Analysis", font=font_h2, fill=config.UI_CONFIG['colors']['report_header']); y += 45
    metrics_h = sorted([m for m in all_metrics if m['orientation'] == 'horizontal'], key=lambda x: x['label'])
    metrics_v = sorted([m for m in all_metrics if m['orientation'] == 'vertical'], key=lambda x: x['label'])

    draw.text((col1_x, y), "Horizontal (Rule of Fifths)", font=font_h1, fill=(200, 200, 255)); y += ls
    if metrics_h:
        for m in metrics_h:
            draw.text((p + 20, y), f"- {m['label']}: {m['percentage']:.1f}%", font=font_h1, fill=config.UI_CONFIG['colors']['report_text']); y += ls
    else:
        draw.text((p + 20, y), "Not available.", font=font_h1, fill=config.UI_CONFIG['colors']['report_text']); y += ls
    y += 15

    draw.text((col1_x, y), "Vertical (Rule of Thirds)", font=font_h1, fill=(200, 200, 255)); y += ls
    if metrics_v:
        for m in metrics_v:
            draw.text((p + 20, y), f"- {m['label']}: {m['percentage']:.1f}%", font=font_h1, fill=config.UI_CONFIG['colors']['report_text']); y += ls
    else:
        draw.text((p + 20, y), "Not available.", font=font_h1, fill=config.UI_CONFIG['colors']['report_text']); y += ls

    # Cột bên phải: Biểu đồ Radar điểm hài hòa
    col2_x = W - 550
    draw.text((col2_x - 50, chart_y_start), "4. Facial Harmony Score", font=font_h2, fill=(200,200,255))
    radar_chart = generate_radar_chart(harmony_scores)
    chart_size = (500, 500)
    radar_chart.thumbnail(chart_size, Image.Resampling.LANCZOS)
    image.paste(radar_chart, (col2_x, chart_y_start + 50), radar_chart)
    
    return image.convert("RGB")