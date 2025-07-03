from src import config

def calculate_all_metrics(best_features, face_box_orig, face_box_trans):
    """
    Tính toán tất cả các số liệu tỷ lệ khuôn mặt (Ngang và Dọc).
    Hàm này sử dụng logic phân chia theo bên để cho kết quả trực quan và chính xác
    ngay cả khi chỉ phát hiện được một bên lông mày.

    Args:
        best_features (list): Danh sách các đối tượng đặc điểm đã được phát hiện.
        face_box_orig (tuple): Bounding box của khuôn mặt gốc (x1, y1, x2, y2) so với chính nó.
        face_box_trans (tuple): Bounding box của khuôn mặt đã được dịch chuyển so với ảnh đã cắt.

    Returns:
        list: Danh sách các dictionary, mỗi dict chứa thông tin chi tiết về một số liệu.
    """
    x1_o, y1_o, x2_o, y2_o = face_box_orig
    x1_t, y1_t, x2_t, y2_t = face_box_trans
    face_w, face_h = float(x2_o - x1_o), float(y2_o - y1_o)
    
    # Tránh lỗi chia cho 0 nếu phát hiện khuôn mặt không hợp lệ
    if face_w == 0 or face_h == 0:
        print("Cảnh báo: Phát hiện khuôn mặt có chiều rộng hoặc chiều cao bằng 0. Bỏ qua các số liệu.")
        return []

    metrics, m_colors = [], config.UI_CONFIG['colors']['metrics']
    
    # Xác định các đặc điểm chính
    all_eyebrows = sorted([p for p in best_features if p.category.id in config.EYEBROW_CLASS_IDS], key=lambda p: p.bbox.minx)
    nose = next((p for p in best_features if p.category.id in config.NOSE_CLASS_IDS), None)
    
    # --- SỐ LIỆU NGANG (QUY TẮC 1/5) ---
    if all_eyebrows:
        face_center_x = face_w / 2.0
        
        # Phân chia lông mày thành bên trái và phải dựa trên vị trí trung tâm của chúng
        left_eyebrows = sorted([p for p in all_eyebrows if (p.bbox.minx + p.bbox.maxx) / 2 < face_center_x], key=lambda p: p.bbox.minx)
        right_eyebrows = sorted([p for p in all_eyebrows if (p.bbox.minx + p.bbox.maxx) / 2 >= face_center_x], key=lambda p: p.bbox.minx)

        # Tính toán vị trí Y ổn định để vẽ tất cả các số liệu lông mày cho thẳng hàng
        avg_y_center = sum([(p.bbox.miny + p.bbox.maxy) for p in all_eyebrows]) / (2 * len(all_eyebrows))
        draw_y_eb = avg_y_center + y1_t

        # --- Tính toán cho bên TRÁI ---
        if left_eyebrows:
            leftmost_eb_box = left_eyebrows[0].bbox
            
            # L. Outer Face (Từ mép mặt đến mép ngoài lông mày trái)
            px = leftmost_eb_box.minx
            p1, p2 = (x1_t, draw_y_eb), (leftmost_eb_box.minx + x1_t, draw_y_eb)
            metrics.append({'label': "L. Outer Face", 'pixels': px, 'percentage': (px/face_w)*100, 'points': [p1,p2], 'color': m_colors['eyebrow_outer_space'], 'orientation': 'horizontal', 'position': 'above', 'label_text': f"{(px/face_w)*100:.1f}%"})

            # L. Eyebrow Width (Chiều rộng lông mày trái)
            px_w = leftmost_eb_box.maxx - leftmost_eb_box.minx
            p1, p2 = (leftmost_eb_box.minx + x1_t, draw_y_eb), (leftmost_eb_box.maxx + x1_t, draw_y_eb)
            metrics.append({'label': "L. Eyebrow Width", 'pixels': px_w, 'percentage': (px_w/face_w)*100, 'points': [p1,p2], 'color': m_colors['eyebrow_width'], 'orientation': 'horizontal', 'position': 'below', 'label_text': f"{(px_w/face_w)*100:.1f}%"})

        # --- Tính toán cho bên PHẢI ---
        if right_eyebrows:
            rightmost_eb_box = right_eyebrows[-1].bbox
            
            # R. Outer Face (Từ mép ngoài lông mày phải đến mép mặt)
            px = face_w - rightmost_eb_box.maxx
            p1, p2 = (rightmost_eb_box.maxx + x1_t, draw_y_eb), (x2_t, draw_y_eb)
            metrics.append({'label': "R. Outer Face", 'pixels': px, 'percentage': (px/face_w)*100, 'points': [p1,p2], 'color': m_colors['eyebrow_outer_space'], 'orientation': 'horizontal', 'position': 'above', 'label_text': f"{(px/face_w)*100:.1f}%"})
            
            # R. Eyebrow Width (Chiều rộng lông mày phải)
            px_w = rightmost_eb_box.maxx - rightmost_eb_box.minx
            p1, p2 = (rightmost_eb_box.minx + x1_t, draw_y_eb), (rightmost_eb_box.maxx + x1_t, draw_y_eb)
            metrics.append({'label': "R. Eyebrow Width", 'pixels': px_w, 'percentage': (px_w/face_w)*100, 'points': [p1,p2], 'color': m_colors['eyebrow_width'], 'orientation': 'horizontal', 'position': 'below', 'label_text': f"{(px_w/face_w)*100:.1f}%"})

        # --- Tính toán giữa hai lông mày ---
        if left_eyebrows and right_eyebrows:
            inner_left_box = left_eyebrows[-1].bbox
            inner_right_box = right_eyebrows[0].bbox

            # Interocular Space (Khoảng cách giữa hai lông mày)
            px_inter = inner_right_box.minx - inner_left_box.maxx
            if px_inter > 0:
                p1, p2 = (inner_left_box.maxx + x1_t, draw_y_eb), (inner_right_box.minx + x1_t, draw_y_eb)
                metrics.append({'label': "Interocular Space", 'pixels': px_inter, 'percentage': (px_inter/face_w)*100, 'points': [p1,p2], 'color': m_colors['eyebrow_inner_gap'], 'orientation': 'horizontal', 'position': 'above', 'label_text': f"{(px_inter/face_w)*100:.1f}%"})

    # --- SỐ LIỆU NGANG DỰA TRÊN MŨI ---
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

    # --- SỐ LIỆU DỌC (QUY TẮC 1/3) & ĐỐI XỨNG TRỤC ---
    # Yêu cầu có ít nhất 2 lông mày và 1 mũi để đảm bảo tính ổn định
    if len(all_eyebrows) >= 2 and nose:
        l_eb, r_eb = all_eyebrows[0].bbox, all_eyebrows[-1].bbox
        eb_y_avg = (l_eb.miny + l_eb.maxy + r_eb.miny + r_eb.maxy) / 4.0
        nose_bottom_y = nose.bbox.maxy
        nose_center_x = (nose.bbox.minx + nose.bbox.maxx) / 2.0
        
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

        # Số liệu khoảng cách từ lông mày đến trục trung tâm
        eb_line_y = eb_y_avg + y1_t
        left_eb_inner_x = (all_eyebrows[0].bbox.maxx if not left_eyebrows else left_eyebrows[-1].bbox.maxx) + x1_t
        right_eb_inner_x = (all_eyebrows[-1].bbox.minx if not right_eyebrows else right_eyebrows[0].bbox.minx) + x1_t
        
        dist_left = vertical_anchor_x - left_eb_inner_x
        if dist_left > 0:
            metrics.append({'label': 'L. Eyebrow to Axis', 'pixels': dist_left, 'percentage': (dist_left/face_w)*100, 'points': [(left_eb_inner_x, eb_line_y), (vertical_anchor_x, eb_line_y)], 'color': m_colors['eyebrow_to_axis'], 'orientation': 'horizontal', 'position': 'below', 'label_text': f"{(dist_left/face_w)*100:.1f}%"})
        
        dist_right = right_eb_inner_x - vertical_anchor_x
        if dist_right > 0:
            metrics.append({'label': 'R. Eyebrow to Axis', 'pixels': dist_right, 'percentage': (dist_right/face_w)*100, 'points': [(vertical_anchor_x, eb_line_y), (right_eb_inner_x, eb_line_y)], 'color': m_colors['eyebrow_to_axis'], 'orientation': 'horizontal', 'position': 'below', 'label_text': f"{(dist_right/face_w)*100:.1f}%"})

    return metrics

def calculate_harmony_scores(all_metrics):
    """
    Tính toán điểm hài hòa của khuôn mặt dựa trên các tỷ lệ vàng.
    Điểm số được tính bằng cách đo độ lệch so với tỷ lệ lý tưởng.

    Args:
        all_metrics (list): Danh sách các số liệu được trả về từ `calculate_all_metrics`.

    Returns:
        dict: Một dictionary chứa điểm số cho từng khía cạnh hài hòa.
    """
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
            scores[key] = 0.0
            continue
        
        avg_perc = sum(relevant_metrics) / len(relevant_metrics)
        
        # Tính điểm dựa trên độ lệch so với lý tưởng
        # Độ lệch càng nhỏ, điểm càng gần 100
        deviation = abs(avg_perc - data['ideal']) / data['ideal']
        score = max(0, 100 * (1 - deviation))
        scores[key] = round(score, 2)
        
    return scores