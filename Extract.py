import cv2
import pytesseract
import numpy as np
import shutil
import os
import fitz 

def setup_tesseract():
    if shutil.which("tesseract"): return
    possible_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Tesseract-OCR\tesseract.exe")
    ]
    for path in possible_paths:
        if os.path.exists(path):
            pytesseract.pytesseract.tesseract_cmd = path
            return
    print("Warning: Tesseract setup failed.")

setup_tesseract()

def yield_document_pages(path):
    """Yields (page_num, image_bgr) for PDF or Image."""
    if path.lower().endswith('.pdf'):
        doc = fitz.open(path)
        if len(doc) == 0: raise ValueError("PDF is empty.")
        for i, page in enumerate(doc):
            pix = page.get_pixmap(dpi=300)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
            if pix.n == 3: img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            elif pix.n == 4: img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            yield i + 1, img
    else:
        img = cv2.imread(path)
        if img is None: raise ValueError(f"Image not found at {path}")
        yield 1, img

def normalize_resolution(img, min_height=1500):
    h, w = img.shape[:2]
    if h < min_height:
        scale = min_height / h
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
    return img

def preprocess_image(img):
    """Returns (gray, thresh)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)
    thresh = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 2)
    return gray, thresh

def get_skew_angle(thresh):
    """Calculates deskew angle from contours."""
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    angles = []
    for cnt in contours:
        if cv2.contourArea(cnt) < 50: continue
        rect = cv2.minAreaRect(cnt)
        w, h = rect[1]
        if w == 0 or h == 0: continue
        if max(w, h) / min(w, h) < 2: continue
        angle = rect[-1]
        angle = -(90 + angle) if angle < -45 else -angle
        if abs(angle) <= 20: angles.append(angle)
        
    if not angles: return 0.0
    median = np.median(angles)
    return 0.0 if abs(median) < 0.5 else median

def rotate_image(image, angle):
    if angle == 0.0: return image
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

def ocr_pass(image, config):
    try:
        data = pytesseract.image_to_data(image, config=config, output_type=pytesseract.Output.DICT)
        full_text = pytesseract.image_to_string(image, config=config)
    except pytesseract.TesseractError:
        return "", 0
    
    confs = [int(c) for i, c in enumerate(data['conf']) if int(data['conf'][i]) != -1 and data['text'][i].strip()]
    return full_text.strip(), (sum(confs) / len(confs)) if confs else 0

def detect_tables(thresh):
    """Detects tables via morph operations on lines."""
    line_min_len = int(thresh.shape[1] / 30)
    hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (line_min_len, 1))
    ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, line_min_len))
    
    vert_lines = cv2.dilate(cv2.erode(thresh, ver_kernel, iterations=1), ver_kernel, iterations=1)
    hor_lines = cv2.dilate(cv2.erode(thresh, hor_kernel, iterations=1), hor_kernel, iterations=1)
    
    mask = cv2.threshold(cv2.addWeighted(vert_lines, 0.5, hor_lines, 0.5, 0.0), 1, 255, cv2.THRESH_BINARY)[1]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    tables = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 50 or h < 50: continue
        
        cells_roi = mask[y:y+h, x:x+w]
        cell_cnts, _ = cv2.findContours(cells_roi, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        valid_cells = []
        for c in cell_cnts:
            cx, cy, cw, ch = cv2.boundingRect(c)
            # Filter noise and the outer box itself
            if cw < 20 or ch < 10 or (cw > 0.9*w and ch > 0.9*h): continue
            valid_cells.append((cx, cy, cw, ch))
            
        if valid_cells: tables.append({"box": (x, y, w, h), "cells_rects": valid_cells})
    return tables

def classify_table_type(grid, cell_spans):
    """
    Classifies table as 'simple' or 'complex' based on structure.
    Complex = High merged cell ratio, multi-row headers, or specific keywords.
    """
    if not grid or not grid[0]: return "simple"
    
    total_cells = sum(len(row) for row in grid)
    merged_cells = sum(1 for span in cell_spans if span['row_span'] > 1)
    
    merged_ratio = merged_cells / max(1, total_cells)
    
    # Heuristic: >10% merged cells -> Complex (Exam schedule)
    if merged_ratio > 0.1: return "complex"
    
    # Heuristic: Check for Header keywords in top 2 rows
    header_keywords = {"date", "time", "subject", "code", "paper", "examiner", "semester"}
    top_text = " ".join(grid[0]).lower() + " " + ( " ".join(grid[1]).lower() if len(grid)>1 else "")
    if any(k in top_text for k in header_keywords) and merged_cells > 0:
        return "complex"
        
    return "simple"

def normalize_table(grid, cell_spans, table_type):
    """
    Applies normalization: Header merging, Smart Fill-Down, Record building.
    """
    if not grid: return {"type": "table", "kind": "empty", "rows": []}
    
    # 1. Header Analysis
    # Simple heuristic: Row 0 is header.
    # If Complex: Check if Row 1 is also header-like? 
    # For now, let's assume Row 0 is primary header.
    headers = grid[0]
    data_start_idx = 1
    
    struct_grid = [row[:] for row in grid] # Copy
    
    if table_type == "complex":
        rows = len(struct_grid)
        cols = len(headers)
        
        # 2. Smart Fill-Down based on Cell Spans
        # We need to map (row, col) to its span info.
        # Spans are stored as list of dicts: {'r':, 'c':, 'row_span':}
        
        # Construct a look-up map for spans
        span_map = {}
        for s in cell_spans:
            span_map[(s['r'], s['c'])] = s['row_span']
            
        for r in range(rows):
            for c in range(cols):
                # Check explicit span from extraction
                row_span = span_map.get((r, c), 1)
                
                # Check implicit "empty cell" inheritance for key columns if span missing?
                # No, strict span-based is safer. 
                # But if OCR missed a span (gap), fall back to "Empty + Header Keyword matches"?
                # Let's rely on explicit span first.
                
                if row_span > 1:
                    val = struct_grid[r][c]
                    for k in range(1, row_span):
                        if r + k < rows:
                            # Only overwrite if currently empty or we trust span absolutely
                            if not struct_grid[r+k][c]:
                                struct_grid[r+k][c] = val
                                
        # 3. Create Logical Records
        # If we have valid headers, create dicts
        records = []
        clean_headers = [h.strip() for h in headers]
        
        # Skip header rows for data
        for i in range(data_start_idx, rows):
            row_data = struct_grid[i]
            # Skip empty rows
            if not any(row_data): continue
            
            rec = {}
            for j, val in enumerate(row_data):
                if j < len(clean_headers):
                    key = clean_headers[j] or f"Col_{j}"
                    rec[key] = val
            records.append(rec)
            
        return {
            "type": "table",
            "table_kind": "complex",
            "headers": clean_headers,
            "rows": struct_grid, # Filled-down rows
            "records": records
        }
        
    else:
        # Simple Table: Return Raw Grid
        return {
            "type": "table",
            "table_kind": "simple",
            "rows": struct_grid
        }

def extract_table_content(gray, table_data):
    """Structure extraction with Dynamic OCR and Normalization."""
    tx, ty, tw, th = table_data['box']
    cell_rects = table_data['cells_rects']
    if not cell_rects: return {"rows": [], "type": "empty"}
    
    heights = [c[3] for c in cell_rects]
    valid_heights = [h for h in heights if h > 5]
    if not valid_heights: return {"rows": [], "type": "empty"}
    std_height = np.median(valid_heights)

    def cluster(coords, threshold):
        coords.sort()
        res = []
        if not coords: return res
        curr = [coords[0]]
        for c in coords[1:]:
            if c - np.mean(curr) <= threshold: curr.append(c)
            else: res.append(np.mean(curr)); curr = [c]
        res.append(np.mean(curr))
        return res

    y_centers = [c[1] + c[3]/2 for c in cell_rects]
    x_centers = [c[0] + c[2]/2 for c in cell_rects]
    
    rows_y = cluster(y_centers, std_height * 0.5)
    cols_x = cluster(x_centers, 20)
    
    num_rows, num_cols = len(rows_y), len(cols_x)
    grid = [["" for _ in range(num_cols)] for _ in range(num_rows)]
    table_roi = gray[ty:ty+th, tx:tx+tw]
    
    cell_spans = [] # Store metadata: {r, c, row_span}
    
    for (cx, cy, cw, ch) in cell_rects:
        r_idx = min(range(num_rows), key=lambda i: abs(rows_y[i] - (cy+ch/2)))
        c_idx = min(range(num_cols), key=lambda i: abs(cols_x[i] - (cx+cw/2)))
        
        y1, y2 = max(0, cy-4), min(table_roi.shape[0], cy+ch+4)
        x1, x2 = max(0, cx-4), min(table_roi.shape[1], cx+cw+4)
        
        cell_crop = table_roi[y1:y2, x1:x2]
        crop_h, crop_w = cell_crop.shape
        
        # Dynamic OCR Selection
        # PSM 7: Single text line (good for numbers, codes)
        # PSM 6: Uniform block (default)
        psm_mode = 6
        if crop_h < 30 or (crop_w / crop_h > 5): 
            psm_mode = 7 # Line-like
            
        txt = pytesseract.image_to_string(cell_crop, config=f'--oem 3 --psm {psm_mode}').strip()
        grid[r_idx][c_idx] = txt
        
        row_span = int(round(ch / std_height))
        cell_spans.append({'r': r_idx, 'c': c_idx, 'row_span': row_span})
        
        # Note: We DON'T fill down here anymore. We delegate to normalize_table.

    # Classification & Normalization
    table_type = classify_table_type(grid, cell_spans)
    normalized_data = normalize_table(grid, cell_spans, table_type)
    
    normalized_data["source"] = "scanned_grid"
    return normalized_data

def process_single_page(image):
    norm = normalize_resolution(image)
    _, temp_thresh = preprocess_image(norm)
    angle = get_skew_angle(temp_thresh)
    
    if angle != 0.0:
        norm = rotate_image(norm, angle)
        
    gray, thresh = preprocess_image(norm)
    tables = detect_tables(thresh)
    
    masked_gray = gray.copy()
    extracted = []
    
    for t in tables:
        extracted.append(extract_table_content(gray, t))
        x, y, w, h = t['box']
        cv2.rectangle(masked_gray, (x, y), (x+w, y+h), (255), -1)
        
    text, conf = ocr_pass(masked_gray, r'--oem 3 --psm 3')
    if conf < 60:
        t2, c2 = ocr_pass(masked_gray, r'--oem 3 --psm 6')
        if c2 > conf: text, conf = t2, c2
            
    return {
        "text": text.strip(),
        "confidence": conf,
        "tables": extracted,
        "status": "Reliable" if conf > 50 else "Unreliable"
    }

def run_ocr_pipeline(path):
    print(f"Processing: {path}")
    try: results = [dict(process_single_page(img), page_num=p) for p, img in yield_document_pages(path)]
    except Exception as e: print(f"Error: {e}"); return None

    if not results: return {'status': 'Failed', 'text': '', 'pages': []}

    full_text = []
    for p in results:
        full_text.append(f"--- Page {p['page_num']} ---\n{p['text']}")
        if p['tables']:
            full_text.append(f"\n[Found {len(p['tables'])} Table(s)]")
            for i, t in enumerate(p['tables']): full_text.append(f"Table {i+1}: {t['rows']}")
            
    avg_conf = sum(p['confidence'] for p in results) / len(results)
    doc_status = "Reliable" if all(p['status'] == "Reliable" for p in results) else "Unreliable/Partial"
    
    return {"text": "\n\n".join(full_text), "avg_confidence": avg_conf, "pages": results, "status": doc_status}

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        res = run_ocr_pipeline(sys.argv[1])
        if res:
            print("="*60 + f"\nSTATUS: {res['status']} (Conf: {res['avg_confidence']:.1f})\n" + "="*60)
            print(res['text'] + "\n" + "="*60)
        else: print("Processing Failed.")
    else: print("Usage: python main.py <file>")
