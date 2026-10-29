import base64
import io
import re
from typing import List, Optional
import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from paddleocr import PaddleOCR

app = FastAPI(title="CRMS Self-Hosted Broadsheet Ad OCR & Measurement API")

# Initialize PaddleOCR once in memory
ocr_engine = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)

class ScanRequest(BaseModel):
    publication: Optional[str] = "THE NEW INDIAN EXPRESS"
    page_number: Optional[int] = 1
    mime_type: Optional[str] = "image/jpeg"
    image_base64: str

CITIES = [
    "BENGALURU", "BANGALORE", "MYSURU", "MYSORE", "CHENNAI", "COIMBATORE",
    "HYDERABAD", "MUMBAI", "DELHI", "NEW DELHI", "KOLKATA", "KOCHI",
    "THIRUVANANTHAPURAM", "MADURAI", "TIRUCHIRAPPALLI", "SALEM", "ERODE",
    "TIRUPPUR", "MANGALURU", "MANGALORE", "HUBBALLI", "BELAGAVI", "SHIVAMOGGA",
    "DAVANAGERE", "BALLARI", "TUMAKURU", "UDUPI", "HASSAN", "MANDYA",
    "VIJAYAWADA", "VISAKHAPATNAM", "TIRUPATI", "GUNTUR", "WARANGAL", "PUNE",
    "AHMEDABAD", "NOIDA", "GURUGRAM", "GURGAON", "FARIDABAD", "CHANDIGARH",
    "JAIPUR", "LUCKNOW", "BHOPAL", "INDORE", "PATNA", "BHUBANESWAR", "GUWAHATI"
]

 EDITORIAL_MARKERS = [
    "EXPRESS NEWS SERVICE", "STAFF REPORTER", "OUR CORRESPONDENT", "PTI", "ANI",
    "REUTERS", "ASSOCIATED PRESS", "CONTINUED ON PAGE", "TURN TO PAGE", "BYLINE",
    "PHOTO:", "EXPRESS PHOTO"
]

AD_MARKERS = [
    "LIMITED", "LTD", "PVT", "PRIVATE", "CORPORATION", "GOVERNMENT", "GOVT",
    "MINISTRY", "DEPARTMENT", "AUTHORITY", "BOARD", "UNIVERSITY", "COLLEGE",
    "SCHOOL", "INSTITUTE", "HOSPITAL", "BANK", "JEWELLERS", "JEWELLERY",
    "GOLD", "DIAMONDS", "SILKS", "TEXTILES", "MOTORS", "AUTOMOBILES",
    "BUILDERS", "DEVELOPERS", "REALTY", "INFRA", "CONSTRUCTIONS", "TENDER",
    "E-TENDER", "NOTICE", "PUBLIC NOTICE", "APPOINTMENT", "WALK-IN", "WANTED",
    "VACANCY", "CAREERS", "ADMISSIONS", "OPEN", "OFFER", "DISCOUNT", "SALE",
    "CALL:", "PH:", "MOB:", "TEL:", "WWW.", ".COM", ".IN", ".ORG", "GSTIN",
    "CIN:", "RERA", "TOLL FREE", "SCAN QR"
]

def classify_ad_text(lines: List[str], w_norm: int, h_norm: int) -> Optional[dict]:
    if not lines:
        return None
    full_text = " ".join(lines)
    upper_text = full_text.upper()

    # Reject obvious editorial stories if long body text with news agency markers and no ad cues
    editorial_hits = sum(1 for m in EDITORIAL_MARKERS if m in upper_text)
    ad_hits = sum(1 for m in AD_MARKERS if m in upper_text)
    if editorial_hits >= 1 and ad_hits == 0:
        return None
    if len(lines) > 35 and ad_hits == 0 and (w_norm * h_norm) < 600000:
        return None

    # Determine Category
    category = "Display"
    if any(k in upper_text for k in ["TENDER", "E-TENDER", "NIT ", "QUOTATION", "BID ", "PROCUREMENT", "CORRIGENDUM"]):
        category = "Tenders"
    elif any(k in upper_text for k in ["PUBLIC NOTICE", "LEGAL NOTICE", "POSSESSION NOTICE", "DEMAND NOTICE", "AUCTION NOTICE", "FORM NO", "BEFORE THE"]):
        category = "Notices"
    elif any(k in upper_text for k in ["WALK-IN", "RECRUITMENT", "VACANCY", "VACANCIES", "APPLICATIONS ARE INVITED", "REQUIRED", "FACULTY", "CAREERS"]):
        category = "Appointment"

    # Determine Vertical
    vertical = "Commercial"
    if any(k in upper_text for k in ["GOVERNMENT", "GOVT", "MINISTRY", "DEPARTMENT", "CORPORATION", "MUNICIPAL", "PANCHAYAT", "RAILWAY", "AUTHORITY", "BESCOM", "BWSSB", "BBMP", "BDA"]):
        vertical = "Government"
    elif any(k in upper_text for k in ["UNIVERSITY", "COLLEGE", "SCHOOL", "ACADEMY", "INSTITUTE", "ADMISSION", "COACHING", "CAMPUS"]):
        vertical = "Education"
    elif any(k in upper_text for k in ["HOSPITAL", "CLINIC", "MEDICAL", "HEALTH", "DIAGNOSTIC", "PHARMA", "AYURVEDA", "DENTAL"]):
        vertical = "Healthcare"
    elif any(k in upper_text for k in ["BANK", "FINANCE", "INSURANCE", "LOAN", "MUTUAL FUND", "CHIT"]):
        vertical = "BFSI"
    elif any(k in upper_text for k in ["JEWEL", "GOLD", "DIAMOND", "SILK", "SAREE", "MALL", "SHOWROOM", "FURNITURE", "ELECTRONICS", "SUPERMARKET"]):
        vertical = "Retail"
    elif any(k in upper_text for k in ["BUILDER", "DEVELOPER", "REALTY", "APARTMENT", "VILLA", "PLOT", "RERA", "HOUSING", "INFRA"]):
        vertical = "Real Estate"
    elif any(k in upper_text for k in ["MOTOR", "HYUNDAI", "TATA", "MARUTI", "TOYOTA", "HONDA", "KIA", "MAHINDRA", "SUZUKI", "EV ", "SCOOTER", "CAR "]):
        vertical = "Automobile"

    # Detect Client Name: prefer line with organizational/brand suffix or prominent header line
    client_name = ""
    org_keywords = ["LIMITED", "LTD", "CORPORATION", "GOVERNMENT", "DEPARTMENT", "MINISTRY", "AUTHORITY", "UNIVERSITY", "COLLEGE", "SCHOOL", "INSTITUTE", "HOSPITAL", "BANK", "JEWELLERS", "SILKS", "BUILDERS", "DEVELOPERS", "MOTORS", "BOARD", "TRUST"]
    for line in lines[:12]:
        clean = re.sub(r"[^A-Za-z0-9\s&\.\-\(\),/]", "", line).strip().upper()
        if len(clean) >= 4 and any(ok in clean for ok in org_keywords):
            client_name = clean[:120]
            break

    if not client_name:
        for line in lines[:8]:
            clean = re.sub(r"[^A-Za-z0-9\s&\.\-\(\)]", "", line).strip().upper()
            if len(clean) >= 4 and not re.match(r"^(DATE|PLACE|PAGE|THE NEW INDIAN EXPRESS|BENGALURU|CALL|PH|MOB|WWW|HTTP|\d+$)", clean):
                client_name = clean[:120]
                break

    if not client_name:
        return None

    # Detect Product / Campaign line (first meaningful line distinct from client_name)
    product = "-"
    for line in lines[:10]:
        clean = re.sub(r"[^A-Za-z0-9\s&\.\-\(\)%₹]", "", line).strip().upper()
        if len(clean) >= 4 and clean != client_name and not clean.startswith(client_name[:10]):
            product = clean[:100]
            break

    # Detect City / Centre
    centre = ""
    for city in CITIES:
        if re.search(r"\b" + re.escape(city) + r"\b", upper_text):
            centre = "Bengaluru" if city == "BANGALORE" else ("Mysuru" if city == "MYSORE" else city.title())
            break

    return {
        "client_name": client_name,
        "client_centre": centre,
        "product": product,
        "ad_category": category,
        "ad_verticals": vertical,
    }

def detect_ad_boxes(img_bgr: np.ndarray) -> List[tuple]:
    """
    Detect distinct rectangular bordered advertisement blocks on the page using OpenCV
    morphology + contour hierarchy, plus full/half/quarter page boxes.
    """
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Adaptive threshold + Canny to highlight printed rectangular ad borders and shaded boxes
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, 40, 140)

    # Close small gaps in horizontal and vertical border lines
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
    closed_h = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_h)
    closed_v = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_v)
    combined = cv2.bitwise_or(closed_h, closed_v)
    combined = cv2.dilate(combined, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)

    contours, _ = cv2.findContours(combined, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    min_w = int(w * 0.095)   # At least ~1 column wide (~3.2 cm)
    min_h = int(h * 0.055)   # At least ~3 cm high
    max_y_header = int(h * 0.04) # Ignore top folio header strip

    candidates = []
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        if cw < min_w or ch < min_h:
            continue
        if y < max_y_header and ch < int(h * 0.08):
            continue
        # Check rectangularity
        area = cv2.contourArea(cnt)
        rect_area = cw * ch
        if rect_area <= 0:
            continue
        if (area / float(rect_area)) < 0.45:
            continue
        candidates.append((x, y, cw, ch, rect_area))

    # Sort largest rectangles first and deduplicate nested/overlapping boxes
    candidates.sort(key=lambda b: b[4], reverse=True)
    kept = []
    for box in candidates:
        x1, y1, w1, h1, a1 = box
        # Skip if it covers >97% of the whole page unless there are no inner boxes
        if a1 > 0.97 * (w * h) and len(candidates) > 3:
            continue
        overlap = False
        for kx, ky, kw, kh, ka in kept:
            ix1 = max(x1, kx)
            iy1 = max(y1, ky)
            ix2 = min(x1 + w1, kx + kw)
            iy2 = min(y1 + h1, ky + kh)
            if ix2 > ix1 and iy2 > iy1:
                inter = (ix2 - ix1) * (iy2 - iy1)
                # If box is mostly inside an already-kept box or nearly identical, skip it
                if inter / float(min(a1, ka)) > 0.72:
                    overlap = True
                    break
        if not overlap:
            kept.append(box)

    return [(b[0], b[1], b[2], b[3]) for b in kept[:15]]

@app.get("/")
def health():
    return {"status": "ok", "service": "CRMS Self-Hosted Broadsheet OCR API"}

@app.post("/scan")
def scan_page(req: ScanRequest):
    try:
        raw_bytes = base64.b64decode(req.image_base64)
        pil_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        img_rgb = np.array(pil_img)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image_base64: {e}")

    H, W = img_bgr.shape[:2]
    boxes = detect_ad_boxes(img_bgr)

    # Run PaddleOCR once on the page and group text lines into detected boxes
    ocr_result = ocr_engine.ocr(img_rgb, cls=True)
    ocr_items = []
    if ocr_result and ocr_result[0]:
        for item in ocr_result[0]:
            pts, (text, conf) = item
            if conf < 0.50 or not text.strip():
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            cx = sum(xs) / 4.0
            cy = sum(ys) / 4.0
            ocr_items.append((cx, cy, min(xs), min(ys), max(xs), max(ys), text.strip(), float(conf)))

    # If no inner ad boxes were found but page has strong commercial text (e.g. borderless full-page jacket ad)
    if not boxes and ocr_items:
        boxes = [(0, 0, W, H)]

    ads_out = []
    for (bx, by, bw, bh) in boxes:
        box_lines = []
        confs = []
        for (cx, cy, x0, y0, x1, y1, text, conf) in ocr_items:
            if bx <= cx <= (bx + bw) and by <= cy <= (by + bh):
                box_lines.append((y0, x0, text))
                confs.append(conf)

        if not box_lines:
            continue
        # Sort top-to-bottom, left-to-right
        box_lines.sort(key=lambda t: (round(t[0] / 18.0), t[1]))
        ordered_texts = [t[2] for t in box_lines]

        nx = max(0, min(999, int(round((bx / float(W)) * 1000))))
        ny = max(0, min(999, int(round((by / float(H)) * 1000))))
        nw = max(1, min(1000 - nx, int(round((bw / float(W)) * 1000))))
        nh = max(1, min(1000 - ny, int(round((bh / float(H)) * 1000))))

        info = classify_ad_text(ordered_texts, nw, nh)
        if not info:
            continue

        # Estimate column span (1..8) on a 33 cm broadsheet
        cols = max(1, min(8, int(round((nw / 1000.0) * 8.0))))
        avg_conf = round(sum(confs) / len(confs), 4) if confs else 0.85

        ads_out.append({
            "client_name": info["client_name"],
            "client_centre": info["client_centre"],
            "product": info["product"],
            "ad_category": info["ad_category"],
            "ad_verticals": info["ad_verticals"],
            "columns": cols,
            "bbox": {"x": nx, "y": ny, "width": nw, "height": nh},
            "confidence": avg_conf
        })

    return {"ads": ads_out}
