import pytesseract
import torch
import difflib
import cv2

def get_best_match(text):
    options = {
    "kredi kartindan nakit avans cekme": "R",
    "kredi karti borcu odeme": "R",
    "kart hareketleri": "R",
    "tum islemler": "R",
    "hesaplar": "L",
    "yatirim ve doviz": "L",
    "kartlar": "L",
    "odemeler": "L",
    "diger": "R",
    "para transferi": "R",
    "para yatirma": "R",
    "nakit avans": "R",
    "emekli maas promosyon cayma": "L",
    "garanti atm sigortasi": "L",
    "sifre parola islemleri": "L",
    "telefon numara islemleri": "L",
    "tanimlamalar": "R",
    "bireysel kredi basvurusu": "R",
    "sim kart islemleri": "R",
    "atm ve sube bilgileri": "R",
    "baskasinin garanti bbva hesabina": "L",
    "hesabima": "R",
    "ana menu": "L",
    "vadesiz hesap ac": "R",
    "ekstre": "R",
    "donem ici islemler": "R",
    "gelecek donem borc bilgileri": "R",
    "acik provizyon": "R",
    "tutar girerek ode": "R",
    "min odeme tutarini ode": "R",
    "kalan borcu ode": "R",
    }

    #0.55 firs
    matches = difflib.get_close_matches(text, options, n=1, cutoff=0.55)
    return matches[0] if matches else "Bilinmeyen İslem"

def detect_fingertip(img, fingertip_model):
    results = fingertip_model(img, conf=0.4, max_det=1)
    if len(results[0].boxes) != 0:
        key_points = results[0].keypoints.xy
        x = int((key_points[0][0][0]).item())
        y = int((key_points[0][0][1]).item())
        return x, y
    else:
        return None

def produce_output(img, button_model, fingertip_model):

    results = button_model(img, save=False)

    try:
        hx, hy = detect_fingertip(img, fingertip_model)
    except:
        return "Parmağınızı ekrana getirip tekrar deneyin"

    for result in results:
        for box in result.boxes:
            if float(box.conf) < 0.6:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            if not (x1 <= hx <= x2 and y1 <= hy <= y2):
                continue

            region = img[y1:y2, x1:x2]
            
            # Preprocess image for better OCR results
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            
            # Perform OCR with Turkish language
            ocr_text = pytesseract.image_to_string(thresh, lang='tur', config='--psm 6')
            ocr_text = ocr_text.strip().lower()


            matched_label = get_best_match(ocr_text)

            if matched_label == "Bilinmeyen İslem":
                if box.cls == torch.tensor([1.]):
                    matched_label = "Parmağınızı sola kaydırıp tekrar deneyin."
                else:
                    matched_label = "Parmağınızı sağa kaydırıp tekrar deneyin."
            return matched_label
    return "Parmağınız hiçbir butonun üzerinde değil."
    