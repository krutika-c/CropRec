"""
CropSense backend — Flask app
Improvements over previous version:
  * Input validation: rejects negatives, enforces pH 0-14, checks all fields
  * Unit handling: users enter N/P/K in kg/ha (matching model training data)
  * Rainfall normalisation: real-world mm values scaled to dataset range
  * Safe t() that never crashes on missing/mismatched format keys
  * Translated crop names used in every explanation sentence
  * Modular structure: validate → convert → predict → explain → respond
"""

from flask import Flask, request, jsonify, send_from_directory
import pickle
import numpy as np
import warnings
import requests as http_requests

warnings.filterwarnings('ignore')

app = Flask(__name__, static_folder='static')

# ── Model ──────────────────────────────────────────────────────────────────────
with open('crop_model.pkl', 'rb') as f:
    model = pickle.load(f)

KNOWN_CROPS = [
    'apple', 'banana', 'blackgram', 'chickpea', 'coconut', 'coffee',
    'cotton', 'grapes', 'jute', 'kidneybeans', 'lentil', 'maize',
    'mango', 'mothbeans', 'mungbean', 'muskmelon', 'orange', 'papaya',
    'pigeonpeas', 'pomegranate', 'rice', 'watermelon'
]

# ── N/P/K units ───────────────────────────────────────────────────────────────
# The model was trained with N/P/K in kg/ha (confirmed from tree thresholds:
# N 0-120, P 7-143, K 14-204). Users enter kg/ha directly — no conversion needed.
#
# Rainfall: training range ~20-300 mm. We log-compress values above 300 mm.
RAINFALL_DATASET_MAX = 300.0
RAINFALL_REAL_MAX    = 3000.0


def normalise_rainfall(mm: float) -> float:
    """
    Scale annual rainfall (mm) into the model training range (~20–300 mm).
    Values ≤ 300 mm pass through unchanged.
    Values > 300 mm are log-compressed into the 150–300 band.
    """
    if mm <= RAINFALL_DATASET_MAX:
        return round(mm, 2)
    frac = min(
        np.log1p(mm - RAINFALL_DATASET_MAX) /
        np.log1p(RAINFALL_REAL_MAX - RAINFALL_DATASET_MAX),
        1.0
    )
    return round(150.0 + frac * 150.0, 2)


# ── Rule-based override (hybrid logic) ────────────────────────────────────────

def rule_based_override(N, P, K, temp, humidity, ph, rainfall, ml_crop, confidence):
    """
    Override ML prediction with agronomic rules when confidence is low (< 60%).
    Returns the ML prediction unchanged if confidence is high enough.
    """
    # Only override when ML is unsure
    if confidence >= 60:
        return ml_crop

    # 🌾 Rice
    if rainfall > 1500 and 20 <= temp <= 35 and humidity > 70:
        return "rice"

    # 🍌 Banana
    if rainfall > 2000 and temp >= 25 and humidity > 70:
        return "banana"

    # 🌽 Maize
    if 500 <= rainfall <= 1000 and 20 <= temp <= 32:
        return "maize"

    # 🌾 Wheat
    if temp < 20 and rainfall < 1000:
        return "wheat"

    # 🫘 Dry crops
    if rainfall < 600 and temp >= 25:
        return "mothbeans"

    # 🍇 Grapes
    if 500 <= rainfall <= 800:
        return "grapes"

    # 🍎 Apple
    if temp < 18:
        return "apple"

    # 🌴 Coconut
    if rainfall > 2500:
        return "coconut"

    return ml_crop


# ── Input validation ───────────────────────────────────────────────────────────
# Bounds match the model training range (kg/ha for NPK, mm for rainfall).
FIELD_BOUNDS = {
    'N':           (0,    140),   # kg/ha — model trained up to ~120
    'P':           (0,    150),   # kg/ha — model trained up to ~143
    'K':           (0,    210),   # kg/ha — model trained up to ~204
    'temperature': (-10,   60),   # °C
    'humidity':    (0,    100),   # %
    'ph':          (0,     14),   # unitless
    'rainfall':    (0,   5000),   # mm/year
}


def validate_inputs(data: dict):
    """
    Parse and range-check all numeric inputs.
    Returns (parsed_dict, None) on success or (None, error_str) on failure.
    """
    parsed = {}
    for field, (lo, hi) in FIELD_BOUNDS.items():
        raw = data.get(field, '')
        if raw == '' or raw is None:
            return None, f"Missing value for '{field}'."
        try:
            val = float(raw)
        except (ValueError, TypeError):
            return None, f"Invalid value for '{field}': must be a number."
        if not (lo <= val <= hi):
            return None, f"'{field}' must be between {lo} and {hi} (got {val})."
        parsed[field] = val
    return parsed, None


def build_model_features(parsed: dict) -> list:
    """
    Assemble the 7-feature vector the model expects:
      [N, P, K, temperature, humidity, ph, rainfall_normalised]
    N/P/K are passed through directly in kg/ha — no conversion needed.
    """
    return [
        parsed['N'],
        parsed['P'],
        parsed['K'],
        parsed['temperature'],
        parsed['humidity'],
        parsed['ph'],
        normalise_rainfall(parsed['rainfall']),
    ]


# ── Crop profiles (kg/ha for NPK, mm for rainfall) ────────────────────────────
# These are used only for agronomic explanations — not for model inference.
CROP_PROFILES = {
    'rice':        dict(N=(60,120), P=(30,60),  K=(30,60),  temp=(20,35), humidity=(70,90), ph=(5.5,7.0), rain=(150,300)),
    'maize':       dict(N=(60,120), P=(30,60),  K=(30,60),  temp=(18,30), humidity=(50,75), ph=(5.5,7.5), rain=(60,120)),
    'chickpea':    dict(N=(20,60),  P=(40,80),  K=(20,40),  temp=(15,25), humidity=(30,60), ph=(6.0,8.0), rain=(30,80)),
    'kidneybeans': dict(N=(20,60),  P=(60,100), K=(20,40),  temp=(15,25), humidity=(50,70), ph=(6.0,7.5), rain=(80,140)),
    'pigeonpeas':  dict(N=(20,40),  P=(40,80),  K=(20,40),  temp=(20,35), humidity=(40,65), ph=(5.5,7.0), rain=(60,120)),
    'mothbeans':   dict(N=(10,30),  P=(20,50),  K=(20,40),  temp=(25,38), humidity=(25,50), ph=(6.0,8.0), rain=(30,70)),
    'mungbean':    dict(N=(10,40),  P=(20,50),  K=(20,40),  temp=(25,35), humidity=(50,75), ph=(6.0,7.5), rain=(60,100)),
    'blackgram':   dict(N=(20,40),  P=(40,70),  K=(20,40),  temp=(25,35), humidity=(60,80), ph=(6.0,7.5), rain=(60,100)),
    'lentil':      dict(N=(20,40),  P=(40,80),  K=(20,40),  temp=(15,25), humidity=(50,70), ph=(6.0,8.0), rain=(30,80)),
    'pomegranate': dict(N=(30,60),  P=(40,70),  K=(40,80),  temp=(20,38), humidity=(25,50), ph=(5.5,7.5), rain=(50,100)),
    'banana':      dict(N=(80,140), P=(60,100), K=(100,200),temp=(20,35), humidity=(70,90), ph=(5.5,7.0), rain=(100,200)),
    'mango':       dict(N=(20,60),  P=(20,50),  K=(30,60),  temp=(24,38), humidity=(40,65), ph=(5.5,7.5), rain=(60,120)),
    'grapes':      dict(N=(20,60),  P=(40,80),  K=(40,80),  temp=(15,30), humidity=(50,70), ph=(5.5,7.0), rain=(50,100)),
    'watermelon':  dict(N=(80,120), P=(40,80),  K=(40,80),  temp=(22,35), humidity=(60,80), ph=(6.0,7.5), rain=(40,80)),
    'muskmelon':   dict(N=(60,100), P=(40,80),  K=(40,80),  temp=(25,38), humidity=(55,75), ph=(6.0,7.5), rain=(30,60)),
    'apple':       dict(N=(20,60),  P=(60,100), K=(40,80),  temp=(5,20),  humidity=(50,75), ph=(5.5,7.0), rain=(100,180)),
    'orange':      dict(N=(20,60),  P=(30,60),  K=(30,60),  temp=(15,30), humidity=(60,80), ph=(6.0,7.5), rain=(100,180)),
    'papaya':      dict(N=(30,60),  P=(30,60),  K=(40,80),  temp=(22,35), humidity=(60,80), ph=(6.0,7.5), rain=(100,200)),
    'coconut':     dict(N=(20,60),  P=(20,50),  K=(60,120), temp=(22,38), humidity=(60,90), ph=(5.5,8.0), rain=(100,250)),
    'cotton':      dict(N=(80,140), P=(40,80),  K=(40,80),  temp=(21,35), humidity=(50,70), ph=(6.0,8.0), rain=(60,100)),
    'jute':        dict(N=(60,100), P=(30,60),  K=(30,60),  temp=(25,38), humidity=(70,90), ph=(6.0,7.5), rain=(150,250)),
    'coffee':      dict(N=(60,100), P=(40,80),  K=(40,80),  temp=(15,28), humidity=(60,80), ph=(5.5,6.5), rain=(150,250)),
}

# ── Simple fertilizer suggestions ─────────────────────────────────────────────
# Thresholds are in kg/ha (matching model training units).
FERT_N_LOW, FERT_N_HIGH = 40, 100    # kg/ha
FERT_P_LOW, FERT_P_HIGH = 20,  60    # kg/ha
FERT_K_LOW, FERT_K_HIGH = 20,  80    # kg/ha
FERT_PH_LOW, FERT_PH_HIGH = 5.5, 8.0

FERTILIZER_DATA = {
    'N_low': [
        dict(name='Urea (46% N)',              dose='25–30 kg/acre', timing='Split: half at sowing, half 30 days later', method='Broadcast + soil incorporation'),
        dict(name='Ammonium Sulphate (21% N)', dose='50–55 kg/acre', timing='Basal dose at sowing',                      method='Drill or broadcast before ploughing'),
    ],
    'P_low': [
        dict(name='Single Super Phosphate (16% P₂O₅)', dose='50–60 kg/acre', timing='Full dose at sowing as basal',    method='Drill into soil near root zone'),
        dict(name='DAP (46% P₂O₅)',                    dose='25–30 kg/acre', timing='Basal dose before transplanting', method='Broadcast + mix into topsoil'),
    ],
    'K_low': [
        dict(name='MOP / Muriate of Potash (60% K₂O)',  dose='20–25 kg/acre', timing='Basal at sowing or transplanting', method='Broadcast and incorporate'),
        dict(name='SOP / Sulphate of Potash (50% K₂O)', dose='25–30 kg/acre', timing='Split: basal + 45 days after',    method='Side-dress near root zone'),
    ],
    'ph_high': [
        dict(name='Gypsum (Calcium Sulphate)', dose='100–200 kg/acre', timing='2–4 weeks before sowing',   method='Broadcast and water in thoroughly'),
        dict(name='Elemental Sulphur',          dose='10–20 kg/acre',   timing='1–2 months before sowing', method='Mix into top 15 cm of soil'),
    ],
    'ph_low': [
        dict(name='Agricultural Lime (CaCO₃)', dose='100–300 kg/acre', timing='4–6 weeks before sowing', method='Broadcast evenly and plough in'),
        dict(name='Dolomite Lime',              dose='100–200 kg/acre', timing='4–6 weeks before sowing', method='Broadcast and incorporate deeply'),
    ],
}


CROP_NAMES = {
    'English':  {'apple':'Apple','banana':'Banana','blackgram':'Black Gram','chickpea':'Chickpea','coconut':'Coconut','coffee':'Coffee','cotton':'Cotton','grapes':'Grapes','jute':'Jute','kidneybeans':'Kidney Beans','lentil':'Lentil','maize':'Maize','mango':'Mango','mothbeans':'Moth Beans','mungbean':'Mung Bean','muskmelon':'Muskmelon','orange':'Orange','papaya':'Papaya','pigeonpeas':'Pigeon Peas','pomegranate':'Pomegranate','rice':'Rice','watermelon':'Watermelon'},
    'Hindi':    {'apple':'सेब','banana':'केला','blackgram':'उड़द','chickpea':'चना','coconut':'नारियल','coffee':'कॉफी','cotton':'कपास','grapes':'अंगूर','jute':'जूट','kidneybeans':'राजमा','lentil':'मसूर','maize':'मक्का','mango':'आम','mothbeans':'मोठ','mungbean':'मूँग','muskmelon':'खरबूज','orange':'संतरा','papaya':'पपीता','pigeonpeas':'अरहर','pomegranate':'अनार','rice':'चावल','watermelon':'तरबूज'},
    'Telugu':   {'apple':'యాపిల్','banana':'అరటి','blackgram':'మినప','chickpea':'శనగ','coconut':'కొబ్బరి','coffee':'కాఫీ','cotton':'పత్తి','grapes':'ద్రాక్ష','jute':'జనపనార','kidneybeans':'రాజ్మా','lentil':'మసూర్','maize':'మొక్కజొన్న','mango':'మామిడి','mothbeans':'మోత్ బీన్స్','mungbean':'పెసర','muskmelon':'ఖర్బూజా','orange':'నారింజ','papaya':'బొప్పాయి','pigeonpeas':'కంది','pomegranate':'దానిమ్మ','rice':'వరి','watermelon':'పుచ్చకాయ'},
    'Tamil':    {'apple':'ஆப்பிள்','banana':'வாழை','blackgram':'உளுந்து','chickpea':'கொண்டைக்கடலை','coconut':'தேங்காய்','coffee':'காபி','cotton':'பருத்தி','grapes':'திராட்சை','jute':'சணல்','kidneybeans':'ராஜ்மா','lentil':'மசூர்','maize':'மக்காச்சோளம்','mango':'மாம்பழம்','mothbeans':'மாத் பீன்ஸ்','mungbean':'பாசிப்பயறு','muskmelon':'முலாம்பழம்','orange':'ஆரஞ்சு','papaya':'பப்பாளி','pigeonpeas':'துவரை','pomegranate':'மாதுளை','rice':'அரிசி','watermelon':'தர்பூசணி'},
    'Kannada':  {'apple':'ಸೇಬು','banana':'ಬಾಳೆ','blackgram':'ಉದ್ದು','chickpea':'ಕಡಲೆ','coconut':'ತೆಂಗು','coffee':'ಕಾಫಿ','cotton':'ಹತ್ತಿ','grapes':'ದ್ರಾಕ್ಷಿ','jute':'ಸೆಣಬು','kidneybeans':'ರಾಜ್ಮಾ','lentil':'ಮಸೂರ','maize':'ಜೋಳ','mango':'ಮಾವು','mothbeans':'ಮೋತ್ ಬೀನ್ಸ್','mungbean':'ಹೆಸರು','muskmelon':'ಕರ್ಬೂಜ','orange':'ಕಿತ್ತಳೆ','papaya':'ಪಪ್ಪಾಯ','pigeonpeas':'ತೊಗರಿ','pomegranate':'ದಾಳಿಂಬೆ','rice':'ಭತ್ತ','watermelon':'ಕಲ್ಲಂಗಡಿ'},
    'Marathi':  {'apple':'सफरचंद','banana':'केळ','blackgram':'उडीद','chickpea':'हरभरा','coconut':'नारळ','coffee':'कॉफी','cotton':'कापूस','grapes':'द्राक्ष','jute':'ताग','kidneybeans':'राजमा','lentil':'मसूर','maize':'मका','mango':'आंबा','mothbeans':'मटकी','mungbean':'मूग','muskmelon':'खरबूज','orange':'संत्री','papaya':'पपई','pigeonpeas':'तूर','pomegranate':'डाळिंब','rice':'तांदूळ','watermelon':'टरबूज'},
    'Bengali':  {'apple':'আপেল','banana':'কলা','blackgram':'কালাই','chickpea':'ছোলা','coconut':'নারকেল','coffee':'কফি','cotton':'তুলা','grapes':'আঙুর','jute':'পাট','kidneybeans':'রাজমা','lentil':'মসুর','maize':'ভুট্টা','mango':'আম','mothbeans':'মথ ডাল','mungbean':'মুগ','muskmelon':'বাঙ্গি','orange':'কমলা','papaya':'পেঁপে','pigeonpeas':'অড়হর','pomegranate':'ডালিম','rice':'ধান','watermelon':'তরমুজ'},
    'Punjabi':  {'apple':'ਸੇਬ','banana':'ਕੇਲਾ','blackgram':'ਮਾਂਹ','chickpea':'ਛੋਲੇ','coconut':'ਨਾਰੀਅਲ','coffee':'ਕੌਫੀ','cotton':'ਕਪਾਹ','grapes':'ਅੰਗੂਰ','jute':'ਜੂਟ','kidneybeans':'ਰਾਜਮਾਂਹ','lentil':'ਮਸਰ','maize':'ਮੱਕੀ','mango':'ਅੰਬ','mothbeans':'ਮੋਠ','mungbean':'ਮੂੰਗੀ','muskmelon':'ਖਰਬੂਜ਼ਾ','orange':'ਸੰਤਰਾ','papaya':'ਪਪੀਤਾ','pigeonpeas':'ਅਰਹਰ','pomegranate':'ਅਨਾਰ','rice':'ਚਾਵਲ','watermelon':'ਤਰਬੂਜ਼'},
    'Gujarati': {'apple':'સફરજન','banana':'કેળ','blackgram':'અડદ','chickpea':'ચણા','coconut':'નારિયેળ','coffee':'કૉફી','cotton':'કપાસ','grapes':'દ્રાક્ષ','jute':'શણ','kidneybeans':'રાજમા','lentil':'મસૂર','maize':'મકાઈ','mango':'કેરી','mothbeans':'મઠ','mungbean':'મગ','muskmelon':'તડબૂચ','orange':'નારંગી','papaya':'પપૈયું','pigeonpeas':'તુવેર','pomegranate':'દાડમ','rice':'ડાંગર','watermelon':'તરબૂચ'},
    'Odia':     {'apple':'ସେଓ','banana':'କଦଳୀ','blackgram':'ବିରି','chickpea':'ଚଣା','coconut':'ନଡ଼ିଆ','coffee':'କଫି','cotton':'କପା','grapes':'ଆଙ୍ଗୁର','jute':'ପାଟ','kidneybeans':'ରାଜମା','lentil':'ମସୁର','maize':'ମକା','mango':'ଆମ୍ବ','mothbeans':'ମୋଥ ଡାଲ','mungbean':'ମୁଗ','muskmelon':'ଖରଭୁଜ','orange':'କମଳା','papaya':'ଅମୃତଭଣ୍ଡା','pigeonpeas':'ହରଡ','pomegranate':'ଡାଳିମ୍ବ','rice':'ଚାଉଳ','watermelon':'ତରଭୁଜ'},
}

FERT_TRANSLATIONS = {
    'English': {
        'N_low_0_name': 'Urea (46% N)', 'N_low_0_dose': '25–30 kg/acre', 'N_low_0_timing': 'Split: half at sowing, half 30 days later', 'N_low_0_method': 'Broadcast + soil incorporation',
        'N_low_1_name': 'Ammonium Sulphate (21% N)', 'N_low_1_dose': '50–55 kg/acre', 'N_low_1_timing': 'Basal dose at sowing', 'N_low_1_method': 'Drill or broadcast before ploughing',
        'P_low_0_name': 'Single Super Phosphate (16% P₂O₅)', 'P_low_0_dose': '50–60 kg/acre', 'P_low_0_timing': 'Full dose at sowing as basal', 'P_low_0_method': 'Drill into soil near root zone',
        'P_low_1_name': 'DAP (46% P₂O₅)', 'P_low_1_dose': '25–30 kg/acre', 'P_low_1_timing': 'Basal dose before transplanting', 'P_low_1_method': 'Broadcast + mix into topsoil',
        'K_low_0_name': 'MOP / Muriate of Potash (60% K₂O)', 'K_low_0_dose': '20–25 kg/acre', 'K_low_0_timing': 'Basal at sowing or transplanting', 'K_low_0_method': 'Broadcast and incorporate',
        'K_low_1_name': 'SOP / Sulphate of Potash (50% K₂O)', 'K_low_1_dose': '25–30 kg/acre', 'K_low_1_timing': 'Split: basal + 45 days after', 'K_low_1_method': 'Side-dress near root zone',
        'ph_high_0_name': 'Gypsum (Calcium Sulphate)', 'ph_high_0_dose': '100–200 kg/acre', 'ph_high_0_timing': '2–4 weeks before sowing', 'ph_high_0_method': 'Broadcast and water in thoroughly',
        'ph_high_1_name': 'Elemental Sulphur', 'ph_high_1_dose': '10–20 kg/acre', 'ph_high_1_timing': '1–2 months before sowing', 'ph_high_1_method': 'Mix into top 15 cm of soil',
        'ph_low_0_name': 'Agricultural Lime (CaCO₃)', 'ph_low_0_dose': '100–300 kg/acre', 'ph_low_0_timing': '4–6 weeks before sowing', 'ph_low_0_method': 'Broadcast evenly and plough in',
        'ph_low_1_name': 'Dolomite Lime', 'ph_low_1_dose': '100–200 kg/acre', 'ph_low_1_timing': '4–6 weeks before sowing', 'ph_low_1_method': 'Broadcast and incorporate deeply',
        'nutrient_N': 'Nitrogen (N)', 'nutrient_P': 'Phosphorus (P)', 'nutrient_K': 'Potassium (K)', 'nutrient_ph': 'Soil pH',
        'status_good': 'good', 'status_low': 'low', 'status_excess': 'excess', 'status_high': 'high',
        'dose_label': 'Dose', 'timing_label': 'Timing', 'method_label': 'Method', 'unit': 'kg/ha',
    },
    'Hindi': {
        'N_low_0_name': 'यूरिया (46% N)', 'N_low_0_dose': '25–30 किग्रा/एकड़', 'N_low_0_timing': 'विभाजित: आधा बुवाई पर, आधा 30 दिन बाद', 'N_low_0_method': 'छिड़काव + मिट्टी में मिलाएँ',
        'N_low_1_name': 'अमोनियम सल्फेट (21% N)', 'N_low_1_dose': '50–55 किग्रा/एकड़', 'N_low_1_timing': 'बुवाई पर मूल खुराक', 'N_low_1_method': 'जुताई से पहले ड्रिल या छिड़काव',
        'P_low_0_name': 'सिंगल सुपर फॉस्फेट (16% P₂O₅)', 'P_low_0_dose': '50–60 किग्रा/एकड़', 'P_low_0_timing': 'बुवाई पर पूरी खुराक', 'P_low_0_method': 'जड़ क्षेत्र के पास मिट्टी में ड्रिल करें',
        'P_low_1_name': 'DAP (46% P₂O₅)', 'P_low_1_dose': '25–30 किग्रा/एकड़', 'P_low_1_timing': 'रोपाई से पहले मूल खुराक', 'P_low_1_method': 'छिड़काव + ऊपरी मिट्टी में मिलाएँ',
        'K_low_0_name': 'MOP / म्यूरेट ऑफ पोटाश (60% K₂O)', 'K_low_0_dose': '20–25 किग्रा/एकड़', 'K_low_0_timing': 'बुवाई या रोपाई पर मूल खुराक', 'K_low_0_method': 'छिड़काव और मिट्टी में मिलाएँ',
        'K_low_1_name': 'SOP / सल्फेट ऑफ पोटाश (50% K₂O)', 'K_low_1_dose': '25–30 किग्रा/एकड़', 'K_low_1_timing': 'विभाजित: मूल खुराक + 45 दिन बाद', 'K_low_1_method': 'जड़ क्षेत्र के पास साइड-ड्रेस',
        'ph_high_0_name': 'जिप्सम (कैल्शियम सल्फेट)', 'ph_high_0_dose': '100–200 किग्रा/एकड़', 'ph_high_0_timing': 'बुवाई से 2–4 सप्ताह पहले', 'ph_high_0_method': 'छिड़काव करें और पानी दें',
        'ph_high_1_name': 'एलिमेंटल सल्फर', 'ph_high_1_dose': '10–20 किग्रा/एकड़', 'ph_high_1_timing': 'बुवाई से 1–2 महीने पहले', 'ph_high_1_method': 'मिट्टी के ऊपरी 15 सेमी में मिलाएँ',
        'ph_low_0_name': 'कृषि चूना (CaCO₃)', 'ph_low_0_dose': '100–300 किग्रा/एकड़', 'ph_low_0_timing': 'बुवाई से 4–6 सप्ताह पहले', 'ph_low_0_method': 'समान रूप से छिड़काव करें और जुताई करें',
        'ph_low_1_name': 'डोलोमाइट चूना', 'ph_low_1_dose': '100–200 किग्रा/एकड़', 'ph_low_1_timing': 'बुवाई से 4–6 सप्ताह पहले', 'ph_low_1_method': 'छिड़काव करें और गहराई से मिलाएँ',
        'nutrient_N': 'नाइट्रोजन (N)', 'nutrient_P': 'फास्फोरस (P)', 'nutrient_K': 'पोटेशियम (K)', 'nutrient_ph': 'मिट्टी pH',
        'status_good': 'good', 'status_low': 'low', 'status_excess': 'excess', 'status_high': 'high',
        'dose_label': 'खुराक', 'timing_label': 'समय', 'method_label': 'तरीका', 'unit': 'किग्रा/हेक्टेयर',
    },
    'Telugu': {
        'N_low_0_name': 'యూరియా (46% N)', 'N_low_0_dose': '25–30 కి.గ్రా/ఎకరం', 'N_low_0_timing': 'విభజన: సగం విత్తనం సమయంలో, సగం 30 రోజుల తర్వాత', 'N_low_0_method': 'చల్లడం + నేలలో కలపడం',
        'N_low_1_name': 'అమ్మోనియం సల్ఫేట్ (21% N)', 'N_low_1_dose': '50–55 కి.గ్రా/ఎకరం', 'N_low_1_timing': 'విత్తనం సమయంలో మూల మోతాదు', 'N_low_1_method': 'దున్నడానికి ముందు డ్రిల్ లేదా చల్లండి',
        'P_low_0_name': 'సింగిల్ సూపర్ ఫాస్ఫేట్ (16% P₂O₅)', 'P_low_0_dose': '50–60 కి.గ్రా/ఎకరం', 'P_low_0_timing': 'విత్తనం సమయంలో పూర్తి మోతాదు', 'P_low_0_method': 'వేరు మండలం దగ్గర నేలలో వేయండి',
        'P_low_1_name': 'DAP (46% P₂O₅)', 'P_low_1_dose': '25–30 కి.గ్రా/ఎకరం', 'P_low_1_timing': 'నాట్లు వేయడానికి ముందు మూల మోతాదు', 'P_low_1_method': 'చల్లండి + పై నేలలో కలపండి',
        'K_low_0_name': 'MOP / మ్యూరేట్ ఆఫ్ పొటాష్ (60% K₂O)', 'K_low_0_dose': '20–25 కి.గ్రా/ఎకరం', 'K_low_0_timing': 'విత్తనం లేదా నాట్ల సమయంలో మూల మోతాదు', 'K_low_0_method': 'చల్లి కలపండి',
        'K_low_1_name': 'SOP / సల్ఫేట్ ఆఫ్ పొటాష్ (50% K₂O)', 'K_low_1_dose': '25–30 కి.గ్రా/ఎకరం', 'K_low_1_timing': 'విభజన: మూల + 45 రోజుల తర్వాత', 'K_low_1_method': 'వేరు మండలం దగ్గర సైడ్-డ్రెస్',
        'ph_high_0_name': 'జిప్సమ్ (కాల్షియం సల్ఫేట్)', 'ph_high_0_dose': '100–200 కి.గ్రా/ఎకరం', 'ph_high_0_timing': 'విత్తనానికి 2–4 వారాల ముందు', 'ph_high_0_method': 'చల్లి బాగా నీరు పెట్టండి',
        'ph_high_1_name': 'ఎలిమెంటల్ సల్ఫర్', 'ph_high_1_dose': '10–20 కి.గ్రా/ఎకరం', 'ph_high_1_timing': 'విత్తనానికి 1–2 నెలల ముందు', 'ph_high_1_method': 'పై 15 సెమీ నేలలో కలపండి',
        'ph_low_0_name': 'వ్యవసాయ సున్నం (CaCO₃)', 'ph_low_0_dose': '100–300 కి.గ్రా/ఎకరం', 'ph_low_0_timing': 'విత్తనానికి 4–6 వారాల ముందు', 'ph_low_0_method': 'సమానంగా చల్లి దున్నండి',
        'ph_low_1_name': 'డోలమైట్ లైమ్', 'ph_low_1_dose': '100–200 కి.గ్రా/ఎకరం', 'ph_low_1_timing': 'విత్తనానికి 4–6 వారాల ముందు', 'ph_low_1_method': 'చల్లి లోతుగా కలపండి',
        'nutrient_N': 'నత్రజని (N)', 'nutrient_P': 'భాస్వరం (P)', 'nutrient_K': 'పొటాషియం (K)', 'nutrient_ph': 'నేల pH',
        'status_good': 'good', 'status_low': 'low', 'status_excess': 'excess', 'status_high': 'high',
        'dose_label': 'మోతాదు', 'timing_label': 'సమయం', 'method_label': 'పద్ధతి', 'unit': 'మి.గ్రా/కి.గ్రా',
    },
    'Tamil': {
        'N_low_0_name': 'யூரியா (46% N)', 'N_low_0_dose': '25–30 கி.கி/ஏக்கர்', 'N_low_0_timing': 'பிரிந்த: பாதி விதைப்பில், பாதி 30 நாட்கள் பின்', 'N_low_0_method': 'தூவல் + மண்ணில் கலவு',
        'N_low_1_name': 'அம்மோனியம் சல்பேட் (21% N)', 'N_low_1_dose': '50–55 கி.கி/ஏக்கர்', 'N_low_1_timing': 'விதைப்பில் அடிப்படை அளவு', 'N_low_1_method': 'உழவுக்கு முன் துளையிட்டு தூவுங்கள்',
        'P_low_0_name': 'சிங்கிள் சூப்பர் பாஸ்பேட் (16% P₂O₅)', 'P_low_0_dose': '50–60 கி.கி/ஏக்கர்', 'P_low_0_timing': 'விதைப்பில் முழு அளவு', 'P_low_0_method': 'வேர் மண்டலத்தில் மண்ணில் போடுங்கள்',
        'P_low_1_name': 'DAP (46% P₂O₅)', 'P_low_1_dose': '25–30 கி.கி/ஏக்கர்', 'P_low_1_timing': 'நடவுக்கு முன் அடிப்படை அளவு', 'P_low_1_method': 'தூவி மேல் மண்ணில் கலவுங்கள்',
        'K_low_0_name': 'MOP / மூரியேட் ஆஃப் பொட்டாஷ் (60% K₂O)', 'K_low_0_dose': '20–25 கி.கி/ஏக்கர்', 'K_low_0_timing': 'விதைப்பு அல்லது நடவில் அடிப்படை', 'K_low_0_method': 'தூவி கலவுங்கள்',
        'K_low_1_name': 'SOP / சல்பேட் ஆஃப் பொட்டாஷ் (50% K₂O)', 'K_low_1_dose': '25–30 கி.கி/ஏக்கர்', 'K_low_1_timing': 'பிரிந்த: அடிப்படை + 45 நாட்கள் பின்', 'K_low_1_method': 'வேர் அருகே சைட்-டிரஸ்',
        'ph_high_0_name': 'ஜிப்சம் (கால்சியம் சல்பேட்)', 'ph_high_0_dose': '100–200 கி.கி/ஏக்கர்', 'ph_high_0_timing': 'விதைப்புக்கு 2–4 வாரங்கள் முன்', 'ph_high_0_method': 'தூவி நன்கு தண்ணீர் பாய்ச்சுங்கள்',
        'ph_high_1_name': 'எலிமென்டல் சல்பர்', 'ph_high_1_dose': '10–20 கி.கி/ஏக்கர்', 'ph_high_1_timing': 'விதைப்புக்கு 1–2 மாதங்கள் முன்', 'ph_high_1_method': 'மேல் 15 செமீ மண்ணில் கலவுங்கள்',
        'ph_low_0_name': 'விவசாய சுண்ணாம்பு (CaCO₃)', 'ph_low_0_dose': '100–300 கி.கி/ஏக்கர்', 'ph_low_0_timing': 'விதைப்புக்கு 4–6 வாரங்கள் முன்', 'ph_low_0_method': 'சமவாக தூவி உழவுங்கள்',
        'ph_low_1_name': 'டோலமைட் சுண்ணாம்பு', 'ph_low_1_dose': '100–200 கி.கி/ஏக்கர்', 'ph_low_1_timing': 'விதைப்புக்கு 4–6 வாரங்கள் முன்', 'ph_low_1_method': 'தூவி ஆழமாக கலவுங்கள்',
        'nutrient_N': 'நைட்ரஜன் (N)', 'nutrient_P': 'பாஸ்பரஸ் (P)', 'nutrient_K': 'பொட்டாசியம் (K)', 'nutrient_ph': 'மண் pH',
        'status_good': 'good', 'status_low': 'low', 'status_excess': 'excess', 'status_high': 'high',
        'dose_label': 'அளவு', 'timing_label': 'நேரம்', 'method_label': 'முறை', 'unit': 'மி.கி/கி.கி',
    },
    'Kannada': {
        'N_low_0_name': 'ಯೂರಿಯಾ (46% N)', 'N_low_0_dose': '25–30 ಕಿ.ಗ್ರಾ/ಎಕರೆ', 'N_low_0_timing': 'ವಿಭಜಿತ: ಅರ್ಧ ಬಿತ್ತನೆಯಲ್ಲಿ, ಅರ್ಧ 30 ದಿನ ನಂತರ', 'N_low_0_method': 'ಬಿತ್ತರಿಸಿ + ಮಣ್ಣಿನಲ್ಲಿ ಬೆರೆಸಿ',
        'N_low_1_name': 'ಅಮೋನಿಯಂ ಸಲ್ಫೇಟ್ (21% N)', 'N_low_1_dose': '50–55 ಕಿ.ಗ್ರಾ/ಎಕರೆ', 'N_low_1_timing': 'ಬಿತ್ತನೆ ಸಮಯದಲ್ಲಿ ಮೂಲ ಪ್ರಮಾಣ', 'N_low_1_method': 'ಉಳುಮೆ ಮೊದಲು ಡ್ರಿಲ್ ಅಥವಾ ಬಿತ್ತರಿಸಿ',
        'P_low_0_name': 'ಸಿಂಗಲ್ ಸೂಪರ್ ಫಾಸ್ಫೇಟ್ (16% P₂O₅)', 'P_low_0_dose': '50–60 ಕಿ.ಗ್ರಾ/ಎಕರೆ', 'P_low_0_timing': 'ಬಿತ್ತನೆ ಸಮಯದಲ್ಲಿ ಪೂರ್ಣ ಪ್ರಮಾಣ', 'P_low_0_method': 'ಬೇರು ವಲಯದ ಬಳಿ ಮಣ್ಣಿನಲ್ಲಿ ಹಾಕಿ',
        'P_low_1_name': 'DAP (46% P₂O₅)', 'P_low_1_dose': '25–30 ಕಿ.ಗ್ರಾ/ಎಕರೆ', 'P_low_1_timing': 'ನಾಟಿ ಮೊದಲು ಮೂಲ ಪ್ರಮಾಣ', 'P_low_1_method': 'ಬಿತ್ತರಿಸಿ + ಮೇಲ್ಮಣ್ಣಿನಲ್ಲಿ ಬೆರೆಸಿ',
        'K_low_0_name': 'MOP / ಮ್ಯೂರಿಯೇಟ್ ಆಫ್ ಪೊಟ್ಯಾಶ್ (60% K₂O)', 'K_low_0_dose': '20–25 ಕಿ.ಗ್ರಾ/ಎಕರೆ', 'K_low_0_timing': 'ಬಿತ್ತನೆ ಅಥವಾ ನಾಟಿ ಸಮಯದಲ್ಲಿ', 'K_low_0_method': 'ಬಿತ್ತರಿಸಿ ಬೆರೆಸಿ',
        'K_low_1_name': 'SOP / ಸಲ್ಫೇಟ್ ಆಫ್ ಪೊಟ್ಯಾಶ್ (50% K₂O)', 'K_low_1_dose': '25–30 ಕಿ.ಗ್ರಾ/ಎಕರೆ', 'K_low_1_timing': 'ವಿಭಜಿತ: ಮೂಲ + 45 ದಿನ ನಂತರ', 'K_low_1_method': 'ಬೇರು ಬಳಿ ಸೈಡ್-ಡ್ರೆಸ್',
        'ph_high_0_name': 'ಜಿಪ್ಸಮ್ (ಕ್ಯಾಲ್ಸಿಯಂ ಸಲ್ಫೇಟ್)', 'ph_high_0_dose': '100–200 ಕಿ.ಗ್ರಾ/ಎಕರೆ', 'ph_high_0_timing': 'ಬಿತ್ತನೆ 2–4 ವಾರ ಮೊದಲು', 'ph_high_0_method': 'ಬಿತ್ತರಿಸಿ ಚೆನ್ನಾಗಿ ನೀರು ಹಾಕಿ',
        'ph_high_1_name': 'ಎಲಿಮೆಂಟಲ್ ಸಲ್ಫರ್', 'ph_high_1_dose': '10–20 ಕಿ.ಗ್ರಾ/ಎಕರೆ', 'ph_high_1_timing': 'ಬಿತ್ತನೆ 1–2 ತಿಂಗಳ ಮೊದಲು', 'ph_high_1_method': 'ಮೇಲ್ 15 ಸೆಂಮೀ ಮಣ್ಣಿನಲ್ಲಿ ಬೆರೆಸಿ',
        'ph_low_0_name': 'ಕೃಷಿ ಸುಣ್ಣ (CaCO₃)', 'ph_low_0_dose': '100–300 ಕಿ.ಗ್ರಾ/ಎಕರೆ', 'ph_low_0_timing': 'ಬಿತ್ತನೆ 4–6 ವಾರ ಮೊದಲು', 'ph_low_0_method': 'ಸಮವಾಗಿ ಬಿತ್ತರಿಸಿ ಉಳುಮೆ ಮಾಡಿ',
        'ph_low_1_name': 'ಡೋಲೋಮೈಟ್ ಸುಣ್ಣ', 'ph_low_1_dose': '100–200 ಕಿ.ಗ್ರಾ/ಎಕರೆ', 'ph_low_1_timing': 'ಬಿತ್ತನೆ 4–6 ವಾರ ಮೊದಲು', 'ph_low_1_method': 'ಬಿತ್ತರಿಸಿ ಆಳವಾಗಿ ಬೆರೆಸಿ',
        'nutrient_N': 'ಸಾರಜನಕ (N)', 'nutrient_P': 'ರಂಜಕ (P)', 'nutrient_K': 'ಪೊಟ್ಯಾಸಿಯಂ (K)', 'nutrient_ph': 'ಮಣ್ಣಿನ pH',
        'status_good': 'good', 'status_low': 'low', 'status_excess': 'excess', 'status_high': 'high',
        'dose_label': 'ಪ್ರಮಾಣ', 'timing_label': 'ಸಮಯ', 'method_label': 'ವಿಧಾನ', 'unit': 'ಮಿ.ಗ್ರಾ/ಕಿ.ಗ್ರಾ',
    },
    'Marathi': {
        'N_low_0_name': 'युरिया (46% N)', 'N_low_0_dose': '25–30 किग्रा/एकर', 'N_low_0_timing': 'विभाजित: अर्धा पेरणीत, अर्धा 30 दिवसांनी', 'N_low_0_method': 'फवारणी + जमिनीत मिसळणे',
        'N_low_1_name': 'अमोनियम सल्फेट (21% N)', 'N_low_1_dose': '50–55 किग्रा/एकर', 'N_low_1_timing': 'पेरणीत मूळ मात्रा', 'N_low_1_method': 'नांगरणीपूर्वी ड्रिल किंवा फवारणी',
        'P_low_0_name': 'सिंगल सुपर फॉस्फेट (16% P₂O₅)', 'P_low_0_dose': '50–60 किग्रा/एकर', 'P_low_0_timing': 'पेरणीत पूर्ण मात्रा', 'P_low_0_method': 'मूळ क्षेत्रापाशी जमिनीत टाका',
        'P_low_1_name': 'DAP (46% P₂O₅)', 'P_low_1_dose': '25–30 किग्रा/एकर', 'P_low_1_timing': 'लावणीपूर्वी मूळ मात्रा', 'P_low_1_method': 'फवारणी + वरच्या मातीत मिसळा',
        'K_low_0_name': 'MOP / म्युरिएट ऑफ पोटाश (60% K₂O)', 'K_low_0_dose': '20–25 किग्रा/एकर', 'K_low_0_timing': 'पेरणी किंवा लावणीवेळी मूळ मात्रा', 'K_low_0_method': 'फवारणी आणि मिसळा',
        'K_low_1_name': 'SOP / सल्फेट ऑफ पोटाश (50% K₂O)', 'K_low_1_dose': '25–30 किग्रा/एकर', 'K_low_1_timing': 'विभाजित: मूळ + 45 दिवसांनी', 'K_low_1_method': 'मूळापाशी साईड-ड्रेस',
        'ph_high_0_name': 'जिप्सम (कॅल्शियम सल्फेट)', 'ph_high_0_dose': '100–200 किग्रा/एकर', 'ph_high_0_timing': 'पेरणीच्या 2–4 आठवडे आधी', 'ph_high_0_method': 'फवारणी करा आणि नीट पाणी घाला',
        'ph_high_1_name': 'एलिमेंटल सल्फर', 'ph_high_1_dose': '10–20 किग्रा/एकर', 'ph_high_1_timing': 'पेरणीच्या 1–2 महिने आधी', 'ph_high_1_method': 'वरच्या 15 सेमी मातीत मिसळा',
        'ph_low_0_name': 'शेतीचा चुना (CaCO₃)', 'ph_low_0_dose': '100–300 किग्रा/एकर', 'ph_low_0_timing': 'पेरणीच्या 4–6 आठवडे आधी', 'ph_low_0_method': 'समान फवारणी करा आणि नांगरा',
        'ph_low_1_name': 'डोलोमाइट चुना', 'ph_low_1_dose': '100–200 किग्रा/एकर', 'ph_low_1_timing': 'पेरणीच्या 4–6 आठवडे आधी', 'ph_low_1_method': 'फवारणी करा आणि खोलवर मिसळा',
        'nutrient_N': 'नायट्रोजन (N)', 'nutrient_P': 'फॉस्फरस (P)', 'nutrient_K': 'पोटॅशियम (K)', 'nutrient_ph': 'मातीचा pH',
        'status_good': 'good', 'status_low': 'low', 'status_excess': 'excess', 'status_high': 'high',
        'dose_label': 'मात्रा', 'timing_label': 'वेळ', 'method_label': 'पद्धत', 'unit': 'मि.ग्रॅ/कि.ग्रॅ',
    },
    'Bengali': {
        'N_low_0_name': 'ইউরিয়া (46% N)', 'N_low_0_dose': '25–30 কিগ্রা/একর', 'N_low_0_timing': 'বিভক্ত: অর্ধেক বপনে, অর্ধেক 30 দিন পরে', 'N_low_0_method': 'ছড়িয়ে + মাটিতে মেশান',
        'N_low_1_name': 'অ্যামোনিয়াম সালফেট (21% N)', 'N_low_1_dose': '50–55 কিগ্রা/একর', 'N_low_1_timing': 'বপনে মূল মাত্রা', 'N_low_1_method': 'চাষের আগে ড্রিল বা ছড়ান',
        'P_low_0_name': 'সিঙ্গল সুপার ফসফেট (16% P₂O₅)', 'P_low_0_dose': '50–60 কিগ্রা/একর', 'P_low_0_timing': 'বপনে সম্পূর্ণ মাত্রা', 'P_low_0_method': 'শিকড় অঞ্চলের কাছে মাটিতে দিন',
        'P_low_1_name': 'DAP (46% P₂O₅)', 'P_low_1_dose': '25–30 কিগ্রা/একর', 'P_low_1_timing': 'রোপণের আগে মূল মাত্রা', 'P_low_1_method': 'ছড়িয়ে উপরের মাটিতে মেশান',
        'K_low_0_name': 'MOP / মিউরিয়েট অব পটাশ (60% K₂O)', 'K_low_0_dose': '20–25 কিগ্রা/একর', 'K_low_0_timing': 'বপন বা রোপণে মূল মাত্রা', 'K_low_0_method': 'ছড়িয়ে মেশান',
        'K_low_1_name': 'SOP / সালফেট অব পটাশ (50% K₂O)', 'K_low_1_dose': '25–30 কিগ্রা/একর', 'K_low_1_timing': 'বিভক্ত: মূল + 45 দিন পরে', 'K_low_1_method': 'শিকড়ের কাছে সাইড-ড্রেস',
        'ph_high_0_name': 'জিপসাম (ক্যালসিয়াম সালফেট)', 'ph_high_0_dose': '100–200 কিগ্রা/একর', 'ph_high_0_timing': 'বপনের 2–4 সপ্তাহ আগে', 'ph_high_0_method': 'ছড়িয়ে ভালো করে জল দিন',
        'ph_high_1_name': 'এলিমেন্টাল সালফার', 'ph_high_1_dose': '10–20 কিগ্রা/একর', 'ph_high_1_timing': 'বপনের 1–2 মাস আগে', 'ph_high_1_method': 'উপরের 15 সেমি মাটিতে মেশান',
        'ph_low_0_name': 'কৃষি চুন (CaCO₃)', 'ph_low_0_dose': '100–300 কিগ্রা/একর', 'ph_low_0_timing': 'বপনের 4–6 সপ্তাহ আগে', 'ph_low_0_method': 'সমানভাবে ছড়িয়ে চাষ করুন',
        'ph_low_1_name': 'ডোলোমাইট চুন', 'ph_low_1_dose': '100–200 কিগ্রা/একর', 'ph_low_1_timing': 'বপনের 4–6 সপ্তাহ আগে', 'ph_low_1_method': 'ছড়িয়ে গভীরে মেশান',
        'nutrient_N': 'নাইট্রোজেন (N)', 'nutrient_P': 'ফসফরাস (P)', 'nutrient_K': 'পটাশিয়াম (K)', 'nutrient_ph': 'মাটির pH',
        'status_good': 'good', 'status_low': 'low', 'status_excess': 'excess', 'status_high': 'high',
        'dose_label': 'মাত্রা', 'timing_label': 'সময়', 'method_label': 'পদ্ধতি', 'unit': 'মিগ্রা/কিগ্রা',
    },
    'Punjabi': {
        'N_low_0_name': 'ਯੂਰੀਆ (46% N)', 'N_low_0_dose': '25–30 ਕਿਗ੍ਰਾ/ਏਕੜ', 'N_low_0_timing': 'ਵੰਡਿਆ: ਅੱਧਾ ਬਿਜਾਈ ਸਮੇਂ, ਅੱਧਾ 30 ਦਿਨ ਬਾਅਦ', 'N_low_0_method': 'ਖਿਲਾਰਨਾ + ਮਿੱਟੀ ਵਿੱਚ ਮਿਲਾਉਣਾ',
        'N_low_1_name': 'ਅਮੋਨੀਅਮ ਸਲਫੇਟ (21% N)', 'N_low_1_dose': '50–55 ਕਿਗ੍ਰਾ/ਏਕੜ', 'N_low_1_timing': 'ਬਿਜਾਈ ਸਮੇਂ ਮੂਲ ਮਾਤਰਾ', 'N_low_1_method': 'ਵਾਹੀ ਤੋਂ ਪਹਿਲਾਂ ਡ੍ਰਿਲ ਜਾਂ ਖਿਲਾਰੋ',
        'P_low_0_name': 'ਸਿੰਗਲ ਸੁਪਰ ਫਾਸਫੇਟ (16% P₂O₅)', 'P_low_0_dose': '50–60 ਕਿਗ੍ਰਾ/ਏਕੜ', 'P_low_0_timing': 'ਬਿਜਾਈ ਸਮੇਂ ਪੂਰੀ ਮਾਤਰਾ', 'P_low_0_method': 'ਜੜ੍ਹ ਖੇਤਰ ਕੋਲ ਮਿੱਟੀ ਵਿੱਚ ਪਾਓ',
        'P_low_1_name': 'DAP (46% P₂O₅)', 'P_low_1_dose': '25–30 ਕਿਗ੍ਰਾ/ਏਕੜ', 'P_low_1_timing': 'ਲਵਾਈ ਤੋਂ ਪਹਿਲਾਂ ਮੂਲ ਮਾਤਰਾ', 'P_low_1_method': 'ਖਿਲਾਰੋ + ਉਪਰਲੀ ਮਿੱਟੀ ਵਿੱਚ ਮਿਲਾਓ',
        'K_low_0_name': 'MOP / ਮਿਊਰੀਏਟ ਆਫ ਪੋਟਾਸ਼ (60% K₂O)', 'K_low_0_dose': '20–25 ਕਿਗ੍ਰਾ/ਏਕੜ', 'K_low_0_timing': 'ਬਿਜਾਈ ਜਾਂ ਲਵਾਈ ਸਮੇਂ ਮੂਲ', 'K_low_0_method': 'ਖਿਲਾਰੋ ਅਤੇ ਮਿਲਾਓ',
        'K_low_1_name': 'SOP / ਸਲਫੇਟ ਆਫ ਪੋਟਾਸ਼ (50% K₂O)', 'K_low_1_dose': '25–30 ਕਿਗ੍ਰਾ/ਏਕੜ', 'K_low_1_timing': 'ਵੰਡਿਆ: ਮੂਲ + 45 ਦਿਨ ਬਾਅਦ', 'K_low_1_method': 'ਜੜ੍ਹ ਕੋਲ ਸਾਈਡ-ਡ੍ਰੈਸ',
        'ph_high_0_name': 'ਜਿਪਸਮ (ਕੈਲਸ਼ੀਅਮ ਸਲਫੇਟ)', 'ph_high_0_dose': '100–200 ਕਿਗ੍ਰਾ/ਏਕੜ', 'ph_high_0_timing': 'ਬਿਜਾਈ ਤੋਂ 2–4 ਹਫਤੇ ਪਹਿਲਾਂ', 'ph_high_0_method': 'ਖਿਲਾਰੋ ਅਤੇ ਚੰਗੀ ਤਰ੍ਹਾਂ ਪਾਣੀ ਦਿਓ',
        'ph_high_1_name': 'ਐਲੀਮੈਂਟਲ ਸਲਫਰ', 'ph_high_1_dose': '10–20 ਕਿਗ੍ਰਾ/ਏਕੜ', 'ph_high_1_timing': 'ਬਿਜਾਈ ਤੋਂ 1–2 ਮਹੀਨੇ ਪਹਿਲਾਂ', 'ph_high_1_method': 'ਉਪਰਲੀ 15 ਸੈਮੀ ਮਿੱਟੀ ਵਿੱਚ ਮਿਲਾਓ',
        'ph_low_0_name': 'ਖੇਤੀਬਾੜੀ ਚੂਨਾ (CaCO₃)', 'ph_low_0_dose': '100–300 ਕਿਗ੍ਰਾ/ਏਕੜ', 'ph_low_0_timing': 'ਬਿਜਾਈ ਤੋਂ 4–6 ਹਫਤੇ ਪਹਿਲਾਂ', 'ph_low_0_method': 'ਬਰਾਬਰ ਖਿਲਾਰੋ ਅਤੇ ਵਾਹੋ',
        'ph_low_1_name': 'ਡੋਲੋਮਾਈਟ ਚੂਨਾ', 'ph_low_1_dose': '100–200 ਕਿਗ੍ਰਾ/ਏਕੜ', 'ph_low_1_timing': 'ਬਿਜਾਈ ਤੋਂ 4–6 ਹਫਤੇ ਪਹਿਲਾਂ', 'ph_low_1_method': 'ਖਿਲਾਰੋ ਅਤੇ ਡੂੰਘੇ ਮਿਲਾਓ',
        'nutrient_N': 'ਨਾਈਟ੍ਰੋਜਨ (N)', 'nutrient_P': 'ਫਾਸਫੋਰਸ (P)', 'nutrient_K': 'ਪੋਟਾਸ਼ੀਅਮ (K)', 'nutrient_ph': 'ਮਿੱਟੀ pH',
        'status_good': 'good', 'status_low': 'low', 'status_excess': 'excess', 'status_high': 'high',
        'dose_label': 'ਮਾਤਰਾ', 'timing_label': 'ਸਮਾਂ', 'method_label': 'ਤਰੀਕਾ', 'unit': 'ਮਿਗ੍ਰਾ/ਕਿਗ੍ਰਾ',
    },
    'Gujarati': {
        'N_low_0_name': 'યુરિયા (46% N)', 'N_low_0_dose': '25–30 કિ.ગ્રા/એકર', 'N_low_0_timing': 'વિભાજિત: અડધો વાવણીમાં, અડધો 30 દિવસ પછી', 'N_low_0_method': 'ઢોળવું + જમીનમાં ભેળવવું',
        'N_low_1_name': 'અમોનિયમ સલ્ફેટ (21% N)', 'N_low_1_dose': '50–55 કિ.ગ્રા/એકર', 'N_low_1_timing': 'વાવણી સમયે મૂળ માત્રા', 'N_low_1_method': 'ખેડ પહેલાં ડ્રિલ અથવા ઢોળો',
        'P_low_0_name': 'સિંગલ સુપર ફોસ્ફેટ (16% P₂O₅)', 'P_low_0_dose': '50–60 કિ.ગ્રા/એકર', 'P_low_0_timing': 'વાવણી સમયે પૂર્ણ માત્રા', 'P_low_0_method': 'મૂળ ક્ષેત્ર પાસે જમીનમાં નાખો',
        'P_low_1_name': 'DAP (46% P₂O₅)', 'P_low_1_dose': '25–30 કિ.ગ્રા/એકર', 'P_low_1_timing': 'રોપણ પહેલાં મૂળ માત્રા', 'P_low_1_method': 'ઢોળો + ઉપરની માટીમાં ભેળવો',
        'K_low_0_name': 'MOP / મ્યુરિએટ ઓફ પોટાશ (60% K₂O)', 'K_low_0_dose': '20–25 કિ.ગ્રા/એકર', 'K_low_0_timing': 'વાવણી અથવા રોપણ સમયે મૂળ', 'K_low_0_method': 'ઢોળો અને ભેળવો',
        'K_low_1_name': 'SOP / સલ્ફેટ ઓફ પોટાશ (50% K₂O)', 'K_low_1_dose': '25–30 કિ.ગ્રા/એકર', 'K_low_1_timing': 'વિભાજિત: મૂળ + 45 દિવસ પછી', 'K_low_1_method': 'મૂળ પાસે સાઇડ-ડ્રેસ',
        'ph_high_0_name': 'જિપ્સમ (કેલ્સિયમ સલ્ફેટ)', 'ph_high_0_dose': '100–200 કિ.ગ્રા/એકર', 'ph_high_0_timing': 'વાવણીના 2–4 અઠવાડિયા પહેલાં', 'ph_high_0_method': 'ઢોળો અને સારી રીતે પાણી આપો',
        'ph_high_1_name': 'એલિમેન્ટલ સલ્ફર', 'ph_high_1_dose': '10–20 કિ.ગ્રા/એકર', 'ph_high_1_timing': 'વાવણીના 1–2 મહિના પહેલાં', 'ph_high_1_method': 'ઉપરની 15 સેમી માટીમાં ભેળવો',
        'ph_low_0_name': 'ખેતી ચૂનો (CaCO₃)', 'ph_low_0_dose': '100–300 કિ.ગ્રા/એકર', 'ph_low_0_timing': 'વાવણીના 4–6 અઠવાડિયા પહેલાં', 'ph_low_0_method': 'સમાન ઢોળો અને ખેડો',
        'ph_low_1_name': 'ડોલોમાઇટ ચૂનો', 'ph_low_1_dose': '100–200 કિ.ગ્રા/એકર', 'ph_low_1_timing': 'વાવણીના 4–6 અઠવાડિયા પહેલાં', 'ph_low_1_method': 'ઢોળો અને ઊંડે ભેળવો',
        'nutrient_N': 'નાઇટ્રોજન (N)', 'nutrient_P': 'ફોસ્ફરસ (P)', 'nutrient_K': 'પોટેશિયમ (K)', 'nutrient_ph': 'જમીન pH',
        'status_good': 'good', 'status_low': 'low', 'status_excess': 'excess', 'status_high': 'high',
        'dose_label': 'માત્રા', 'timing_label': 'સમય', 'method_label': 'પદ્ધતિ', 'unit': 'મિ.ગ્રા/કિ.ગ્રા',
    },
    'Odia': {
        'N_low_0_name': 'ୟୁରିଆ (46% N)', 'N_low_0_dose': '25–30 କିଗ୍ରା/ଏକର', 'N_low_0_timing': 'ବିଭକ୍ତ: ଅଧା ବୁଣିବା ସମୟ, ଅଧା 30 ଦିନ ପରେ', 'N_low_0_method': 'ଛଟ + ମାଟିରେ ମିଶ୍ରଣ',
        'N_low_1_name': 'ଆମୋନିୟମ ସଲ୍ଫେଟ (21% N)', 'N_low_1_dose': '50–55 କିଗ୍ରା/ଏକର', 'N_low_1_timing': 'ବୁଣିବା ସମୟ ମୂଳ ମାତ୍ରା', 'N_low_1_method': 'ହଳ ପୂର୍ବରୁ ଡ୍ରିଲ ବା ଛଟ',
        'P_low_0_name': 'ସିଙ୍ଗଲ ସୁପର ଫସ୍ଫେଟ (16% P₂O₅)', 'P_low_0_dose': '50–60 କିଗ୍ରା/ଏକର', 'P_low_0_timing': 'ବୁଣିବା ସମୟ ସଂପୂର୍ଣ ମାତ୍ରା', 'P_low_0_method': 'ମୂଳ ଅଞ୍ଚଳ ପାଖ ମାଟିରେ ଦିଅ',
        'P_low_1_name': 'DAP (46% P₂O₅)', 'P_low_1_dose': '25–30 କିଗ୍ରା/ଏକର', 'P_low_1_timing': 'ରୋପଣ ପୂର୍ବରୁ ମୂଳ ମାତ୍ରା', 'P_low_1_method': 'ଛଟ + ଉପର ମାଟିରେ ମିଶ୍ରଣ',
        'K_low_0_name': 'MOP / ମ୍ୟୁରିଏଟ ଅଫ ପୋଟାଶ (60% K₂O)', 'K_low_0_dose': '20–25 କିଗ୍ରା/ଏକର', 'K_low_0_timing': 'ବୁଣିବା ବା ରୋପଣ ସମୟ ମୂଳ', 'K_low_0_method': 'ଛଟ ଓ ମିଶ୍ରଣ',
        'K_low_1_name': 'SOP / ସଲ୍ଫେଟ ଅଫ ପୋଟାଶ (50% K₂O)', 'K_low_1_dose': '25–30 କିଗ୍ରା/ଏକର', 'K_low_1_timing': 'ବିଭକ୍ତ: ମୂଳ + 45 ଦିନ ପରେ', 'K_low_1_method': 'ମୂଳ ପାଖ ସାଇଡ-ଡ୍ରେସ',
        'ph_high_0_name': 'ଜିପ୍ସମ (କ୍ୟାଲସିୟମ ସଲ୍ଫେଟ)', 'ph_high_0_dose': '100–200 କିଗ୍ରା/ଏକର', 'ph_high_0_timing': 'ବୁଣିବା 2–4 ସପ୍ତାହ ପୂର୍ବ', 'ph_high_0_method': 'ଛଟ ଓ ଭଲ ଭାବ ପାଣି ଦିଅ',
        'ph_high_1_name': 'ଏଲିମେଣ୍ଟାଲ ସଲ୍ଫର', 'ph_high_1_dose': '10–20 କିଗ୍ରା/ଏକର', 'ph_high_1_timing': 'ବୁଣିବା 1–2 ମାସ ପୂର୍ବ', 'ph_high_1_method': 'ଉପର 15 ସେଁ.ମି ମାଟିରେ ମିଶ୍ରଣ',
        'ph_low_0_name': 'କୃଷି ଚୁନ (CaCO₃)', 'ph_low_0_dose': '100–300 କିଗ୍ରା/ଏକର', 'ph_low_0_timing': 'ବୁଣିବା 4–6 ସପ୍ତାହ ପୂର୍ବ', 'ph_low_0_method': 'ସମାନ ଭାବ ଛଟ ଓ ହଳ',
        'ph_low_1_name': 'ଡୋଲୋମାଇଟ ଚୁନ', 'ph_low_1_dose': '100–200 କିଗ୍ରା/ଏକର', 'ph_low_1_timing': 'ବୁଣିବା 4–6 ସପ୍ତାହ ପୂର୍ବ', 'ph_low_1_method': 'ଛଟ ଓ ଗଭୀର ଭାବ ମିଶ୍ରଣ',
        'nutrient_N': 'ନାଇଟ୍ରୋଜେନ (N)', 'nutrient_P': 'ଫସ୍ଫରସ (P)', 'nutrient_K': 'ପୋଟାସିୟମ (K)', 'nutrient_ph': 'ମାଟି pH',
        'status_good': 'good', 'status_low': 'low', 'status_excess': 'excess', 'status_high': 'high',
        'dose_label': 'ମାତ୍ରା', 'timing_label': 'ସମୟ', 'method_label': 'ପ୍ରଣାଳୀ', 'unit': 'ମି.ଗ୍ରା/କି.ଗ୍ରା',
    },
}

def get_crop_name(crop, lang):
    """Get translated crop name, fallback to English."""
    return CROP_NAMES.get(lang, CROP_NAMES['English']).get(crop, CROP_NAMES['English'].get(crop, crop))

def get_fert(lang, key):
    """Get translated fertilizer string."""
    ft = FERT_TRANSLATIONS.get(lang, FERT_TRANSLATIONS['English'])
    return ft.get(key, FERT_TRANSLATIONS['English'].get(key, key))

def build_fertilizer_suggestions(crop, inputs, lang='English'):
    """Compare user's N, P, K, pH against crop profile and return translated suggestions."""
    profile = CROP_PROFILES.get(crop)
    if not profile:
        return []

    N  = float(inputs['N'])
    P  = float(inputs['P'])
    K  = float(inputs['K'])
    ph = float(inputs['ph'])
    unit = get_fert(lang, 'unit')

    def make_ferts(prefix, count):
        return [
            {
                'name':   get_fert(lang, f'{prefix}_{i}_name'),
                'dose':   get_fert(lang, f'{prefix}_{i}_dose'),
                'timing': get_fert(lang, f'{prefix}_{i}_timing'),
                'method': get_fert(lang, f'{prefix}_{i}_method'),
            }
            for i in range(count)
        ]

    suggestions = []

    # N check
    if N < profile['N'][0]:
        suggestions.append({'nutrient': get_fert(lang,'nutrient_N'), 'status': 'low', 'your_value': N,
            'ideal': f"{profile['N'][0]}–{profile['N'][1]} {unit}", 'fertilizers': make_ferts('N_low', 2)})
    elif N > profile['N'][1]:
        suggestions.append({'nutrient': get_fert(lang,'nutrient_N'), 'status': 'excess', 'your_value': N,
            'ideal': f"{profile['N'][0]}–{profile['N'][1]} {unit}", 'fertilizers': []})
    else:
        suggestions.append({'nutrient': get_fert(lang,'nutrient_N'), 'status': 'good', 'your_value': N,
            'ideal': f"{profile['N'][0]}–{profile['N'][1]} {unit}", 'fertilizers': []})

    # P check
    if P < profile['P'][0]:
        suggestions.append({'nutrient': get_fert(lang,'nutrient_P'), 'status': 'low', 'your_value': P,
            'ideal': f"{profile['P'][0]}–{profile['P'][1]} {unit}", 'fertilizers': make_ferts('P_low', 2)})
    elif P > profile['P'][1]:
        suggestions.append({'nutrient': get_fert(lang,'nutrient_P'), 'status': 'excess', 'your_value': P,
            'ideal': f"{profile['P'][0]}–{profile['P'][1]} {unit}", 'fertilizers': []})
    else:
        suggestions.append({'nutrient': get_fert(lang,'nutrient_P'), 'status': 'good', 'your_value': P,
            'ideal': f"{profile['P'][0]}–{profile['P'][1]} {unit}", 'fertilizers': []})

    # K check
    if K < profile['K'][0]:
        suggestions.append({'nutrient': get_fert(lang,'nutrient_K'), 'status': 'low', 'your_value': K,
            'ideal': f"{profile['K'][0]}–{profile['K'][1]} {unit}", 'fertilizers': make_ferts('K_low', 2)})
    elif K > profile['K'][1]:
        suggestions.append({'nutrient': get_fert(lang,'nutrient_K'), 'status': 'excess', 'your_value': K,
            'ideal': f"{profile['K'][0]}–{profile['K'][1]} {unit}", 'fertilizers': []})
    else:
        suggestions.append({'nutrient': get_fert(lang,'nutrient_K'), 'status': 'good', 'your_value': K,
            'ideal': f"{profile['K'][0]}–{profile['K'][1]} {unit}", 'fertilizers': []})

    # pH check
    if ph < profile['ph'][0]:
        suggestions.append({'nutrient': get_fert(lang,'nutrient_ph'), 'status': 'low', 'your_value': ph,
            'ideal': f"{profile['ph'][0]}–{profile['ph'][1]}", 'fertilizers': make_ferts('ph_low', 2)})
    elif ph > profile['ph'][1]:
        suggestions.append({'nutrient': get_fert(lang,'nutrient_ph'), 'status': 'high', 'your_value': ph,
            'ideal': f"{profile['ph'][0]}–{profile['ph'][1]}", 'fertilizers': make_ferts('ph_high', 2)})
    else:
        suggestions.append({'nutrient': get_fert(lang,'nutrient_ph'), 'status': 'good', 'your_value': ph,
            'ideal': f"{profile['ph'][0]}–{profile['ph'][1]}", 'fertilizers': []})

    return suggestions


# Translations for key phrases

TRANSLATIONS = {
    'English': {
        'key_reason':       "The conditions in your field are well-suited for {crop}.",
        'temp_high':        "The temperature ({val}°C) is a bit high for {crop}, which prefers {min}–{max}°C.",
        'temp_low':         "The temperature ({val}°C) is a bit low for {crop}, which grows best at {min}–{max}°C.",
        'temp_good':        "The temperature ({val}°C) is ideal for {crop}.",
        'rain_high':        "Rainfall ({val} mm) is higher than what {crop} needs ({min}–{max} mm) — ensure good drainage.",
        'rain_low':         "Rainfall ({val} mm) is lower than what {crop} needs ({min}–{max} mm) — irrigation will help.",
        'rain_good':        "Rainfall ({val} mm) is just right for {crop}.",
        'ph_high':          "Soil pH ({val}) is slightly alkaline for {crop} (ideal: {min}–{max}) — adding acidic compost can help.",
        'ph_low':           "Soil pH ({val}) is slightly acidic for {crop} (ideal: {min}–{max}) — liming the soil may help.",
        'ph_good':          "Soil pH ({val}) is within the ideal range for {crop}.",
        'n_low':            "Nitrogen ({val} kg/ha) is on the lower side — applying urea or compost can boost growth.",
        'conf_high':        "The model is {conf}% confident — your conditions are a strong match for {crop}.",
        'conf_mid':         "The model is {conf}% confident — most conditions suit {crop}, but a few factors are borderline.",
        'conf_low':         "The model is {conf}% confident — conditions partially match {crop}; careful management is advised.",
        'tip_drainage':     "Make sure your field has proper drainage to avoid waterlogging.",
        'tip_irrigation':   "Set up drip or flood irrigation to meet the water needs of {crop}.",
        'tip_general':      "Monitor soil moisture regularly and apply fertiliser as per your Soil Health Card.",
        # Yield query
        'yield_good':       "Your field conditions are well-suited for growing {crop} — expect a good yield.",
        'yield_moderate':   "Your field can support {crop}, but the yield may be moderate due to some suboptimal conditions.",
        'yield_poor':       "Growing {crop} in these conditions may be difficult and yield is likely to be low.",
        'yield_very_poor':  "These conditions are not suitable for {crop} — yield is expected to be very poor.",
        'risk_temp':        "The main risk is temperature — {crop} may experience heat or cold stress.",
        'risk_water':       "The main risk is water — {crop} needs careful irrigation or drainage management.",
        'risk_ph':          "The main risk is soil pH — {crop} may struggle to absorb nutrients at this pH level.",
        'risk_general':     "Monitor crop health closely throughout the growing season.",
        'alt_suggest':      "Consider growing {alt} instead, which is ranked #1 for your current conditions.",
        'alt_encourage':    "Your field is among the best suited for {crop} — you are on the right track!",
        'recommended_crop': 'Recommended Crop',
        'why_this_crop':    'Why this crop?',
        'yield_outlook_for':'Yield Outlook For',
        'yield_analysis':   'Yield Analysis',
        'match_label':      'How well your conditions match',
        'best_crop_label':  'Best crop for your field right now',
        'calculating':      'Calculating…',
        'analyzing':        'Analyzing…',
        'outlook_good':     'Good Yield Expected',
        'outlook_moderate': 'Moderate Yield Expected',
        'outlook_poor':     'Poor Yield Expected',
        'outlook_very_poor':'Very Poor Yield Expected',
        'fert_excess_note': 'Excess {nutrient} can harm the crop. Avoid nitrogen-heavy fertilizers this season and let levels deplete naturally.',
        'your_value':       'Your value',
        'ideal_for':        'Ideal for',
    },
    'Hindi': {
        'key_reason':       "आपके खेत की परिस्थितियाँ {crop} के लिए उपयुक्त हैं।",
        'temp_high':        "तापमान ({val}°C) {crop} के लिए थोड़ा अधिक है, जो {min}–{max}°C पसंद करता है।",
        'temp_low':         "तापमान ({val}°C) {crop} के लिए थोड़ा कम है, जो {min}–{max}°C में सबसे अच्छा उगता है।",
        'temp_good':        "तापमान ({val}°C) {crop} के लिए आदर्श है।",
        'rain_high':        "वर्षा ({val} मिमी) {crop} की जरूरत ({min}–{max} मिमी) से अधिक है — अच्छी जल निकासी सुनिश्चित करें।",
        'rain_low':         "वर्षा ({val} मिमी) {crop} की जरूरत ({min}–{max} मिमी) से कम है — सिंचाई से मदद मिलेगी।",
        'rain_good':        "वर्षा ({val} मिमी) {crop} के लिए बिल्कुल सही है।",
        'ph_high':          "मिट्टी का pH ({val}) {crop} के लिए थोड़ा क्षारीय है (आदर्श: {min}–{max}) — अम्लीय खाद डालें।",
        'ph_low':           "मिट्टी का pH ({val}) {crop} के लिए थोड़ा अम्लीय है (आदर्श: {min}–{max}) — चूना मिलाने से मदद मिलेगी।",
        'ph_good':          "मिट्टी का pH ({val}) {crop} के लिए आदर्श सीमा में है।",
        'n_low':            "नाइट्रोजन ({val} किग्रा/हेक्टेयर) कम है — यूरिया या खाद से वृद्धि बढ़ाएँ।",
        'conf_high':        "मॉडल {conf}% आश्वस्त है — आपकी परिस्थितियाँ {crop} के लिए बहुत उपयुक्त हैं।",
        'conf_mid':         "मॉडल {conf}% आश्वस्त है — अधिकांश परिस्थितियाँ {crop} के लिए ठीक हैं, कुछ सुधार की जरूरत है।",
        'conf_low':         "मॉडल {conf}% आश्वस्त है — परिस्थितियाँ आंशिक रूप से मेल खाती हैं; सावधानीपूर्वक प्रबंधन करें।",
        'tip_drainage':     "जलभराव से बचने के लिए खेत में अच्छी जल निकासी सुनिश्चित करें।",
        'tip_irrigation':   "{crop} की पानी की जरूरत पूरी करने के लिए ड्रिप या बाढ़ सिंचाई लगाएँ।",
        'tip_general':      "मिट्टी की नमी की नियमित जाँच करें और मृदा स्वास्थ्य कार्ड के अनुसार उर्वरक डालें।",
        'yield_good':       "आपके खेत की परिस्थितियाँ {crop} उगाने के लिए अच्छी हैं — अच्छी उपज की उम्मीद है।",
        'yield_moderate':   "आपका खेत {crop} के लिए ठीक है, लेकिन उपज मध्यम हो सकती है।",
        'yield_poor':       "इन परिस्थितियों में {crop} उगाना कठिन हो सकता है और उपज कम रह सकती है।",
        'yield_very_poor':  "ये परिस्थितियाँ {crop} के लिए उपयुक्त नहीं हैं — उपज बहुत कम रहने की संभावना है।",
        'risk_temp':        "मुख्य खतरा तापमान है — {crop} को गर्मी या ठंड का तनाव हो सकता है।",
        'risk_water':       "मुख्य खतरा पानी है — {crop} के लिए सिंचाई या जल निकासी पर ध्यान दें।",
        'risk_ph':          "मुख्य खतरा मिट्टी का pH है — इस pH स्तर पर {crop} को पोषक तत्व मिलने में कठिनाई हो सकती है।",
        'risk_general':     "पूरे उगाने के मौसम में फसल की सेहत पर नज़र रखें।",
        'alt_suggest':      "इसके बजाय {alt} उगाने पर विचार करें, जो आपकी वर्तमान परिस्थितियों के लिए #1 पर है।",
        'alt_encourage':    "आपका खेत {crop} के लिए सबसे उपयुक्त खेतों में से एक है — आप सही रास्ते पर हैं!",
        'recommended_crop': 'अनुशंसित फसल',
        'why_this_crop':    'यह फसल क्यों?',
        'yield_outlook_for':'उपज दृष्टिकोण',
        'yield_analysis':   'उपज विश्लेषण',
        'match_label':      'आपकी परिस्थितियाँ कितनी मेल खाती हैं',
        'best_crop_label':  'आपके खेत के लिए अभी सबसे अच्छी फसल',
        'calculating':      'गणना हो रही है…',
        'analyzing':        'विश्लेषण हो रहा है…',
        'outlook_good':     'अच्छी उपज अपेक्षित',
        'outlook_moderate': 'मध्यम उपज अपेक्षित',
        'outlook_poor':     'कम उपज अपेक्षित',
        'outlook_very_poor':'बहुत कम उपज अपेक्षित',
        'fert_excess_note': 'अधिक {nutrient} फसल को नुकसान पहुँचा सकता है। इस मौसम में नाइट्रोजन युक्त उर्वरक से बचें।',
        'your_value':       'आपका मान',
        'ideal_for':        'के लिए आदर्श',
    },
    'Telugu': {
        'key_reason':       "మీ పొలం పరిస్థితులు {crop} పెంపకానికి బాగా అనుకూలంగా ఉన్నాయి।",
        'temp_high':        "ఉష్ణోగ్రత ({val}°C) {crop}కు కొంచెం ఎక్కువగా ఉంది, ఇది {min}–{max}°C ఇష్టపడుతుంది।",
        'temp_low':         "ఉష్ణోగ్రత ({val}°C) {crop}కు కొంచెం తక్కువగా ఉంది, ఇది {min}–{max}°Cలో బాగా పెరుగుతుంది।",
        'temp_good':        "ఉష్ణోగ్రత ({val}°C) {crop}కు అనువైనది।",
        'rain_high':        "వర్షపాతం ({val} మిమీ) {crop}కు అవసరమైన దానికంటే ({min}–{max} మిమీ) ఎక్కువగా ఉంది — మంచి నీటి పారుదల నిర్వహించండి।",
        'rain_low':         "వర్షపాతం ({val} మిమీ) {crop}కు అవసరమైన దానికంటే ({min}–{max} మిమీ) తక్కువగా ఉంది — నీటిపారుదల సహాయపడుతుంది।",
        'rain_good':        "వర్షపాతం ({val} మిమీ) {crop}కు సరిగ్గా సరిపోతుంది।",
        'ph_high':          "నేల pH ({val}) {crop}కు కొంచెం క్షారంగా ఉంది (అనువైనది: {min}–{max}) — ఆమ్ల కంపోస్ట్ వేయండి।",
        'ph_low':           "నేల pH ({val}) {crop}కు కొంచెం ఆమ్లంగా ఉంది (అనువైనది: {min}–{max}) — సున్నం వేయడం సహాయపడవచ్చు।",
        'ph_good':          "నేల pH ({val}) {crop}కు అనువైన పరిధిలో ఉంది।",
        'n_low':            "నత్రజని ({val} కి.గ్రా/హె) తక్కువగా ఉంది — యూరియా లేదా కంపోస్ట్ వేయండి।",
        'conf_high':        "మోడల్ {conf}% నమ్మకంగా ఉంది — మీ పరిస్థితులు {crop}కు బాగా సరిపోతున్నాయి।",
        'conf_mid':         "మోడల్ {conf}% నమ్మకంగా ఉంది — చాలా పరిస్థితులు {crop}కు అనుకూలంగా ఉన్నాయి, కొన్ని మెరుగుపరచాలి।",
        'conf_low':         "మోడల్ {conf}% నమ్మకంగా ఉంది — పరిస్థితులు పాక్షికంగా సరిపోతున్నాయి; జాగ్రత్తగా నిర్వహించండి।",
        'tip_drainage':     "నీరు నిలబడకుండా పొలంలో మంచి నీటి పారుదల ఏర్పాటు చేయండి।",
        'tip_irrigation':   "{crop} నీటి అవసరాలు తీర్చడానికి బిందు లేదా వరద సేద్యం ఏర్పాటు చేయండి।",
        'tip_general':      "నేల తేమను క్రమం తప్పకుండా తనిఖీ చేసి, నేల ఆరోగ్య కార్డు ప్రకారం ఎరువులు వేయండి।",
        'yield_good':       "మీ పొలం పరిస్థితులు {crop} పెంచడానికి చాలా అనుకూలంగా ఉన్నాయి — మంచి దిగుబడి వస్తుంది।",
        'yield_moderate':   "మీ పొలం {crop}కు మద్దతు ఇవ్వగలదు, కానీ దిగుబడి మధ్యమంగా ఉండవచ్చు।",
        'yield_poor':       "ఈ పరిస్థితుల్లో {crop} పెంచడం కష్టం, దిగుబడి తక్కువగా ఉండవచ్చు।",
        'yield_very_poor':  "ఈ పరిస్థితులు {crop}కు అనుకూలంగా లేవు — దిగుబడి చాలా తక్కువగా ఉండవచ్చు।",
        'risk_temp':        "ప్రధాన ప్రమాదం ఉష్ణోగ్రత — {crop}కు వేడి లేదా చలి ఒత్తిడి కలగవచ్చు।",
        'risk_water':       "ప్రధాన ప్రమాదం నీరు — {crop}కు జాగ్రత్తగా నీటిపారుదల లేదా పారుదల నిర్వహించండి।",
        'risk_ph':          "ప్రధాన ప్రమాదం నేల pH — ఈ pH స్థాయిలో {crop} పోషకాలు తీసుకోవడంలో ఇబ్బంది పడవచ్చు।",
        'risk_general':     "పెంపకం మొత్తం వ్యవధిలో పంట ఆరోగ్యాన్ని జాగ్రత్తగా గమనించండి।",
        'alt_suggest':      "బదులుగా {alt} పెంచడం పరిగణించండి, ఇది మీ ప్రస్తుత పరిస్థితులకు #1గా ఉంది।",
        'alt_encourage':    "మీ పొలం {crop} పెంచడానికి అత్యుత్తమంగా ఉంది — మీరు సరైన మార్గంలో ఉన్నారు!",
        'recommended_crop': 'సిఫారసు చేసిన పంట',
        'why_this_crop':    'ఈ పంట ఎందుకు?',
        'yield_outlook_for':'దిగుబడి అంచనా',
        'yield_analysis':   'దిగుబడి విశ్లేషణ',
        'match_label':      'మీ పరిస్థితులు ఎంత సరిపోతున్నాయి',
        'best_crop_label':  'ఇప్పుడు మీ పొలానికి ఉత్తమ పంట',
        'calculating':      'లెక్కిస్తున్నాం…',
        'analyzing':        'విశ్లేషిస్తున్నాం…',
        'outlook_good':     'మంచి దిగుబడి వస్తుంది',
        'outlook_moderate': 'మధ్యమ దిగుబడి వస్తుంది',
        'outlook_poor':     'తక్కువ దిగుబడి వస్తుంది',
        'outlook_very_poor':'చాలా తక్కువ దిగుబడి వస్తుంది',
        'fert_excess_note': 'అధిక {nutrient} పంటకు హాని చేయవచ్చు. ఈ సీజన్‌లో నత్రజని ఎరువులు వేయకండి.',
        'your_value':       'మీ విలువ',
        'ideal_for':        'కు అనువైనది',
    },
    'Tamil': {
        'key_reason':       "உங்கள் வயல் நிலைமைகள் {crop} சாகுபடிக்கு மிகவும் ஏற்றதாக உள்ளன।",
        'temp_high':        "வெப்பநிலை ({val}°C) {crop}-க்கு சற்று அதிகமாக உள்ளது, இது {min}–{max}°C விரும்புகிறது।",
        'temp_low':         "வெப்பநிலை ({val}°C) {crop}-க்கு சற்று குறைவாக உள்ளது, இது {min}–{max}°C-ல் சிறப்பாக வளர்கிறது।",
        'temp_good':        "வெப்பநிலை ({val}°C) {crop}-க்கு ஏற்றதாக உள்ளது।",
        'rain_high':        "மழையளவு ({val} மிமீ) {crop}-க்கு தேவையானதை ({min}–{max} மிமீ) விட அதிகம் — நல்ல வடிகால் வசதி உறுதிசெய்யுங்கள்।",
        'rain_low':         "மழையளவு ({val} மிமீ) {crop}-க்கு தேவையானதை ({min}–{max} மிமீ) விட குறைவு — நீர்ப்பாசனம் உதவும்।",
        'rain_good':        "மழையளவு ({val} மிமீ) {crop}-க்கு சரியானது।",
        'ph_high':          "மண் pH ({val}) {crop}-க்கு சற்று காரத்தன்மை அதிகம் (ஏற்றது: {min}–{max}) — அமில உரம் சேர்க்கவும்।",
        'ph_low':           "மண் pH ({val}) {crop}-க்கு சற்று அமிலத்தன்மை அதிகம் (ஏற்றது: {min}–{max}) — சுண்ணாம்பு சேர்ப்பது உதவும்।",
        'ph_good':          "மண் pH ({val}) {crop}-க்கு ஏற்ற வரம்பில் உள்ளது।",
        'n_low':            "நைட்ரஜன் ({val} கி.கி/ஹெ) குறைவாக உள்ளது — யூரியா அல்லது உரம் இட்டு வளர்ச்சியை மேம்படுத்துங்கள்।",
        'conf_high':        "மாதிரி {conf}% நம்பிக்கையுடன் உள்ளது — உங்கள் நிலைமைகள் {crop}-க்கு மிகவும் ஏற்றவை।",
        'conf_mid':         "மாதிரி {conf}% நம்பிக்கையுடன் உள்ளது — பெரும்பாலான நிலைமைகள் {crop}-க்கு ஏற்றவை, சில சரிசெய்யப்படலாம்।",
        'conf_low':         "மாதிரி {conf}% நம்பிக்கையுடன் உள்ளது — நிலைமைகள் ஓரளவு பொருந்துகின்றன; கவனமான மேலாண்மை தேவை।",
        'tip_drainage':     "நீர் தேங்காமல் இருக்க வயலில் நல்ல வடிகால் வசதி உறுதிசெய்யுங்கள்।",
        'tip_irrigation':   "{crop}-இன் தண்ணீர் தேவையை பூர்த்திசெய்ய சொட்டு நீர் அல்லது வெள்ள நீர்ப்பாசனம் அமையுங்கள்।",
        'tip_general':      "மண் ஈரப்பதத்தை தொடர்ந்து கண்காணித்து, மண் சுகாதார அட்டையின்படி உரம் இடுங்கள்।",
        'yield_good':       "உங்கள் வயல் நிலைமைகள் {crop} சாகுபடிக்கு சிறப்பாக உள்ளன — நல்ல மகசூல் எதிர்பார்க்கலாம்।",
        'yield_moderate':   "உங்கள் வயல் {crop} சாகுபடியை ஆதரிக்கலாம், ஆனால் மகசூல் மிதமாக இருக்கலாம்।",
        'yield_poor':       "இந்த நிலைமைகளில் {crop} வளர்ப்பது கடினமாக இருக்கலாம், மகசூல் குறைவாக இருக்கலாம்।",
        'yield_very_poor':  "இந்த நிலைமைகள் {crop}-க்கு ஏற்றதாக இல்லை — மகசூல் மிகவும் குறைவாக இருக்கலாம்।",
        'risk_temp':        "முக்கிய ஆபத்து வெப்பநிலை — {crop}-க்கு வெப்பம் அல்லது குளிர் அழுத்தம் ஏற்படலாம்।",
        'risk_water':       "முக்கிய ஆபத்து நீர் — {crop}-க்கு கவனமான நீர்ப்பாசனம் அல்லது வடிகால் மேலாண்மை தேவை।",
        'risk_ph':          "முக்கிய ஆபத்து மண் pH — இந்த pH அளவில் {crop} ஊட்டச்சத்துகளை உறிஞ்சுவதில் சிரமம் ஏற்படலாம்।",
        'risk_general':     "வளர்ப்பு காலம் முழுவதும் பயிரின் ஆரோக்கியத்தை கவனமாக கண்காணியுங்கள்।",
        'alt_suggest':      "பதிலாக {alt} வளர்ப்பதை கருத்தில் கொள்ளுங்கள், இது உங்கள் தற்போதைய நிலைமைகளுக்கு #1ஆக உள்ளது।",
        'alt_encourage':    "உங்கள் வயல் {crop} சாகுபடிக்கு மிகவும் ஏற்றதாக உள்ளது — நீங்கள் சரியான பாதையில் இருக்கிறீர்கள்!",
        'recommended_crop': 'பரிந்துரைக்கப்பட்ட பயிர்',
        'why_this_crop':    'இந்த பயிர் ஏன்?',
        'yield_outlook_for':'மகசூல் கணிப்பு',
        'yield_analysis':   'மகசூல் பகுப்பாய்வு',
        'match_label':      'உங்கள் நிலைமைகள் எவ்வளவு பொருந்துகின்றன',
        'best_crop_label':  'இப்போது உங்கள் வயலுக்கு சிறந்த பயிர்',
        'calculating':      'கணக்கிடுகிறோம்…',
        'analyzing':        'பகுப்பாய்கிறோம்…',
        'outlook_good':     'நல்ல மகசூல் எதிர்பார்க்கலாம்',
        'outlook_moderate': 'மிதமான மகசூல் எதிர்பார்க்கலாம்',
        'outlook_poor':     'குறைந்த மகசூல் எதிர்பார்க்கலாம்',
        'outlook_very_poor':'மிகவும் குறைந்த மகசூல் எதிர்பார்க்கலாம்',
        'fert_excess_note': 'அதிகமான {nutrient} பயிரை பாதிக்கும். இந்த சீசனில் நைட்ரஜன் உரங்களை தவிர்க்கவும்.',
        'your_value':       'உங்கள் மதிப்பு',
        'ideal_for':        'க்கு ஏற்றது',
    },
    'Kannada': {
        'key_reason':       "ನಿಮ್ಮ ಹೊಲದ ಪರಿಸ್ಥಿತಿಗಳು {crop} ಬೆಳೆಗೆ ಉತ್ತಮವಾಗಿ ಸೂಕ್ತವಾಗಿವೆ।",
        'temp_high':        "ತಾಪಮಾನ ({val}°C) {crop}ಗೆ ಸ್ವಲ್ಪ ಹೆಚ್ಚಾಗಿದೆ, ಇದು {min}–{max}°C ಇಷ್ಟಪಡುತ್ತದೆ।",
        'temp_low':         "ತಾಪಮಾನ ({val}°C) {crop}ಗೆ ಸ್ವಲ್ಪ ಕಡಿಮೆಯಾಗಿದೆ, ಇದು {min}–{max}°Cಯಲ್ಲಿ ಉತ್ತಮವಾಗಿ ಬೆಳೆಯುತ್ತದೆ।",
        'temp_good':        "ತಾಪಮಾನ ({val}°C) {crop}ಗೆ ಸೂಕ್ತವಾಗಿದೆ।",
        'rain_high':        "ಮಳೆ ({val} ಮಿಮೀ) {crop}ಗೆ ಬೇಕಾದ ಪ್ರಮಾಣಕ್ಕಿಂತ ({min}–{max} ಮಿಮೀ) ಹೆಚ್ಚಾಗಿದೆ — ಉತ್ತಮ ಒಳಚರಂಡಿ ನಿರ್ಮಿಸಿ।",
        'rain_low':         "ಮಳೆ ({val} ಮಿಮೀ) {crop}ಗೆ ಬೇಕಾದ ಪ್ರಮಾಣಕ್ಕಿಂತ ({min}–{max} ಮಿಮೀ) ಕಡಿಮೆ — ನೀರಾವರಿ ಸಹಾಯ ಮಾಡುತ್ತದೆ।",
        'rain_good':        "ಮಳೆ ({val} ಮಿಮೀ) {crop}ಗೆ ಸರಿಯಾಗಿದೆ।",
        'ph_high':          "ಮಣ್ಣಿನ pH ({val}) {crop}ಗೆ ಸ್ವಲ್ಪ ಕ್ಷಾರೀಯವಾಗಿದೆ (ಸೂಕ್ತ: {min}–{max}) — ಆಮ್ಲ ಗೊಬ್ಬರ ಹಾಕಿ।",
        'ph_low':           "ಮಣ್ಣಿನ pH ({val}) {crop}ಗೆ ಸ್ವಲ್ಪ ಆಮ್ಲೀಯವಾಗಿದೆ (ಸೂಕ್ತ: {min}–{max}) — ಸುಣ್ಣ ಹಾಕುವುದು ಸಹಾಯ ಮಾಡಬಹುದು।",
        'ph_good':          "ಮಣ್ಣಿನ pH ({val}) {crop}ಗೆ ಸೂಕ್ತ ವ್ಯಾಪ್ತಿಯಲ್ಲಿದೆ।",
        'n_low':            "ಸಾರಜನಕ ({val} ಕಿ.ಗ್ರಾ/ಹೆ) ಕಡಿಮೆ ಇದೆ — ಯೂರಿಯಾ ಅಥವಾ ಗೊಬ್ಬರ ಹಾಕಿ ಬೆಳವಣಿಗೆ ಹೆಚ್ಚಿಸಿ।",
        'conf_high':        "ಮಾದರಿ {conf}% ವಿಶ್ವಾಸದಿಂದ ಇದೆ — ನಿಮ್ಮ ಪರಿಸ್ಥಿತಿಗಳು {crop}ಗೆ ಬಲವಾದ ಹೊಂದಾಣಿಕೆ।",
        'conf_mid':         "ಮಾದರಿ {conf}% ವಿಶ್ವಾಸದಿಂದ ಇದೆ — ಹೆಚ್ಚಿನ ಪರಿಸ್ಥಿತಿಗಳು {crop}ಗೆ ಸೂಕ್ತ, ಕೆಲವು ಸುಧಾರಣೆ ಬೇಕು।",
        'conf_low':         "ಮಾದರಿ {conf}% ವಿಶ್ವಾಸದಿಂದ ಇದೆ — ಪರಿಸ್ಥಿತಿಗಳು ಭಾಗಶಃ ಹೊಂದುತ್ತವೆ; ಜಾಗರೂಕ ನಿರ್ವಹಣೆ ಅಗತ್ಯ।",
        'tip_drainage':     "ನೀರು ನಿಲ್ಲದಂತೆ ಹೊಲದಲ್ಲಿ ಉತ್ತಮ ಒಳಚರಂಡಿ ಮಾಡಿ।",
        'tip_irrigation':   "{crop}ದ ನೀರಿನ ಅಗತ್ಯ ಪೂರೈಸಲು ಹನಿ ನೀರಾವರಿ ಅಥವಾ ಪ್ರವಾಹ ನೀರಾವರಿ ಹಾಕಿ।",
        'tip_general':      "ಮಣ್ಣಿನ ತೇವಾಂಶ ನಿಯಮಿತವಾಗಿ ಪರೀಕ್ಷಿಸಿ ಮತ್ತು ಮಣ್ಣು ಆರೋಗ್ಯ ಕಾರ್ಡ್ ಪ್ರಕಾರ ಗೊಬ್ಬರ ಹಾಕಿ।",
        'yield_good':       "ನಿಮ್ಮ ಹೊಲದ ಪರಿಸ್ಥಿತಿಗಳು {crop} ಬೆಳೆಯಲು ಉತ್ತಮವಾಗಿ ಸೂಕ್ತ — ಉತ್ತಮ ಇಳುವರಿ ನಿರೀಕ್ಷಿಸಬಹುದು।",
        'yield_moderate':   "ನಿಮ್ಮ ಹೊಲ {crop} ಬೆಳೆಯಲು ಬೆಂಬಲ ನೀಡಬಲ್ಲದು, ಆದರೆ ಇಳುವರಿ ಮಧ್ಯಮ ಆಗಿರಬಹುದು।",
        'yield_poor':       "ಈ ಪರಿಸ್ಥಿತಿಗಳಲ್ಲಿ {crop} ಬೆಳೆಯುವುದು ಕಷ್ಟ ಮತ್ತು ಇಳುವರಿ ಕಡಿಮೆ ಆಗಿರಬಹುದು।",
        'yield_very_poor':  "ಈ ಪರಿಸ್ಥಿತಿಗಳು {crop}ಗೆ ಸೂಕ್ತವಲ್ಲ — ಇಳುವರಿ ತುಂಬಾ ಕಡಿಮೆ ಆಗಿರಬಹುದು।",
        'risk_temp':        "ಮುಖ್ಯ ಅಪಾಯ ತಾಪಮಾನ — {crop}ಗೆ ಶಾಖ ಅಥವಾ ಚಳಿ ಒತ್ತಡ ಉಂಟಾಗಬಹುದು।",
        'risk_water':       "ಮುಖ್ಯ ಅಪಾಯ ನೀರು — {crop}ಗೆ ಜಾಗರೂಕ ನೀರಾವರಿ ಅಥವಾ ಒಳಚರಂಡಿ ನಿರ್ವಹಣೆ ಬೇಕು।",
        'risk_ph':          "ಮುಖ್ಯ ಅಪಾಯ ಮಣ್ಣಿನ pH — ಈ pH ಮಟ್ಟದಲ್ಲಿ {crop} ಪೋಷಕಾಂಶ ಹೀರಿಕೊಳ್ಳಲು ತೊಂದರೆ ಆಗಬಹುದು।",
        'risk_general':     "ಬೆಳೆ ಋತುವಿನಾದ್ಯಂತ ಬೆಳೆಯ ಆರೋಗ್ಯ ಗಮನಿಸಿ।",
        'alt_suggest':      "ಬದಲಾಗಿ {alt} ಬೆಳೆಯಲು ಯೋಚಿಸಿ, ಇದು ನಿಮ್ಮ ಪ್ರಸ್ತುತ ಪರಿಸ್ಥಿತಿಗಳಿಗೆ #1 ಆಗಿದೆ।",
        'alt_encourage':    "ನಿಮ್ಮ ಹೊಲ {crop} ಬೆಳೆಯಲು ಅತ್ಯುತ್ತಮ ಹೊಲಗಳಲ್ಲಿ ಒಂದು — ನೀವು ಸರಿಯಾದ ಹಾದಿಯಲ್ಲಿದ್ದೀರಿ!",
        'recommended_crop': 'ಶಿಫಾರಸು ಮಾಡಿದ ಬೆಳೆ',
        'why_this_crop':    'ಈ ಬೆಳೆ ಏಕೆ?',
        'yield_outlook_for':'ಇಳುವರಿ ನಿರೀಕ್ಷೆ',
        'yield_analysis':   'ಇಳುವರಿ ವಿಶ್ಲೇಷಣೆ',
        'match_label':      'ನಿಮ್ಮ ಪರಿಸ್ಥಿತಿಗಳು ಎಷ್ಟು ಹೊಂದುತ್ತವೆ',
        'best_crop_label':  'ಈಗ ನಿಮ್ಮ ಹೊಲಕ್ಕೆ ಉತ್ತಮ ಬೆಳೆ',
        'calculating':      'ಲೆಕ್ಕ ಮಾಡಲಾಗುತ್ತಿದೆ…',
        'analyzing':        'ವಿಶ್ಲೇಷಿಸಲಾಗುತ್ತಿದೆ…',
        'outlook_good':     'ಉತ್ತಮ ಇಳುವರಿ ನಿರೀಕ್ಷಿತ',
        'outlook_moderate': 'ಮಧ್ಯಮ ಇಳುವರಿ ನಿರೀಕ್ಷಿತ',
        'outlook_poor':     'ಕಡಿಮೆ ಇಳುವರಿ ನಿರೀಕ್ಷಿತ',
        'outlook_very_poor':'ತುಂಬಾ ಕಡಿಮೆ ಇಳುವರಿ ನಿರೀಕ್ಷಿತ',
        'fert_excess_note': 'ಹೆಚ್ಚು {nutrient} ಬೆಳೆಗೆ ಹಾನಿಕಾರಕ. ಈ ಋತುವಿನಲ್ಲಿ ಸಾರಜನಕ ಗೊಬ್ಬರ ಹಾಕಬೇಡಿ.',
        'your_value':       'ನಿಮ್ಮ ಮೌಲ್ಯ',
        'ideal_for':        'ಗೆ ಸೂಕ್ತ',
    },
    'Marathi': {
        'key_reason':       "तुमच्या शेताच्या परिस्थिती {crop} साठी चांगल्या प्रकारे अनुकूल आहेत।",
        'temp_high':        "तापमान ({val}°C) {crop} साठी थोडे जास्त आहे, जे {min}–{max}°C पसंत करते।",
        'temp_low':         "तापमान ({val}°C) {crop} साठी थोडे कमी आहे, जे {min}–{max}°C मध्ये सर्वोत्तम वाढते।",
        'temp_good':        "तापमान ({val}°C) {crop} साठी आदर्श आहे।",
        'rain_high':        "पाऊस ({val} मिमी) {crop} ला लागणाऱ्यापेक्षा ({min}–{max} मिमी) जास्त आहे — चांगली निचरा व्यवस्था ठेवा।",
        'rain_low':         "पाऊस ({val} मिमी) {crop} ला लागणाऱ्यापेक्षा ({min}–{max} मिमी) कमी आहे — सिंचन मदत करेल।",
        'rain_good':        "पाऊस ({val} मिमी) {crop} साठी अगदी योग्य आहे।",
        'ph_high':          "मातीचा pH ({val}) {crop} साठी थोडा क्षारीय आहे (आदर्श: {min}–{max}) — आम्लयुक्त कंपोस्ट घाला।",
        'ph_low':           "मातीचा pH ({val}) {crop} साठी थोडा आम्लीय आहे (आदर्श: {min}–{max}) — चुना टाकणे मदत करू शकते।",
        'ph_good':          "मातीचा pH ({val}) {crop} साठी आदर्श श्रेणीत आहे।",
        'n_low':            "नायट्रोजन ({val} किग्रा/हेक्टर) कमी आहे — युरिया किंवा कंपोस्ट घालून वाढ सुधारा।",
        'conf_high':        "मॉडेल {conf}% विश्वासाने सांगतो — तुमच्या परिस्थिती {crop} साठी उत्तम जुळतात।",
        'conf_mid':         "मॉडेल {conf}% विश्वासाने सांगतो — बहुतांश परिस्थिती {crop} साठी योग्य आहेत, काही सुधारणा आवश्यक।",
        'conf_low':         "मॉडेल {conf}% विश्वासाने सांगतो — परिस्थिती अंशतः जुळतात; काळजीपूर्वक व्यवस्थापन करा।",
        'tip_drainage':     "पाणी साचू नये म्हणून शेतात चांगली निचरा व्यवस्था करा।",
        'tip_irrigation':   "{crop} च्या पाण्याच्या गरजा पूर्ण करण्यासाठी ठिबक किंवा पूर सिंचन लावा।",
        'tip_general':      "मातीतील ओलावा नियमितपणे तपासा आणि माती आरोग्य कार्डनुसार खत द्या।",
        'yield_good':       "तुमच्या शेताच्या परिस्थिती {crop} वाढवण्यासाठी चांगल्या आहेत — चांगले उत्पादन अपेक्षित आहे।",
        'yield_moderate':   "तुमचे शेत {crop} ला आधार देऊ शकते, परंतु उत्पादन मध्यम असू शकते।",
        'yield_poor':       "या परिस्थितींमध्ये {crop} वाढवणे कठीण असू शकते आणि उत्पादन कमी राहू शकते।",
        'yield_very_poor':  "या परिस्थिती {crop} साठी अनुकूल नाहीत — उत्पादन खूपच कमी असण्याची शक्यता आहे।",
        'risk_temp':        "मुख्य धोका तापमान आहे — {crop} ला उष्णता किंवा थंडीचा ताण येऊ शकतो।",
        'risk_water':       "मुख्य धोका पाणी आहे — {crop} साठी काळजीपूर्वक सिंचन किंवा निचरा व्यवस्थापन करा।",
        'risk_ph':          "मुख्य धोका मातीचा pH आहे — या pH पातळीवर {crop} ला पोषक तत्त्वे शोषण्यात अडचण येऊ शकते।",
        'risk_general':     "संपूर्ण हंगामात पिकाच्या आरोग्यावर बारकाईने लक्ष ठेवा।",
        'alt_suggest':      "त्याऐवजी {alt} वाढवण्याचा विचार करा, जे तुमच्या सध्याच्या परिस्थितीसाठी #1 आहे।",
        'alt_encourage':    "तुमचे शेत {crop} वाढवण्यासाठी सर्वोत्तम शेतांपैकी एक आहे — तुम्ही योग्य मार्गावर आहात!",
        'recommended_crop': 'शिफारस केलेले पीक',
        'why_this_crop':    'हे पीक का?',
        'yield_outlook_for':'उत्पादन दृष्टिकोन',
        'yield_analysis':   'उत्पादन विश्लेषण',
        'match_label':      'तुमच्या परिस्थिती किती जुळतात',
        'best_crop_label':  'सध्या तुमच्या शेतासाठी सर्वोत्तम पीक',
        'calculating':      'मोजत आहे…',
        'analyzing':        'विश्लेषण करत आहे…',
        'outlook_good':     'चांगले उत्पादन अपेक्षित',
        'outlook_moderate': 'मध्यम उत्पादन अपेक्षित',
        'outlook_poor':     'कमी उत्पादन अपेक्षित',
        'outlook_very_poor':'खूप कमी उत्पादन अपेक्षित',
        'fert_excess_note': 'जास्त {nutrient} पिकाला हानी पोहोचवू शकते. या हंगामात नायट्रोजनयुक्त खत टाळा.',
        'your_value':       'तुमचे मूल्य',
        'ideal_for':        'साठी आदर्श',
    },
    'Bengali': {
        'key_reason':       "আপনার জমির পরিস্থিতি {crop} চাষের জন্য বেশ উপযুক্ত।",
        'temp_high':        "তাপমাত্রা ({val}°C) {crop}-এর জন্য একটু বেশি, যা {min}–{max}°C পছন্দ করে।",
        'temp_low':         "তাপমাত্রা ({val}°C) {crop}-এর জন্য একটু কম, যা {min}–{max}°C-এ সবচেয়ে ভালো জন্মায়।",
        'temp_good':        "তাপমাত্রা ({val}°C) {crop}-এর জন্য আদর্শ।",
        'rain_high':        "বৃষ্টিপাত ({val} মিমি) {crop}-এর প্রয়োজনের ({min}–{max} মিমি) চেয়ে বেশি — ভালো নিষ্কাশন নিশ্চিত করুন।",
        'rain_low':         "বৃষ্টিপাত ({val} মিমি) {crop}-এর প্রয়োজনের ({min}–{max} মিমি) চেয়ে কম — সেচ সাহায্য করবে।",
        'rain_good':        "বৃষ্টিপাত ({val} মিমি) {crop}-এর জন্য একদম সঠিক।",
        'ph_high':          "মাটির pH ({val}) {crop}-এর জন্য একটু ক্ষারীয় (আদর্শ: {min}–{max}) — অম্লীয় সার যোগ করুন।",
        'ph_low':           "মাটির pH ({val}) {crop}-এর জন্য একটু অম্লীয় (আদর্শ: {min}–{max}) — চুন মেশানো সাহায্য করতে পারে।",
        'ph_good':          "মাটির pH ({val}) {crop}-এর জন্য আদর্শ সীমার মধ্যে।",
        'n_low':            "নাইট্রোজেন ({val} কেজি/হেক্টর) কম — ইউরিয়া বা সার দিয়ে বৃদ্ধি বাড়ান।",
        'conf_high':        "মডেল {conf}% আত্মবিশ্বাসী — আপনার পরিস্থিতি {crop}-এর জন্য শক্তিশালী মিল।",
        'conf_mid':         "মডেল {conf}% আত্মবিশ্বাসী — বেশিরভাগ পরিস্থিতি {crop}-এর জন্য উপযুক্ত, কিছু উন্নতি দরকার।",
        'conf_low':         "মডেল {conf}% আত্মবিশ্বাসী — পরিস্থিতি আংশিকভাবে মেলে; সতর্ক ব্যবস্থাপনা প্রয়োজন।",
        'tip_drainage':     "জলাবদ্ধতা এড়াতে জমিতে ভালো নিষ্কাশন নিশ্চিত করুন।",
        'tip_irrigation':   "{crop}-এর জলের চাহিদা পূরণ করতে ড্রিপ বা বন্যা সেচ স্থাপন করুন।",
        'tip_general':      "নিয়মিত মাটির আর্দ্রতা পরীক্ষা করুন এবং মৃত্তিকা স্বাস্থ্য কার্ড অনুযায়ী সার দিন।",
        'yield_good':       "আপনার জমির পরিস্থিতি {crop} চাষের জন্য ভালো — ভালো ফলন আশা করা যায়।",
        'yield_moderate':   "আপনার জমি {crop} সমর্থন করতে পারে, তবে ফলন মাঝারি হতে পারে।",
        'yield_poor':       "এই পরিস্থিতিতে {crop} চাষ কঠিন হতে পারে এবং ফলন কম হতে পারে।",
        'yield_very_poor':  "এই পরিস্থিতি {crop}-এর জন্য উপযুক্ত নয় — ফলন খুব কম হওয়ার সম্ভাবনা।",
        'risk_temp':        "প্রধান ঝুঁকি তাপমাত্রা — {crop}-এ তাপ বা ঠান্ডার চাপ হতে পারে।",
        'risk_water':       "প্রধান ঝুঁকি পানি — {crop}-এর জন্য সতর্ক সেচ বা নিষ্কাশন ব্যবস্থাপনা দরকার।",
        'risk_ph':          "প্রধান ঝুঁকি মাটির pH — এই pH স্তরে {crop} পুষ্টি শোষণে সমস্যা হতে পারে।",
        'risk_general':     "সমগ্র চাষ মৌসুমে ফসলের স্বাস্থ্য মনোযোগ দিয়ে পর্যবেক্ষণ করুন।",
        'alt_suggest':      "পরিবর্তে {alt} চাষ করার কথা বিবেচনা করুন, যা আপনার বর্তমান পরিস্থিতির জন্য #1।",
        'alt_encourage':    "আপনার জমি {crop} চাষের জন্য সেরা জমিগুলির মধ্যে একটি — আপনি সঠিক পথে আছেন!",
        'recommended_crop': 'প্রস্তাবিত ফসল',
        'why_this_crop':    'এই ফসল কেন?',
        'yield_outlook_for':'ফলন পূর্বাভাস',
        'yield_analysis':   'ফলন বিশ্লেষণ',
        'match_label':      'আপনার পরিস্থিতি কতটা মেলে',
        'best_crop_label':  'এখন আপনার মাঠের জন্য সেরা ফসল',
        'calculating':      'গণনা করা হচ্ছে…',
        'analyzing':        'বিশ্লেষণ করা হচ্ছে…',
        'outlook_good':     'ভালো ফলন প্রত্যাশিত',
        'outlook_moderate': 'মাঝারি ফলন প্রত্যাশিত',
        'outlook_poor':     'কম ফলন প্রত্যাশিত',
        'outlook_very_poor':'অত্যন্ত কম ফলন প্রত্যাশিত',
        'fert_excess_note': 'অতিরিক্ত {nutrient} ফসলের ক্ষতি করতে পারে। এই মৌসুমে নাইট্রোজেন সার এড়িয়ে চলুন।',
        'your_value':       'আপনার মান',
        'ideal_for':        'এর জন্য আদর্শ',
    },
    'Punjabi': {
        'key_reason':       "ਤੁਹਾਡੇ ਖੇਤ ਦੀਆਂ ਹਾਲਤਾਂ {crop} ਲਈ ਬਹੁਤ ਢੁਕਵੀਆਂ ਹਨ।",
        'temp_high':        "ਤਾਪਮਾਨ ({val}°C) {crop} ਲਈ ਥੋੜ੍ਹਾ ਜ਼ਿਆਦਾ ਹੈ, ਜੋ {min}–{max}°C ਪਸੰਦ ਕਰਦਾ ਹੈ।",
        'temp_low':         "ਤਾਪਮਾਨ ({val}°C) {crop} ਲਈ ਥੋੜ੍ਹਾ ਘੱਟ ਹੈ, ਜੋ {min}–{max}°C ਵਿੱਚ ਸਭ ਤੋਂ ਵਧੀਆ ਉੱਗਦਾ ਹੈ।",
        'temp_good':        "ਤਾਪਮਾਨ ({val}°C) {crop} ਲਈ ਆਦਰਸ਼ ਹੈ।",
        'rain_high':        "ਬਾਰਿਸ਼ ({val} ਮਿਮੀ) {crop} ਦੀ ਲੋੜ ({min}–{max} ਮਿਮੀ) ਨਾਲੋਂ ਵੱਧ ਹੈ — ਚੰਗੀ ਨਿਕਾਸੀ ਯਕੀਨੀ ਕਰੋ।",
        'rain_low':         "ਬਾਰਿਸ਼ ({val} ਮਿਮੀ) {crop} ਦੀ ਲੋੜ ({min}–{max} ਮਿਮੀ) ਨਾਲੋਂ ਘੱਟ ਹੈ — ਸਿੰਚਾਈ ਮਦਦ ਕਰੇਗੀ।",
        'rain_good':        "ਬਾਰਿਸ਼ ({val} ਮਿਮੀ) {crop} ਲਈ ਬਿਲਕੁਲ ਸਹੀ ਹੈ।",
        'ph_high':          "ਮਿੱਟੀ ਦਾ pH ({val}) {crop} ਲਈ ਥੋੜ੍ਹਾ ਖਾਰਾ ਹੈ (ਆਦਰਸ਼: {min}–{max}) — ਤੇਜ਼ਾਬੀ ਖਾਦ ਪਾਓ।",
        'ph_low':           "ਮਿੱਟੀ ਦਾ pH ({val}) {crop} ਲਈ ਥੋੜ੍ਹਾ ਤੇਜ਼ਾਬੀ ਹੈ (ਆਦਰਸ਼: {min}–{max}) — ਚੂਨਾ ਮਿਲਾਉਣਾ ਮਦਦ ਕਰ ਸਕਦਾ ਹੈ।",
        'ph_good':          "ਮਿੱਟੀ ਦਾ pH ({val}) {crop} ਲਈ ਆਦਰਸ਼ ਸੀਮਾ ਵਿੱਚ ਹੈ।",
        'n_low':            "ਨਾਈਟ੍ਰੋਜਨ ({val} ਕਿਗ੍ਰਾ/ਹੈਕਟੇਅਰ) ਘੱਟ ਹੈ — ਯੂਰੀਆ ਜਾਂ ਖਾਦ ਪਾ ਕੇ ਵਾਧਾ ਵਧਾਓ।",
        'conf_high':        "ਮਾਡਲ {conf}% ਭਰੋਸੇ ਨਾਲ ਦੱਸਦਾ ਹੈ — ਤੁਹਾਡੀਆਂ ਹਾਲਤਾਂ {crop} ਲਈ ਮਜ਼ਬੂਤ ਮੇਲ ਖਾਂਦੀਆਂ ਹਨ।",
        'conf_mid':         "ਮਾਡਲ {conf}% ਭਰੋਸੇ ਨਾਲ ਦੱਸਦਾ ਹੈ — ਜ਼ਿਆਦਾਤਰ ਹਾਲਤਾਂ {crop} ਲਈ ਢੁਕਵੀਆਂ ਹਨ, ਕੁਝ ਸੁਧਾਰ ਦੀ ਲੋੜ।",
        'conf_low':         "ਮਾਡਲ {conf}% ਭਰੋਸੇ ਨਾਲ ਦੱਸਦਾ ਹੈ — ਹਾਲਤਾਂ ਅੰਸ਼ਕ ਤੌਰ 'ਤੇ ਮੇਲ ਖਾਂਦੀਆਂ ਹਨ; ਧਿਆਨ ਨਾਲ ਪ੍ਰਬੰਧਨ ਕਰੋ।",
        'tip_drainage':     "ਪਾਣੀ ਖੜ੍ਹਾ ਹੋਣ ਤੋਂ ਬਚਾਉਣ ਲਈ ਖੇਤ ਵਿੱਚ ਚੰਗੀ ਨਿਕਾਸੀ ਯਕੀਨੀ ਕਰੋ।",
        'tip_irrigation':   "{crop} ਦੀਆਂ ਪਾਣੀ ਦੀਆਂ ਲੋੜਾਂ ਪੂਰੀਆਂ ਕਰਨ ਲਈ ਤੁਪਕਾ ਜਾਂ ਹੜ੍ਹ ਸਿੰਚਾਈ ਲਗਾਓ।",
        'tip_general':      "ਮਿੱਟੀ ਦੀ ਨਮੀ ਨਿਯਮਿਤ ਤੌਰ 'ਤੇ ਜਾਂਚੋ ਅਤੇ ਮਿੱਟੀ ਸਿਹਤ ਕਾਰਡ ਅਨੁਸਾਰ ਖਾਦ ਪਾਓ।",
        'yield_good':       "ਤੁਹਾਡੇ ਖੇਤ ਦੀਆਂ ਹਾਲਤਾਂ {crop} ਉਗਾਉਣ ਲਈ ਵਧੀਆ ਹਨ — ਚੰਗੀ ਪੈਦਾਵਾਰ ਦੀ ਉਮੀਦ ਕਰੋ।",
        'yield_moderate':   "ਤੁਹਾਡਾ ਖੇਤ {crop} ਦਾ ਸਮਰਥਨ ਕਰ ਸਕਦਾ ਹੈ, ਪਰ ਪੈਦਾਵਾਰ ਦਰਮਿਆਨੀ ਹੋ ਸਕਦੀ ਹੈ।",
        'yield_poor':       "ਇਹਨਾਂ ਹਾਲਤਾਂ ਵਿੱਚ {crop} ਉਗਾਉਣਾ ਔਖਾ ਹੋ ਸਕਦਾ ਹੈ ਅਤੇ ਪੈਦਾਵਾਰ ਘੱਟ ਰਹਿ ਸਕਦੀ ਹੈ।",
        'yield_very_poor':  "ਇਹ ਹਾਲਤਾਂ {crop} ਲਈ ਅਨੁਕੂਲ ਨਹੀਂ ਹਨ — ਪੈਦਾਵਾਰ ਬਹੁਤ ਘੱਟ ਹੋਣ ਦੀ ਸੰਭਾਵਨਾ ਹੈ।",
        'risk_temp':        "ਮੁੱਖ ਖ਼ਤਰਾ ਤਾਪਮਾਨ ਹੈ — {crop} ਨੂੰ ਗਰਮੀ ਜਾਂ ਠੰਡ ਦਾ ਦਬਾਅ ਹੋ ਸਕਦਾ ਹੈ।",
        'risk_water':       "ਮੁੱਖ ਖ਼ਤਰਾ ਪਾਣੀ ਹੈ — {crop} ਲਈ ਧਿਆਨ ਨਾਲ ਸਿੰਚਾਈ ਜਾਂ ਨਿਕਾਸੀ ਪ੍ਰਬੰਧਨ ਕਰੋ।",
        'risk_ph':          "ਮੁੱਖ ਖ਼ਤਰਾ ਮਿੱਟੀ ਦਾ pH ਹੈ — ਇਸ pH ਪੱਧਰ 'ਤੇ {crop} ਨੂੰ ਪੋਸ਼ਕ ਤੱਤ ਸੋਖਣ ਵਿੱਚ ਮੁਸ਼ਕਲ ਹੋ ਸਕਦੀ ਹੈ।",
        'risk_general':     "ਸਾਰੇ ਉਗਾਉਣ ਦੇ ਮੌਸਮ ਵਿੱਚ ਫ਼ਸਲ ਦੀ ਸਿਹਤ ਧਿਆਨ ਨਾਲ ਦੇਖੋ।",
        'alt_suggest':      "ਇਸ ਦੀ ਬਜਾਏ {alt} ਉਗਾਉਣ ਬਾਰੇ ਸੋਚੋ, ਜੋ ਤੁਹਾਡੀਆਂ ਮੌਜੂਦਾ ਹਾਲਤਾਂ ਲਈ #1 ਹੈ।",
        'alt_encourage':    "ਤੁਹਾਡਾ ਖੇਤ {crop} ਉਗਾਉਣ ਲਈ ਸਭ ਤੋਂ ਵਧੀਆ ਖੇਤਾਂ ਵਿੱਚੋਂ ਇੱਕ ਹੈ — ਤੁਸੀਂ ਸਹੀ ਰਸਤੇ 'ਤੇ ਹੋ!",
        'recommended_crop': 'ਸਿਫਾਰਸ਼ ਕੀਤੀ ਫਸਲ',
        'why_this_crop':    'ਇਹ ਫਸਲ ਕਿਉਂ?',
        'yield_outlook_for':'ਝਾੜ ਦਾ ਅਨੁਮਾਨ',
        'yield_analysis':   'ਝਾੜ ਵਿਸ਼ਲੇਸ਼ਣ',
        'match_label':      'ਤੁਹਾਡੀਆਂ ਹਾਲਤਾਂ ਕਿੰਨੀਆਂ ਮੇਲ ਖਾਂਦੀਆਂ ਹਨ',
        'best_crop_label':  'ਹੁਣ ਤੁਹਾਡੇ ਖੇਤ ਲਈ ਸਭ ਤੋਂ ਵਧੀਆ ਫਸਲ',
        'calculating':      'ਗਿਣਿਆ ਜਾ ਰਿਹਾ ਹੈ…',
        'analyzing':        'ਵਿਸ਼ਲੇਸ਼ਣ ਹੋ ਰਿਹਾ ਹੈ…',
        'outlook_good':     'ਚੰਗੀ ਪੈਦਾਵਾਰ ਉਮੀਦ',
        'outlook_moderate': 'ਦਰਮਿਆਨੀ ਪੈਦਾਵਾਰ ਉਮੀਦ',
        'outlook_poor':     'ਘੱਟ ਪੈਦਾਵਾਰ ਉਮੀਦ',
        'outlook_very_poor':'ਬਹੁਤ ਘੱਟ ਪੈਦਾਵਾਰ ਉਮੀਦ',
        'fert_excess_note': 'ਜ਼ਿਆਦਾ {nutrient} ਫਸਲ ਨੂੰ ਨੁਕਸਾਨ ਕਰ ਸਕਦਾ ਹੈ। ਇਸ ਮੌਸਮ ਵਿੱਚ ਨਾਈਟ੍ਰੋਜਨ ਖਾਦ ਤੋਂ ਬਚੋ।',
        'your_value':       'ਤੁਹਾਡਾ ਮੁੱਲ',
        'ideal_for':        'ਲਈ ਆਦਰਸ਼',
    },
    'Gujarati': {
        'key_reason':       "તમારા ખેતરની પરિસ્થિતિ {crop} ઉગાડવા માટે ઘણી અનુકૂળ છે।",
        'temp_high':        "તાપમાન ({val}°C) {crop} માટે થોડું વધારે છે, જે {min}–{max}°C પસંદ કરે છે।",
        'temp_low':         "તાપમાન ({val}°C) {crop} માટે થોડું ઓછું છે, જે {min}–{max}°C માં સૌથી સારું ઊગે છે।",
        'temp_good':        "તાપમાન ({val}°C) {crop} માટે આદર્શ છે।",
        'rain_high':        "વરસાદ ({val} મિમી) {crop} ને જોઈએ તેના ({min}–{max} મિમી) કરતાં વધારે છે — સારી ગટર વ્યવસ્થા રાખો।",
        'rain_low':         "વરસાદ ({val} મિમી) {crop} ને જોઈએ તેના ({min}–{max} મિમી) કરતાં ઓછો છે — સિંચાઈ મદદ કરશે।",
        'rain_good':        "વરસાદ ({val} મિમી) {crop} માટે બરાબર છે।",
        'ph_high':          "માટીનો pH ({val}) {crop} માટે સ્હેજ ક્ષારીય છે (આદર્શ: {min}–{max}) — ઍસિડ ખાતર ઉમેરો।",
        'ph_low':           "માટીનો pH ({val}) {crop} માટે સ્હેજ ખાટો છે (આદર્શ: {min}–{max}) — ચૂનો ઉમેરવાથી મદદ થઈ શકે।",
        'ph_good':          "માટીનો pH ({val}) {crop} માટે આદર્શ શ્રેણીમાં છે।",
        'n_low':            "નાઇટ્રોજન ({val} કિ.ગ્રા/હે.) ઓછો છે — યુરિયા અથવા ખાતર નાખીને વૃદ્ધિ વધારો।",
        'conf_high':        "મોડેલ {conf}% વિશ્વાસ સાથે કહે છે — તમારી પરિસ્થિતિ {crop} સાથે ઘણી સારી મળે છે।",
        'conf_mid':         "મોડેલ {conf}% વિશ્વાસ સાથે કહે છે — મોટાભાગની પરિસ્થિતિ {crop} માટે ઠીક છે, થોડો સુધારો જરૂરી।",
        'conf_low':         "મોડેલ {conf}% વિશ્વાસ સાથે કહે છે — પરિસ્થિતિ આંશિક રીતે મળે છે; સાવધાનીથી સંભાળો।",
        'tip_drainage':     "પાણી ભરાઈ ન જાય તે માટે ખેતરમાં સારી ગટર વ્યવસ્થા કરો।",
        'tip_irrigation':   "{crop} ની પાણીની જરૂરિયાત પૂરી કરવા ટપક અથવા પૂર સિંચાઈ ગોઠવો।",
        'tip_general':      "માટીની ભેજ નિયમિત ચકાસો અને ભૂ-આરોગ્ય કાર્ડ પ્રમાણે ખાતર આપો।",
        'yield_good':       "તમારા ખેતરની પરિસ્થિતિ {crop} ઉગાડવા માટે સારી છે — સારું ઉત્પાદન અપેક્ષિત છે।",
        'yield_moderate':   "તમારું ખેતર {crop} ટેકો આપી શકે છે, પરંતુ ઉત્પાદન સામાન્ય હોઈ શકે।",
        'yield_poor':       "આ પરિસ્થિતિઓમાં {crop} ઉગાડવું મુશ્કેલ હોઈ શકે અને ઉત્પાદન ઓછું રહી શકે।",
        'yield_very_poor':  "આ પરિસ્થિતિઓ {crop} માટે અનુકૂળ નથી — ઉત્પાદન ઘણું ઓછું થઈ શકે।",
        'risk_temp':        "મુખ્ય જોખમ તાપમાન છે — {crop} ને ગરમી અથવા ઠંડીનો તણાવ થઈ શકે।",
        'risk_water':       "મુખ્ય જોખમ પાણી છે — {crop} માટે સાવધાનીથી સિંચાઈ અથવા ગટર સંભાળ જરૂરી।",
        'risk_ph':          "મુખ્ય જોખમ માટીનો pH છે — આ pH સ્તરે {crop} ને પોષક તત્ત્વો શોષવામાં મુશ્કેલી પડી શકે।",
        'risk_general':     "સમગ્ર ઉગાડવાની ઋતુ દરમ્યાન પાકની તંદુરસ્તી ધ્યાનથી નીરખો।",
        'alt_suggest':      "તેના બદલે {alt} ઉગાડવાનો વિચાર કરો, જે તમારી હાલની પરિસ્થિતિ માટે #1 છે।",
        'alt_encourage':    "તમારું ખેતર {crop} ઉગાડવા માટે સૌથી શ્રેષ્ઠ ખેતરોમાંનું એક છે — તમે સાચા રસ્તે છો!",
        'recommended_crop': 'ભલામણ કરેલ પાક',
        'why_this_crop':    'આ પાક શા માટે?',
        'yield_outlook_for':'ઉત્પાદન અનુમાન',
        'yield_analysis':   'ઉત્પાદન વિશ્લેષણ',
        'match_label':      'તમારી પરિસ્થિતિ કેટલી મળે છે',
        'best_crop_label':  'હવે તમારા ખેત માટે સૌથી સારો પાક',
        'calculating':      'ગણતરી થઈ રહી છે…',
        'analyzing':        'વિશ્લેષણ થઈ રહ્યું છે…',
        'outlook_good':     'સારું ઉત્પાદન અપેક્ષિત',
        'outlook_moderate': 'સામાન્ય ઉત્પાદન અપેક્ષિત',
        'outlook_poor':     'ઓછું ઉત્પાદન અપેક્ષિત',
        'outlook_very_poor':'ખૂબ ઓછું ઉત્પાદન અપેક્ષિત',
        'fert_excess_note': 'વધુ {nutrient} પાકને નુકસાન કરી શકે. આ સિઝનમાં નાઇટ્રોજન ખાતર ટાળો.',
        'your_value':       'તમારું મૂલ્ય',
        'ideal_for':        'માટે આદર્શ',
    },
    'Odia': {
        'key_reason':       "ଆପଣଙ୍କ ଜମିର ପରିସ୍ଥିତି {crop} ଚାଷ ପାଇଁ ବହୁ ଉପଯୁକ୍ତ।",
        'temp_high':        "ତାପମାତ୍ରା ({val}°C) {crop} ପାଇଁ ଟିକେ ଅଧିକ, ଯାହା {min}–{max}°C ପସନ୍ଦ କରେ।",
        'temp_low':         "ତାପମାତ୍ରା ({val}°C) {crop} ପାଇଁ ଟିକେ କମ, ଯାହା {min}–{max}°Cରେ ସର୍ବୋତ୍ତମ ବଢ଼େ।",
        'temp_good':        "ତାପମାତ୍ରା ({val}°C) {crop} ପାଇଁ ଆଦର୍ଶ।",
        'rain_high':        "ବର୍ଷାପାତ ({val} ମିମି) {crop}ର ଆବଶ୍ୟକତା ({min}–{max} ମିମି)ଠାରୁ ଅଧିକ — ଭଲ ଜଳ ନିଷ୍କାସନ ନିଶ୍ଚିତ କରନ୍ତୁ।",
        'rain_low':         "ବର୍ଷାପାତ ({val} ମିମି) {crop}ର ଆବଶ୍ୟକତା ({min}–{max} ମିମି)ଠାରୁ କମ — ଜଳସେଚନ ସାହାଯ୍ୟ କରିବ।",
        'rain_good':        "ବର୍ଷାପାତ ({val} ମିମି) {crop} ପାଇଁ ଠିକ୍ ଅଟେ।",
        'ph_high':          "ମାଟି pH ({val}) {crop} ପାଇଁ ଟିକେ କ୍ଷାରୀୟ (ଆଦର୍ଶ: {min}–{max}) — ଅମ୍ଳ ସାର ଯୋଗ କରନ୍ତୁ।",
        'ph_low':           "ମାଟି pH ({val}) {crop} ପାଇଁ ଟିକେ ଅମ୍ଳ (ଆଦର୍ଶ: {min}–{max}) — ଚୂନ ମିଶ୍ରଣ ସାହାଯ୍ୟ କରିପାରେ।",
        'ph_good':          "ମାଟି pH ({val}) {crop} ପାଇଁ ଆଦର୍ଶ ପରିସରରେ ଅଛି।",
        'n_low':            "ନାଇଟ୍ରୋଜେନ ({val} କି.ଗ୍ରା/ହେ.) କମ — ୟୁରିଆ ବା ସାର ଦେଇ ବୃଦ୍ଧି ବଢ଼ାନ୍ତୁ।",
        'conf_high':        "ମଡେଲ {conf}% ଆତ୍ମବିଶ୍ୱାସୀ — ଆପଣଙ୍କ ପରିସ୍ଥିତି {crop} ପାଇଁ ଶକ୍ତିଶାଳୀ ମିଳ।",
        'conf_mid':         "ମଡେଲ {conf}% ଆତ୍ମବିଶ୍ୱାସୀ — ଅଧିକାଂଶ ପରିସ୍ଥିତି {crop} ପାଇଁ ଠିକ, କିଛି ଉନ୍ନତି ଦରକାର।",
        'conf_low':         "ମଡେଲ {conf}% ଆତ୍ମବିଶ୍ୱାସୀ — ପରିସ୍ଥିତି ଆଂଶିକ ମିଳୁଛି; ଯତ୍ନ ସହ ପରିଚାଳନା କରନ୍ତୁ।",
        'tip_drainage':     "ଜଳ ଜମା ନ ହେବା ପାଇଁ ଜମିରେ ଭଲ ଜଳ ନିଷ୍କାସନ ନିଶ୍ଚିତ କରନ୍ତୁ।",
        'tip_irrigation':   "{crop}ର ଜଳ ଆବଶ୍ୟକତା ପୂରଣ ପାଇଁ ଟୋପା ବା ବନ୍ୟା ଜଳସେଚନ ଲଗାନ୍ତୁ।",
        'tip_general':      "ନିୟମିତ ଭୂମି ଆର୍ଦ୍ରତା ଯାଞ୍ଚ କରନ୍ତୁ ଏବଂ ମୃତ୍ତିକା ସ୍ୱାସ୍ଥ୍ୟ କାର୍ଡ ଅନୁଯାୟୀ ସାର ଦିଅନ୍ତୁ।",
        'yield_good':       "ଆପଣଙ୍କ ଜମିର ପରିସ୍ଥିତି {crop} ଚାଷ ପାଇଁ ଭଲ — ଭଲ ଅମଳ ଆଶା କରନ୍ତୁ।",
        'yield_moderate':   "ଆପଣଙ୍କ ଜମି {crop} ସହାୟ କରିପାରିବ, କିନ୍ତୁ ଅମଳ ମଧ୍ୟମ ହୋଇପାରେ।",
        'yield_poor':       "ଏହି ପରିସ୍ଥିତିରେ {crop} ଚାଷ କଠିନ ହୋଇପାରେ ଏବଂ ଅମଳ କମ ହୋଇପାରେ।",
        'yield_very_poor':  "ଏହି ପରିସ୍ଥିତି {crop} ପାଇଁ ଉପଯୁକ୍ତ ନୁହେଁ — ଅମଳ ବହୁ କମ ହୋଇପାରେ।",
        'risk_temp':        "ମୁଖ୍ୟ ବିପଦ ତାପମାତ୍ରା — {crop}ରେ ଗରମ ବା ଥଣ୍ଡା ଚାପ ହୋଇପାରେ।",
        'risk_water':       "ମୁଖ୍ୟ ବିପଦ ଜଳ — {crop} ପାଇଁ ଯତ୍ନ ସହ ଜଳସେଚନ ବା ନିଷ୍କାସନ ପରିଚାଳନା କରନ୍ତୁ।",
        'risk_ph':          "ମୁଖ୍ୟ ବିପଦ ମାଟି pH — ଏହି pH ସ୍ତରରେ {crop} ପୋଷକ ଶୋଷଣରେ ଅସୁବିଧା ହୋଇପାରେ।",
        'risk_general':     "ଚାଷ ଋତୁ ସାରା ଫସଲର ସ୍ୱାସ୍ଥ୍ୟ ଯତ୍ନ ସହ ଦେଖନ୍ତୁ।",
        'alt_suggest':      "ବଦଳରେ {alt} ଚାଷ ବିଷୟରେ ଭାବନ୍ତୁ, ଯାହା ଆପଣଙ୍କ ର୍ତ୍ତମାନ ପରିସ୍ଥିତି ପାଇଁ #1।",
        'alt_encourage':    "ଆପଣଙ୍କ ଜମି {crop} ଚାଷ ପାଇଁ ସର୍ବୋତ୍ତମ ଜମିଗୁଡ଼ିକ ମଧ୍ୟରୁ ଗୋଟିଏ — ଆପଣ ସଠିକ ରାସ୍ତାରେ ଅଛନ୍ତି!",
        'recommended_crop': 'ପ୍ରସ୍ତାବିତ ଫସଲ',
        'why_this_crop':    'ଏହି ଫସଲ କାହିଁକି?',
        'yield_outlook_for':'ଅମଳ ଅନୁମାନ',
        'yield_analysis':   'ଅମଳ ବିଶ୍ଲେଷଣ',
        'match_label':      'ଆପଣଙ୍କ ପରିସ୍ଥିତି କେତେ ମିଳୁଛି',
        'best_crop_label':  'ବର୍ତ୍ତମାନ ଆପଣଙ୍କ ଜମି ପାଇଁ ସର୍ବୋତ୍ତମ ଫସଲ',
        'calculating':      'ଗଣନା ହେଉଛି…',
        'analyzing':        'ବିଶ୍ଲେଷଣ ହେଉଛି…',
        'outlook_good':     'ଭଲ ଅମଳ ଆଶା',
        'outlook_moderate': 'ମଧ୍ୟମ ଅମଳ ଆଶା',
        'outlook_poor':     'କମ ଅମଳ ଆଶା',
        'outlook_very_poor':'ବହୁ କମ ଅମଳ ଆଶା',
        'fert_excess_note': 'ଅଧିକ {nutrient} ଫସଲ ଖ୍ଷତି କରିପାରେ। ଏହି ଋତୁରେ ନାଇଟ୍ରୋଜେନ ସାର ଦିଅ ନାହିଁ।',
        'your_value':       'ଆପଣଙ୍କ ମୂଲ୍ୟ',
        'ideal_for':        'ପାଇଁ ଆଦର୍ଶ',
    },
}


# ── Translation helpers ────────────────────────────────────────────────────────

def t(lang: str, key: str, **kwargs) -> str:
    """
    Get a translated string. Never raises — falls back to English template,
    then to the raw key string. Handles mismatched format args gracefully.
    """
    strings  = TRANSLATIONS.get(lang, TRANSLATIONS['English'])
    template = strings.get(key, TRANSLATIONS['English'].get(key, key))
    try:
        return template.format(**kwargs)
    except (KeyError, IndexError):
        # Try English template with same kwargs
        try:
            return TRANSLATIONS['English'].get(key, key).format(**kwargs)
        except Exception:
            return template


def get_crop_name(crop: str, lang: str) -> str:
    """Return the translated display name for a crop key."""
    return CROP_NAMES.get(lang, CROP_NAMES['English']).get(crop, crop.capitalize())


def get_fert(lang: str, key: str) -> str:
    """Return a translated fertilizer string."""
    ft = FERT_TRANSLATIONS.get(lang, FERT_TRANSLATIONS['English'])
    return ft.get(key, FERT_TRANSLATIONS['English'].get(key, key))


# ── Fertilizer suggestions ─────────────────────────────────────────────────────

def build_fertilizer_suggestions(crop: str, parsed: dict, lang: str) -> list:
    """
    Simple threshold-based fertilizer advice using user-input kg/ha values.
    Returns a list of nutrient status dicts for the frontend to render.
    """
    N  = parsed['N']    # kg/ha (as entered by user)
    P  = parsed['P']
    K  = parsed['K']
    ph = parsed['ph']
    unit = get_fert(lang, 'unit')  # localised unit label

    def make_ferts(prefix, count):
        return [
            {
                'name':   get_fert(lang, f'{prefix}_{i}_name'),
                'dose':   get_fert(lang, f'{prefix}_{i}_dose'),
                'timing': get_fert(lang, f'{prefix}_{i}_timing'),
                'method': get_fert(lang, f'{prefix}_{i}_method'),
            }
            for i in range(count)
        ]

    suggestions = []

    # Nitrogen
    if N < FERT_N_LOW:
        suggestions.append({'nutrient': get_fert(lang,'nutrient_N'), 'status': 'low',
            'your_value': N, 'ideal': f'{FERT_N_LOW}–{FERT_N_HIGH} {unit}',
            'fertilizers': make_ferts('N_low', 2)})
    elif N > FERT_N_HIGH:
        suggestions.append({'nutrient': get_fert(lang,'nutrient_N'), 'status': 'excess',
            'your_value': N, 'ideal': f'{FERT_N_LOW}–{FERT_N_HIGH} {unit}', 'fertilizers': []})
    else:
        suggestions.append({'nutrient': get_fert(lang,'nutrient_N'), 'status': 'good',
            'your_value': N, 'ideal': f'{FERT_N_LOW}–{FERT_N_HIGH} {unit}', 'fertilizers': []})

    # Phosphorus
    if P < FERT_P_LOW:
        suggestions.append({'nutrient': get_fert(lang,'nutrient_P'), 'status': 'low',
            'your_value': P, 'ideal': f'{FERT_P_LOW}–{FERT_P_HIGH} {unit}',
            'fertilizers': make_ferts('P_low', 2)})
    elif P > FERT_P_HIGH:
        suggestions.append({'nutrient': get_fert(lang,'nutrient_P'), 'status': 'excess',
            'your_value': P, 'ideal': f'{FERT_P_LOW}–{FERT_P_HIGH} {unit}', 'fertilizers': []})
    else:
        suggestions.append({'nutrient': get_fert(lang,'nutrient_P'), 'status': 'good',
            'your_value': P, 'ideal': f'{FERT_P_LOW}–{FERT_P_HIGH} {unit}', 'fertilizers': []})

    # Potassium
    if K < FERT_K_LOW:
        suggestions.append({'nutrient': get_fert(lang,'nutrient_K'), 'status': 'low',
            'your_value': K, 'ideal': f'{FERT_K_LOW}–{FERT_K_HIGH} {unit}',
            'fertilizers': make_ferts('K_low', 2)})
    elif K > FERT_K_HIGH:
        suggestions.append({'nutrient': get_fert(lang,'nutrient_K'), 'status': 'excess',
            'your_value': K, 'ideal': f'{FERT_K_LOW}–{FERT_K_HIGH} {unit}', 'fertilizers': []})
    else:
        suggestions.append({'nutrient': get_fert(lang,'nutrient_K'), 'status': 'good',
            'your_value': K, 'ideal': f'{FERT_K_LOW}–{FERT_K_HIGH} {unit}', 'fertilizers': []})

    # Soil pH
    if ph < FERT_PH_LOW:
        suggestions.append({'nutrient': get_fert(lang,'nutrient_ph'), 'status': 'low',
            'your_value': ph, 'ideal': f'{FERT_PH_LOW}–{FERT_PH_HIGH}',
            'fertilizers': make_ferts('ph_low', 2)})
    elif ph > FERT_PH_HIGH:
        suggestions.append({'nutrient': get_fert(lang,'nutrient_ph'), 'status': 'high',
            'your_value': ph, 'ideal': f'{FERT_PH_LOW}–{FERT_PH_HIGH}',
            'fertilizers': make_ferts('ph_high', 2)})
    else:
        suggestions.append({'nutrient': get_fert(lang,'nutrient_ph'), 'status': 'good',
            'your_value': ph, 'ideal': f'{FERT_PH_LOW}–{FERT_PH_HIGH}', 'fertilizers': []})

    return suggestions


# ── Agronomic explanation builders ────────────────────────────────────────────

def analyse_conditions(crop: str, parsed: dict, lang: str) -> str:
    """
    Build one or two sentences describing the most impactful field conditions
    for a given crop. Uses real-world input values (not converted) for display,
    and CROP_PROFILES (kg/ha for NPK) for comparison.
    """
    crop_t  = get_crop_name(crop, lang)
    profile = CROP_PROFILES.get(crop)
    if not profile:
        return t(lang, 'key_reason', crop=crop_t)

    temp = parsed['temperature']
    rain = parsed['rainfall']   # user's raw mm — compare against profile mm
    ph   = parsed['ph']

    issues = []
    if temp > profile['temp'][1]:
        issues.append(t(lang, 'temp_high', val=temp, crop=crop_t,
                        min=profile['temp'][0], max=profile['temp'][1]))
    elif temp < profile['temp'][0]:
        issues.append(t(lang, 'temp_low', val=temp, crop=crop_t,
                        min=profile['temp'][0], max=profile['temp'][1]))

    if rain > profile['rain'][1]:
        issues.append(t(lang, 'rain_high', val=rain, crop=crop_t,
                        min=profile['rain'][0], max=profile['rain'][1]))
    elif rain < profile['rain'][0]:
        issues.append(t(lang, 'rain_low', val=rain, crop=crop_t,
                        min=profile['rain'][0], max=profile['rain'][1]))

    if ph > profile['ph'][1]:
        issues.append(t(lang, 'ph_high', val=ph, crop=crop_t,
                        min=profile['ph'][0], max=profile['ph'][1]))
    elif ph < profile['ph'][0]:
        issues.append(t(lang, 'ph_low', val=ph, crop=crop_t,
                        min=profile['ph'][0], max=profile['ph'][1]))

    if issues:
        return issues[0]

    # No issues — give a positive summary
    sentences = [t(lang, 'key_reason', crop=crop_t)]
    if profile['temp'][0] <= temp <= profile['temp'][1]:
        sentences.append(t(lang, 'temp_good', val=temp, crop=crop_t))
    return ' '.join(sentences)


def build_explanation(crop: str, parsed: dict, confidence: float, lang: str) -> str:
    """3-sentence prediction explanation: condition → confidence → tip."""
    crop_t = get_crop_name(crop, lang)

    sentence1 = analyse_conditions(crop, parsed, lang)

    if confidence >= 80:
        sentence2 = t(lang, 'conf_high', conf=confidence, crop=crop_t)
    elif confidence >= 50:
        sentence2 = t(lang, 'conf_mid',  conf=confidence, crop=crop_t)
    else:
        sentence2 = t(lang, 'conf_low',  conf=confidence, crop=crop_t)

    rain    = parsed['rainfall']
    profile = CROP_PROFILES.get(crop)
    if profile and rain > profile['rain'][1]:
        sentence3 = t(lang, 'tip_drainage')
    elif profile and rain < profile['rain'][0]:
        sentence3 = t(lang, 'tip_irrigation', crop=crop_t)
    else:
        sentence3 = t(lang, 'tip_general')

    return f"{sentence1} {sentence2} {sentence3}"


def build_yield_analysis(crop: str, parsed: dict, score: float,
                         rank: int, best_alt: str, lang: str) -> str:
    """4-sentence yield analysis: outlook → risk → tip → encouragement/alternative."""
    crop_t    = get_crop_name(crop, lang)
    alt_t     = get_crop_name(best_alt, lang)
    profile   = CROP_PROFILES.get(crop)

    # Sentence 1: overall outlook
    if score >= 75:
        s1 = t(lang, 'yield_good',      crop=crop_t)
    elif score >= 50:
        s1 = t(lang, 'yield_moderate',  crop=crop_t)
    elif score >= 25:
        s1 = t(lang, 'yield_poor',      crop=crop_t)
    else:
        s1 = t(lang, 'yield_very_poor', crop=crop_t)

    # Sentence 2: biggest risk factor
    if profile:
        temp_ok = profile['temp'][0] <= parsed['temperature'] <= profile['temp'][1]
        rain_ok = profile['rain'][0] <= parsed['rainfall']    <= profile['rain'][1]
        ph_ok   = profile['ph'][0]   <= parsed['ph']          <= profile['ph'][1]
        if not temp_ok:
            s2 = t(lang, 'risk_temp',  crop=crop_t)
        elif not rain_ok:
            s2 = t(lang, 'risk_water', crop=crop_t)
        elif not ph_ok:
            s2 = t(lang, 'risk_ph',    crop=crop_t)
        else:
            s2 = t(lang, 'risk_general')
    else:
        s2 = t(lang, 'risk_general')

    # Sentence 3: practical tip
    rain    = parsed['rainfall']
    if profile and rain > profile['rain'][1]:
        s3 = t(lang, 'tip_drainage')
    elif profile and rain < profile['rain'][0]:
        s3 = t(lang, 'tip_irrigation', crop=crop_t)
    else:
        s3 = t(lang, 'tip_general')

    # Sentence 4: alternative or encouragement
    s4 = t(lang, 'alt_encourage', crop=crop_t) if rank <= 3 else \
         t(lang, 'alt_suggest',   alt=alt_t)

    return f"{s1} {s2} {s3} {s4}"


# ── Geocoding proxy ───────────────────────────────────────────────────────────

@app.route('/geocode')
def geocode():
    """Proxy for Nominatim — avoids browser CORS restrictions."""
    q     = request.args.get('q', '').strip()
    limit = request.args.get('limit', 5)
    if not q:
        return jsonify([])
    try:
        resp = http_requests.get(
            'https://nominatim.openstreetmap.org/search',
            params={'q': q, 'format': 'json', 'limit': limit, 'addressdetails': 1},
            headers={'Accept-Language': 'en', 'User-Agent': 'CropSense/1.0'},
            timeout=8,
        )
        return jsonify(resp.json())
    except Exception as e:
        print(f"[Geocode error] {e}")
        return jsonify({'error': str(e)}), 502


@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


# ── Predict endpoint ──────────────────────────────────────────────────────────

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data     = request.get_json()
        language = data.get('language', 'English')

        # 1. Validate
        parsed, err = validate_inputs(data)
        if err:
            return jsonify({'error': err}), 400

        # 2. Convert to model feature vector
        features = build_model_features(parsed)

        # 3. Predict
        prediction = model.predict([features])[0]
        proba      = model.predict_proba([features])[0]
        confidence = round(float(np.max(proba)) * 100, 2)

        # 3b. Apply hybrid rule-based override when ML confidence is low
        final_crop = rule_based_override(
            parsed['N'], parsed['P'], parsed['K'],
            parsed['temperature'],
            parsed['humidity'],
            parsed['ph'],
            parsed['rainfall'],
            prediction,
            confidence
        )

        # 4. Build top alternatives (top 3 crops excluding the winner, ranked by proba)
        classes        = list(model.classes_)
        sorted_indices = list(np.argsort(proba)[::-1])
        top_alternatives = []
        for idx in sorted_indices:
            c = classes[idx]
            if c == final_crop:
                continue
            top_alternatives.append({
                'crop':  c,
                'name':  get_crop_name(c, language),
                'score': round(float(proba[idx]) * 100, 2),
            })
            if len(top_alternatives) >= 3:
                break

        # 5. Build response
        crop_name   = get_crop_name(final_crop, language)
        explanation = build_explanation(final_crop, parsed, confidence, language)
        fertilizer  = build_fertilizer_suggestions(final_crop, parsed, language)

        print(f"[Predict] ml_crop={prediction}, final_crop={final_crop}, conf={confidence}%, lang={language}")
        return jsonify({
            'crop':              final_crop,
            'crop_name':         crop_name,
            'confidence':        confidence,
            'explanation':       explanation,
            'language':          language,
            'fertilizer':        fertilizer,
            'top_alternatives':  top_alternatives,
            'ui': {
                'recommended_crop': t(language, 'recommended_crop'),
                'why_this_crop':    t(language, 'why_this_crop'),
                'your_value':       t(language, 'your_value'),
                'ideal_for':        t(language, 'ideal_for'),
            },
        })

    except Exception as e:
        print(f"[Predict error] {e}")
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 400


# ── Yield query endpoint ──────────────────────────────────────────────────────

@app.route('/yield_query', methods=['POST'])
def yield_query():
    try:
        data     = request.get_json()
        language = data.get('language', 'English')
        crop     = data.get('crop', '').strip().lower()

        if not crop:
            return jsonify({'error': 'Please enter a crop name.'}), 400

        # 1. Validate
        parsed, err = validate_inputs(data)
        if err:
            return jsonify({'error': err}), 400

        # 2. Convert
        features = build_model_features(parsed)

        # 3. Predict probabilities
        proba   = model.predict_proba([features])[0]
        classes = list(model.classes_)

        if crop in classes:
            crop_idx         = classes.index(crop)
            suitability      = round(float(proba[crop_idx]) * 100, 2)
            sorted_p         = sorted(proba, reverse=True)
            rank             = sorted_p.index(proba[crop_idx]) + 1
            best_idx         = int(np.argmax(proba))
            best_crop        = classes[best_idx]
            if best_crop == crop:
                best_crop = classes[int(np.argsort(proba)[-2])]
        else:
            suitability = 0.0
            rank        = len(KNOWN_CROPS) + 1
            best_crop   = classes[int(np.argmax(proba))]

        # ── Smart optimality decision ──────────────────────────────────────────
        # A crop is "optimal" if it is the top-ranked crop, or if its probability
        # is within 5 percentage points of the top-ranked crop.
        max_proba   = float(np.max(proba))
        crop_proba  = float(proba[classes.index(crop)]) if crop in classes else 0.0
        is_optimal  = (crop_proba >= max_proba)   

        # ── Top alternatives (top 3, excluding the queried crop) ───────────────
        sorted_indices = list(np.argsort(proba)[::-1])
        top_alternatives = []
        for idx in sorted_indices:
            c = classes[idx]
            if c == crop:
                continue
            top_alternatives.append({
                'crop':  c,
                'name':  get_crop_name(c, language),
                'score': round(float(proba[idx]) * 100, 2),
            })
            if len(top_alternatives) >= 3:
                break

        # 4. Outlook label
        if suitability >= 75:
            outlook, color = 'Good',      'green'
        elif suitability >= 50:
            outlook, color = 'Moderate',  'yellow'
        elif suitability >= 25:
            outlook, color = 'Poor',      'orange'
        else:
            outlook, color = 'Very Poor', 'red'

        outlook_label = t(language, f'outlook_{outlook.lower().replace(" ", "_")}')

        # 5. Build analysis + fertilizer
        crop_name     = get_crop_name(crop,      language)
        best_alt_name = get_crop_name(best_crop, language)
        analysis      = build_yield_analysis(crop, parsed, suitability, rank, best_crop, language)
        fertilizer    = build_fertilizer_suggestions(crop, parsed, language)

        # 6. Best-alternative ideal conditions (for UI panels)
        best_alt_score  = 0.0
        best_alt_conds  = None
        if best_crop in classes:
            best_alt_score = round(float(proba[classes.index(best_crop)]) * 100, 2)
        bp = CROP_PROFILES.get(best_crop)
        if bp:
            def ok(v, r): return r[0] <= v <= r[1]
            unit = get_fert(language, 'unit')
            _pl  = {
                'English': {'temp':'Temperature','humidity':'Humidity','rain':'Annual Rainfall'},
                'Hindi':   {'temp':'तापमान','humidity':'आर्द्रता','rain':'वार्षिक वर्षा'},
                'Telugu':  {'temp':'ఉష్ణోగ్రత','humidity':'తేమ','rain':'వార్షిక వర్షపాతం'},
                'Tamil':   {'temp':'வெப்பநிலை','humidity':'ஈரப்பதம்','rain':'வருடாந்திர மழையளவு'},
                'Kannada': {'temp':'ತಾಪಮಾನ','humidity':'ಆರ್ದ್ರತೆ','rain':'ವಾರ್ಷಿಕ ಮಳೆ'},
                'Marathi': {'temp':'तापमान','humidity':'आर्द्रता','rain':'वार्षिक पाऊस'},
                'Bengali': {'temp':'তাপমাত্রা','humidity':'আর্দ্রতা','rain':'বার্ষিক বৃষ্টিপাত'},
                'Punjabi': {'temp':'ਤਾਪਮਾਨ','humidity':'ਨਮੀ','rain':'ਸਾਲਾਨਾ ਬਾਰਸ਼'},
                'Gujarati':{'temp':'તાપમાન','humidity':'ભેજ','rain':'વાર્ષિક વરસાદ'},
                'Odia':    {'temp':'ତାପମାତ୍ରା','humidity':'ଆର୍ଦ୍ରତା','rain':'ବାର୍ଷିକ ବର୍ଷା'},
            }.get(language, {'temp':'Temperature','humidity':'Humidity','rain':'Annual Rainfall'})

            best_alt_conds = [
                {'key':'N',        'label':get_fert(language,'nutrient_N'), 'your':parsed['N'],            'ideal':f"{bp['N'][0]}–{bp['N'][1]}",               'unit':unit,  'ok':ok(parsed['N'],           bp['N'])},
                {'key':'P',        'label':get_fert(language,'nutrient_P'), 'your':parsed['P'],            'ideal':f"{bp['P'][0]}–{bp['P'][1]}",               'unit':unit,  'ok':ok(parsed['P'],           bp['P'])},
                {'key':'K',        'label':get_fert(language,'nutrient_K'), 'your':parsed['K'],            'ideal':f"{bp['K'][0]}–{bp['K'][1]}",               'unit':unit,  'ok':ok(parsed['K'],           bp['K'])},
                {'key':'temp',     'label':_pl['temp'],                     'your':parsed['temperature'],  'ideal':f"{bp['temp'][0]}–{bp['temp'][1]}",         'unit':'°C',  'ok':ok(parsed['temperature'], bp['temp'])},
                {'key':'humidity', 'label':_pl['humidity'],                 'your':parsed['humidity'],     'ideal':f"{bp['humidity'][0]}–{bp['humidity'][1]}", 'unit':'%',   'ok':ok(parsed['humidity'],    bp['humidity'])},
                {'key':'ph',       'label':get_fert(language,'nutrient_ph'),'your':parsed['ph'],           'ideal':f"{bp['ph'][0]}–{bp['ph'][1]}",             'unit':'',    'ok':ok(parsed['ph'],          bp['ph'])},
                {'key':'rain',     'label':_pl['rain'],                     'your':parsed['rainfall'],     'ideal':f"{bp['rain'][0]}–{bp['rain'][1]}",         'unit':'mm',  'ok':ok(parsed['rainfall'],    bp['rain'])},
            ]

        # 7. Go-ahead logic (suitability ≥ 60 % and a different crop scores higher)
        GO_AHEAD_THRESHOLD = 60
        go_ahead      = suitability >= GO_AHEAD_THRESHOLD and best_crop != crop
        go_ahead_body = ''
        if go_ahead:
            go_ahead_body = t(language, 'go_ahead_body',
                              crop=crop_name, score=suitability,
                              alt=best_alt_name, alt_score=best_alt_score)

        print(f"[YieldQuery] crop={crop}, score={suitability}%, is_optimal={is_optimal}, rank={rank}, lang={language}")
        return jsonify({
            'crop':                      crop,
            'crop_name':                 crop_name,
            'suitability_score':         suitability,
            'rank':                      rank,
            'total_crops':               len(KNOWN_CROPS),
            'outlook':                   outlook,
            'outlook_label':             outlook_label,
            'outlook_color':             color,
            'is_optimal':                is_optimal,
            'top_alternatives':          top_alternatives,
            'best_alternative':          best_crop,
            'best_alternative_name':     best_alt_name,
            'best_alternative_score':    best_alt_score,
            'best_alternative_conditions': best_alt_conds,
            'go_ahead':                  go_ahead,
            'go_ahead_title':            t(language, 'go_ahead_title'),
            'go_ahead_body':             go_ahead_body,
            'ideal_conditions_label':    t(language, 'ideal_conditions').format(alt=best_alt_name),
            'fit_yes':                   t(language, 'fit_yes'),
            'fit_no':                    t(language, 'fit_no'),
            'param_label':               t(language, 'param_label'),
            'yours_label':               t(language, 'yours_label'),
            'ideal_label':               t(language, 'ideal_label').format(alt=best_alt_name),
            'fit_label':                 t(language, 'fit_label'),
            'analysis':                  analysis,
            'language':                  language,
            'known_crop':                crop in classes,
            'fertilizer':                fertilizer,
            'ui': {
                'yield_outlook_for': t(language, 'yield_outlook_for'),
                'yield_analysis':    t(language, 'yield_analysis'),
                'match_label':       t(language, 'match_label'),
                'best_crop_label':   t(language, 'best_crop_label'),
                'your_value':        t(language, 'your_value'),
                'ideal_for':         t(language, 'ideal_for'),
            },
        })

    except Exception as e:
        print(f"[YieldQuery error] {e}")
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 400


@app.route('/crops', methods=['GET'])
def get_crops():
    return jsonify({'crops': sorted(KNOWN_CROPS)})


if __name__ == '__main__':
    app.run(debug=True)