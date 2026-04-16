from flask import Flask, request, jsonify, send_from_directory
import pickle
import numpy as np
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__, static_folder='static')

# Load model
with open('crop_model.pkl', 'rb') as f:
    model = pickle.load(f)

KNOWN_CROPS = [
    'apple', 'banana', 'blackgram', 'chickpea', 'coconut', 'coffee',
    'cotton', 'grapes', 'jute', 'kidneybeans', 'lentil', 'maize',
    'mango', 'mothbeans', 'mungbean', 'muskmelon', 'orange', 'papaya',
    'pigeonpeas', 'pomegranate', 'rice', 'watermelon'
]

# Ideal ranges for each crop: (N, P, K, temp_min, temp_max, humidity_min, humidity_max, ph_min, ph_max, rain_min, rain_max)
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
        'n_low':            "Nitrogen ({val} mg/kg) is on the lower side — applying urea or compost can boost growth.",
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
        'n_low':            "नाइट्रोजन ({val} मिग्रा/किग्रा) कम है — यूरिया या खाद से वृद्धि बढ़ाएँ।",
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
        'n_low':            "నత్రజని ({val} మిగ్రా/కిగ్రా) తక్కువగా ఉంది — యూరియా లేదా కంపోస్ట్ వేయండి।",
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
        'n_low':            "நைட்ரஜன் ({val} மி.கி/கி.கி) குறைவாக உள்ளது — யூரியா அல்லது உரம் இட்டு வளர்ச்சியை மேம்படுத்துங்கள்।",
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
        'n_low':            "ಸಾರಜನಕ ({val} ಮಿಗ್ರಾ/ಕಿಗ್ರಾ) ಕಡಿಮೆ ಇದೆ — ಯೂರಿಯಾ ಅಥವಾ ಗೊಬ್ಬರ ಹಾಕಿ ಬೆಳವಣಿಗೆ ಹೆಚ್ಚಿಸಿ।",
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
        'n_low':            "नायट्रोजन ({val} मि.ग्रॅ/कि.ग्रॅ) कमी आहे — युरिया किंवा कंपोस्ट घालून वाढ सुधारा।",
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
        'n_low':            "নাইট্রোজেন ({val} মিগ্রা/কিগ্রা) কম — ইউরিয়া বা সার দিয়ে বৃদ্ধি বাড়ান।",
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
        'n_low':            "ਨਾਈਟ੍ਰੋਜਨ ({val} ਮਿਗ੍ਰਾ/ਕਿਗ੍ਰਾ) ਘੱਟ ਹੈ — ਯੂਰੀਆ ਜਾਂ ਖਾਦ ਪਾ ਕੇ ਵਾਧਾ ਵਧਾਓ।",
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
        'n_low':            "નાઇટ્રોજન ({val} મિ.ગ્રા/કિ.ગ્રા) ઓછો છે — યુરિયા અથવા ખાતર નાખીને વૃદ્ધિ વધારો।",
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
        'n_low':            "ନାଇଟ୍ରୋଜେନ ({val} ମି.ଗ୍ରା/କି.ଗ୍ରା) କମ — ୟୁରିଆ ବା ସାର ଦେଇ ବୃଦ୍ଧି ବଢ଼ାନ୍ତୁ।",
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
    },
}


def t(lang, key, **kwargs):
    """Get translated string, fallback to English."""
    strings = TRANSLATIONS.get(lang, TRANSLATIONS['English'])
    template = strings.get(key, TRANSLATIONS['English'].get(key, key))
    return template.format(**kwargs)


def analyse_conditions(crop, inputs, lang):
    """Build a rule-based explanation for prediction."""
    N   = float(inputs['N'])
    P   = float(inputs['P'])
    K   = float(inputs['K'])
    temp     = float(inputs['temperature'])
    humidity = float(inputs['humidity'])
    ph       = float(inputs['ph'])
    rain     = float(inputs['rainfall'])

    profile = CROP_PROFILES.get(crop)
    sentences = []

    if profile:
        # Sentence 1: most impactful factor
        issues = []
        if temp > profile['temp'][1]:
            issues.append(('temp', t(lang, 'temp_high', val=temp, crop=crop, min=profile['temp'][0], max=profile['temp'][1])))
        elif temp < profile['temp'][0]:
            issues.append(('temp', t(lang, 'temp_low', val=temp, crop=crop, min=profile['temp'][0], max=profile['temp'][1])))

        if rain > profile['rain'][1]:
            issues.append(('water', t(lang, 'rain_high', val=rain, crop=crop, min=profile['rain'][0], max=profile['rain'][1])))
        elif rain < profile['rain'][0]:
            issues.append(('water', t(lang, 'rain_low', val=rain, crop=crop, min=profile['rain'][0], max=profile['rain'][1])))

        if ph > profile['ph'][1]:
            issues.append(('ph', t(lang, 'ph_high', val=ph, crop=crop, min=profile['ph'][0], max=profile['ph'][1])))
        elif ph < profile['ph'][0]:
            issues.append(('ph', t(lang, 'ph_low', val=ph, crop=crop, min=profile['ph'][0], max=profile['ph'][1])))

        if issues:
            sentences.append(issues[0][1])
        else:
            sentences.append(t(lang, 'key_reason', crop=crop))
            if temp >= profile['temp'][0] and temp <= profile['temp'][1]:
                sentences.append(t(lang, 'temp_good', val=temp, crop=crop))
    else:
        sentences.append(t(lang, 'key_reason', crop=crop))

    return ' '.join(sentences)


def build_explanation(crop, inputs, confidence, lang):
    """3-sentence prediction explanation."""
    sentence1 = analyse_conditions(crop, inputs, lang)

    # Sentence 2: confidence meaning
    if confidence >= 80:
        sentence2 = t(lang, 'conf_high', conf=confidence, crop=crop)
    elif confidence >= 50:
        sentence2 = t(lang, 'conf_mid', conf=confidence, crop=crop)
    else:
        sentence2 = t(lang, 'conf_low', conf=confidence, crop=crop)

    # Sentence 3: tip
    rain = float(inputs['rainfall'])
    profile = CROP_PROFILES.get(crop)
    if profile and rain > profile['rain'][1]:
        sentence3 = t(lang, 'tip_drainage')
    elif profile and rain < profile['rain'][0]:
        sentence3 = t(lang, 'tip_irrigation', crop=crop)
    else:
        sentence3 = t(lang, 'tip_general')

    return f"{sentence1} {sentence2} {sentence3}"


def build_yield_analysis(crop, inputs, suitability_score, rank, best_alt, lang):
    """4-sentence yield analysis."""
    # Sentence 1: yield outlook
    if suitability_score >= 75:
        s1 = t(lang, 'yield_good', crop=crop)
    elif suitability_score >= 50:
        s1 = t(lang, 'yield_moderate', crop=crop)
    elif suitability_score >= 25:
        s1 = t(lang, 'yield_poor', crop=crop)
    else:
        s1 = t(lang, 'yield_very_poor', crop=crop)

    # Sentence 2: biggest risk
    profile = CROP_PROFILES.get(crop)
    if profile:
        temp = float(inputs['temperature'])
        rain = float(inputs['rainfall'])
        ph   = float(inputs['ph'])
        temp_ok = profile['temp'][0] <= temp <= profile['temp'][1]
        rain_ok = profile['rain'][0] <= rain <= profile['rain'][1]
        ph_ok   = profile['ph'][0]   <= ph   <= profile['ph'][1]
        if not temp_ok:
            s2 = t(lang, 'risk_temp', crop=crop)
        elif not rain_ok:
            s2 = t(lang, 'risk_water', crop=crop)
        elif not ph_ok:
            s2 = t(lang, 'risk_ph', crop=crop)
        else:
            s2 = t(lang, 'risk_general')
    else:
        s2 = t(lang, 'risk_general')

    # Sentence 3: tip
    rain_val = float(inputs['rainfall'])
    if profile and rain_val > profile['rain'][1]:
        s3 = t(lang, 'tip_drainage')
    elif profile and rain_val < profile['rain'][0]:
        s3 = t(lang, 'tip_irrigation', crop=crop)
    else:
        s3 = t(lang, 'tip_general')

    # Sentence 4: alternative or encouragement
    if rank <= 3:
        s4 = t(lang, 'alt_encourage', crop=crop)
    else:
        s4 = t(lang, 'alt_suggest', alt=best_alt)

    return f"{s1} {s2} {s3} {s4}"


@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        language = data.get('language', 'English')

        features = [
            float(data['N']), float(data['P']), float(data['K']),
            float(data['temperature']), float(data['humidity']),
            float(data['ph']), float(data['rainfall']),
        ]

        prediction = model.predict([features])[0]
        proba = model.predict_proba([features])[0]
        confidence = round(float(max(proba)) * 100, 2)
        explanation = build_explanation(prediction, data, confidence, language)

        print(f"[Predict] crop={prediction}, confidence={confidence}, lang={language}")
        return jsonify({'crop': prediction, 'confidence': confidence,
                        'explanation': explanation, 'language': language})

    except Exception as e:
        print(f"[Predict error] {e}")
        return jsonify({'error': str(e)}), 400


@app.route('/yield_query', methods=['POST'])
def yield_query():
    try:
        data = request.get_json()
        language = data.get('language', 'English')
        crop = data.get('crop', '').strip().lower()

        if not crop:
            return jsonify({'error': 'Please enter a crop name.'}), 400

        features = [
            float(data['N']), float(data['P']), float(data['K']),
            float(data['temperature']), float(data['humidity']),
            float(data['ph']), float(data['rainfall']),
        ]

        proba = model.predict_proba([features])[0]
        classes = list(model.classes_)

        if crop in classes:
            crop_idx = classes.index(crop)
            suitability_score = round(float(proba[crop_idx]) * 100, 2)
            sorted_proba = sorted(proba, reverse=True)
            rank = list(sorted_proba).index(proba[crop_idx]) + 1
            best_idx = int(np.argmax(proba))
            best_crop = classes[best_idx]
            if best_crop == crop:
                second_idx = int(np.argsort(proba)[-2])
                best_crop = classes[second_idx]
        else:
            suitability_score = 0
            rank = len(KNOWN_CROPS) + 1
            best_crop = classes[int(np.argmax(proba))]

        if suitability_score >= 75:
            outlook, color = "Good", "green"
        elif suitability_score >= 50:
            outlook, color = "Moderate", "yellow"
        elif suitability_score >= 25:
            outlook, color = "Poor", "orange"
        else:
            outlook, color = "Very Poor", "red"

        analysis = build_yield_analysis(crop, data, suitability_score, rank, best_crop, language)

        print(f"[YieldQuery] crop={crop}, score={suitability_score}, rank={rank}, lang={language}")
        return jsonify({
            'crop': crop, 'suitability_score': suitability_score,
            'rank': rank, 'total_crops': len(KNOWN_CROPS),
            'outlook': outlook, 'outlook_color': color,
            'best_alternative': best_crop, 'analysis': analysis,
            'language': language, 'known_crop': crop in classes
        })

    except Exception as e:
        print(f"[YieldQuery error] {e}")
        return jsonify({'error': str(e)}), 400


@app.route('/crops', methods=['GET'])
def get_crops():
    return jsonify({'crops': sorted(KNOWN_CROPS)})


if __name__ == '__main__':
    app.run(debug=True)