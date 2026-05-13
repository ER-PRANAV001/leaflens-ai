import os
import json
import numpy as np
from flask import Flask, request, render_template, jsonify
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from PIL import Image

app = Flask(__name__)
@app.before_request
def handle_ngrok():
    pass
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
app.config['PROPAGATE_EXCEPTIONS'] = True

@app.after_request
def skip_ngrok_warning(response):
    response.headers['ngrok-skip-browser-warning'] = 'true'
    return response

# ─── Load model ───────────────────────────────────────────────────────────────
print("Loading model... please wait...")
try:
    import tensorflow as tf
    model = tf.keras.models.load_model('crop_disease_model.h5')
    print("Model loaded OK")
except Exception as e:
    print(f"Model load error: {e}")
    model = None

# ─── Load class names ─────────────────────────────────────────────────────────
try:
    with open('class_names.json') as f:
        CLASS_NAMES = json.load(f)
    print(f"Classes loaded OK: {len(CLASS_NAMES)}")
except Exception as e:
    print(f"Class names error: {e}")
    CLASS_NAMES = []

# ─── Disease info ─────────────────────────────────────────────────────────────
DISEASE_INFO = {
    'Apple___Apple_scab': {"name": "Apple Scab", "crop": "Apple", "severity": "Medium", "description": "Caused by Venturia inaequalis. Olive-green to brown velvety lesions on leaves and fruit.", "treatment": "Apply captan or myclobutanil fungicide. Remove fallen infected leaves.", "prevention": "Plant resistant varieties, rake fallen leaves, apply dormant sprays.", "color": "#7f8c8d"},
    'Apple___Black_rot': {"name": "Apple Black Rot", "crop": "Apple", "severity": "High", "description": "Caused by Botryosphaeria obtusa. Brown rotting lesions on fruit and purple-bordered spots on leaves.", "treatment": "Apply thiophanate-methyl fungicide. Prune and destroy infected branches.", "prevention": "Remove mummified fruit, prune dead wood, maintain orchard sanitation.", "color": "#c0392b"},
    'Apple___Cedar_apple_rust': {"name": "Cedar Apple Rust", "crop": "Apple", "severity": "Medium", "description": "Caused by Gymnosporangium juniperi-virginianae. Bright orange-yellow spots on upper leaf surfaces.", "treatment": "Apply myclobutanil from pink bud stage through cover sprays.", "prevention": "Remove nearby cedar trees, plant rust-resistant apple varieties.", "color": "#e67e22"},
    'Apple___healthy': {"name": "Healthy Apple", "crop": "Apple", "severity": "None", "description": "No disease detected. The apple plant is healthy.", "treatment": "No treatment required.", "prevention": "Maintain orchard hygiene, balanced fertilization, regular pruning.", "color": "#27ae60"},
    'Blueberry___healthy': {"name": "Healthy Blueberry", "crop": "Blueberry", "severity": "None", "description": "No disease detected. The blueberry plant is healthy.", "treatment": "No treatment required.", "prevention": "Maintain soil pH 4.5-5.5 and good drainage.", "color": "#27ae60"},
    'Cherry_(including_sour)___Powdery_mildew': {"name": "Cherry Powdery Mildew", "crop": "Cherry", "severity": "Medium", "description": "Caused by Podosphaera clandestina. White powdery fungal growth on young leaves.", "treatment": "Apply sulfur-based fungicides. Remove infected shoots.", "prevention": "Improve air circulation, avoid excess nitrogen.", "color": "#f39c12"},
    'Cherry_(including_sour)___healthy': {"name": "Healthy Cherry", "crop": "Cherry", "severity": "None", "description": "No disease detected. The cherry plant is healthy.", "treatment": "No treatment required.", "prevention": "Maintain proper pruning and balanced fertilization.", "color": "#27ae60"},
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': {"name": "Corn Gray Leaf Spot", "crop": "Corn", "severity": "High", "description": "Caused by Cercospora zeae-maydis. Rectangular tan-gray lesions along leaf veins.", "treatment": "Apply strobilurin or triazole fungicides. Destroy infected residue.", "prevention": "Rotate crops, use resistant hybrids, till soil.", "color": "#95a5a6"},
    'Corn_(maize)___Common_rust_': {"name": "Corn Common Rust", "crop": "Corn", "severity": "Medium", "description": "Caused by Puccinia sorghi. Small cinnamon-brown pustules on both leaf surfaces.", "treatment": "Apply mancozeb or propiconazole at early infection stage.", "prevention": "Plant resistant hybrids, monitor during cool wet weather.", "color": "#d35400"},
    'Corn_(maize)___Northern_Leaf_Blight': {"name": "Corn Northern Leaf Blight", "crop": "Corn", "severity": "High", "description": "Caused by Exserohilum turcicum. Long cigar-shaped gray-green lesions on leaves.", "treatment": "Apply propiconazole or azoxystrobin. Remove infected debris.", "prevention": "Plant resistant hybrids, crop rotation, avoid dense planting.", "color": "#e67e22"},
    'Corn_(maize)___healthy': {"name": "Healthy Corn", "crop": "Corn", "severity": "None", "description": "No disease detected. The corn plant is healthy.", "treatment": "No treatment required.", "prevention": "Maintain proper spacing, balanced fertilization, crop rotation.", "color": "#27ae60"},
    'Grape___Black_rot': {"name": "Grape Black Rot", "crop": "Grapes", "severity": "High", "description": "Caused by Guignardia bidwellii. Tan spots on leaves and berries turn black and shrivel.", "treatment": "Apply mancozeb or myclobutanil. Remove mummified berries.", "prevention": "Remove mummified fruit, prune infected canes.", "color": "#1a1a2e"},
    'Grape___Esca_(Black_Measles)': {"name": "Grape Esca Black Measles", "crop": "Grapes", "severity": "High", "description": "Caused by wood-rotting fungi. Tiger-stripe leaf patterns and dark berry spots.", "treatment": "No complete cure. Remove infected wood. Apply wound sealant.", "prevention": "Avoid large pruning wounds, use clean tools.", "color": "#8e44ad"},
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': {"name": "Grape Leaf Blight", "crop": "Grapes", "severity": "Medium", "description": "Caused by Isariopsis clavispora. Dark brown spots causing premature leaf drop.", "treatment": "Apply copper oxychloride or mancozeb. Remove affected leaves.", "prevention": "Maintain vine spacing, avoid water stress.", "color": "#784212"},
    'Grape___healthy': {"name": "Healthy Grape", "crop": "Grapes", "severity": "None", "description": "No disease detected. The grapevine is healthy.", "treatment": "No treatment required.", "prevention": "Proper pruning, canopy management, preventive sprays.", "color": "#27ae60"},
    'Orange___Haunglongbing_(Citrus_greening)': {"name": "Citrus Greening HLB", "crop": "Orange", "severity": "High", "description": "Caused by Candidatus Liberibacter. Yellow mottled leaves and lopsided bitter fruit.", "treatment": "No cure. Remove infected trees. Control psyllid insects.", "prevention": "Use certified disease-free plants, control Asian citrus psyllid.", "color": "#f39c12"},
    'Peach___Bacterial_spot': {"name": "Peach Bacterial Spot", "crop": "Peach", "severity": "Medium", "description": "Caused by Xanthomonas arboricola. Water-soaked spots on leaves and fruit cracking.", "treatment": "Apply copper-based bactericides. Avoid overhead irrigation.", "prevention": "Plant resistant varieties, apply copper sprays in early spring.", "color": "#e74c3c"},
    'Peach___healthy': {"name": "Healthy Peach", "crop": "Peach", "severity": "None", "description": "No disease detected. The peach plant is healthy.", "treatment": "No treatment required.", "prevention": "Proper pruning, fertilization, pest monitoring.", "color": "#27ae60"},
    'Pepper,_bell___Bacterial_spot': {"name": "Pepper Bacterial Spot", "crop": "Bell Pepper", "severity": "Medium", "description": "Caused by Xanthomonas campestris. Water-soaked lesions turning brown on leaves.", "treatment": "Apply copper-based bactericide. Remove infected debris.", "prevention": "Use disease-free seeds, avoid overhead watering.", "color": "#e74c3c"},
    'Pepper,_bell___healthy': {"name": "Healthy Bell Pepper", "crop": "Bell Pepper", "severity": "None", "description": "No disease detected. The pepper plant is healthy.", "treatment": "No treatment required.", "prevention": "Proper irrigation, balanced nutrition, regular scouting.", "color": "#27ae60"},
    'Potato___Early_blight': {"name": "Potato Early Blight", "crop": "Potato", "severity": "Medium", "description": "Caused by Alternaria solani. Dark concentric ring lesions on older leaves.", "treatment": "Apply chlorothalonil or mancozeb. Remove infected lower leaves.", "prevention": "Crop rotation, certified seed tubers, avoid overhead irrigation.", "color": "#d35400"},
    'Potato___Late_blight': {"name": "Potato Late Blight", "crop": "Potato", "severity": "High", "description": "Caused by Phytophthora infestans. Dark water-soaked lesions with white mold.", "treatment": "Apply copper-based fungicides immediately. Destroy infected material.", "prevention": "Use resistant varieties, plant certified tubers.", "color": "#c0392b"},
    'Potato___healthy': {"name": "Healthy Potato", "crop": "Potato", "severity": "None", "description": "No disease detected. The potato plant is healthy.", "treatment": "No treatment required.", "prevention": "Certified seed tubers, proper hilling, balanced fertilization.", "color": "#27ae60"},
    'Raspberry___healthy': {"name": "Healthy Raspberry", "crop": "Raspberry", "severity": "None", "description": "No disease detected. The raspberry plant is healthy.", "treatment": "No treatment required.", "prevention": "Proper pruning, air circulation, weed control.", "color": "#27ae60"},
    'Soybean___healthy': {"name": "Healthy Soybean", "crop": "Soybean", "severity": "None", "description": "No disease detected. The soybean plant is healthy.", "treatment": "No treatment required.", "prevention": "Crop rotation, proper plant spacing, weed management.", "color": "#27ae60"},
    'Squash___Powdery_mildew': {"name": "Squash Powdery Mildew", "crop": "Squash", "severity": "Medium", "description": "Caused by Podosphaera xanthii. White powdery coating on leaves and stems.", "treatment": "Apply sulfur sprays. Remove infected leaves.", "prevention": "Improve air circulation, avoid overhead watering.", "color": "#f39c12"},
    'Strawberry___Leaf_scorch': {"name": "Strawberry Leaf Scorch", "crop": "Strawberry", "severity": "Medium", "description": "Caused by Diplocarpon earlianum. Purple-red spots giving a scorched appearance.", "treatment": "Apply captan or myclobutanil. Remove infected leaves.", "prevention": "Avoid overhead irrigation, well-drained soil.", "color": "#e74c3c"},
    'Strawberry___healthy': {"name": "Healthy Strawberry", "crop": "Strawberry", "severity": "None", "description": "No disease detected. The strawberry plant is healthy.", "treatment": "No treatment required.", "prevention": "Proper spacing, good drainage, regular monitoring.", "color": "#27ae60"},
    'Tomato___Bacterial_spot': {"name": "Tomato Bacterial Spot", "crop": "Tomato", "severity": "Medium", "description": "Caused by Xanthomonas vesicatoria. Dark brown spots with yellow halo on leaves.", "treatment": "Apply copper bactericide sprays. Remove infected material.", "prevention": "Disease-free seeds, avoid overhead watering, crop rotation.", "color": "#e74c3c"},
    'Tomato___Early_blight': {"name": "Tomato Early Blight", "crop": "Tomato", "severity": "Medium", "description": "Caused by Alternaria solani. Concentric dark brown rings on older leaves.", "treatment": "Apply chlorothalonil or mancozeb. Remove infected lower leaves.", "prevention": "Crop rotation, mulching, avoid wetting foliage.", "color": "#e67e22"},
    'Tomato___Late_blight': {"name": "Tomato Late Blight", "crop": "Tomato", "severity": "High", "description": "Caused by Phytophthora infestans. Dark lesions with white mold in humid conditions.", "treatment": "Apply copper-based fungicides. Remove infected parts.", "prevention": "Resistant varieties, avoid overhead irrigation.", "color": "#c0392b"},
    'Tomato___Leaf_Mold': {"name": "Tomato Leaf Mold", "crop": "Tomato", "severity": "Medium", "description": "Caused by Passalora fulva. Yellow spots on upper leaf and olive mold on underside.", "treatment": "Apply copper or chlorothalonil. Improve ventilation.", "prevention": "Reduce humidity, improve air circulation.", "color": "#7f8c8d"},
    'Tomato___Septoria_leaf_spot': {"name": "Tomato Septoria Leaf Spot", "crop": "Tomato", "severity": "Medium", "description": "Caused by Septoria lycopersici. Small circular gray-centered spots spreading upward.", "treatment": "Apply mancozeb or copper fungicide. Remove infected leaves.", "prevention": "Crop rotation, disease-free seeds, stake plants.", "color": "#7f8c8d"},
    'Tomato___Spider_mites Two-spotted_spider_mite': {"name": "Tomato Spider Mites", "crop": "Tomato", "severity": "Medium", "description": "Caused by Tetranychus urticae. Yellow stippling and fine webbing on leaves.", "treatment": "Apply miticides or neem oil. Increase humidity.", "prevention": "Avoid water stress, use predatory mites.", "color": "#d35400"},
    'Tomato___Target_Spot': {"name": "Tomato Target Spot", "crop": "Tomato", "severity": "Medium", "description": "Caused by Corynespora cassiicola. Dark brown concentric ring spots on leaves.", "treatment": "Apply azoxystrobin or chlorothalonil. Remove infected debris.", "prevention": "Improve air circulation, avoid leaf wetness.", "color": "#784212"},
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': {"name": "Tomato Yellow Leaf Curl Virus", "crop": "Tomato", "severity": "High", "description": "Caused by TYLCV via whiteflies. Leaves curl upward and yellow, plants stunted.", "treatment": "No cure. Remove infected plants. Control whiteflies.", "prevention": "Virus-resistant varieties, insect-proof nets.", "color": "#f39c12"},
    'Tomato___Tomato_mosaic_virus': {"name": "Tomato Mosaic Virus", "crop": "Tomato", "severity": "High", "description": "Caused by ToMV. Mosaic pattern on leaves, distortion, stunted growth.", "treatment": "No cure. Remove and destroy infected plants immediately.", "prevention": "Resistant varieties, disinfect tools, control aphids.", "color": "#c0392b"},
    'Tomato___healthy': {"name": "Healthy Tomato", "crop": "Tomato", "severity": "None", "description": "No disease detected. The tomato plant is perfectly healthy.", "treatment": "No treatment required.", "prevention": "Crop rotation, balanced fertilization, proper staking.", "color": "#27ae60"},
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

def classify_image(image_path):
    arr        = preprocess_image(image_path)
    preds      = model.predict(arr, verbose=0)[0]
    top_idx    = int(np.argmax(preds))
    confidence = float(preds[top_idx]) * 100
    label      = CLASS_NAMES[top_idx]

    disease = DISEASE_INFO.get(label, {
        "name":        label.replace('___', ' - ').replace('_', ' '),
        "crop":        label.split('___')[0].replace('_', ' '),
        "severity":    "Unknown",
        "description": "No additional info available.",
        "treatment":   "Consult a local agronomist.",
        "prevention":  "Follow standard agricultural practices.",
        "color":       "#7f8c8d"
    })

    top3 = np.argsort(preds)[::-1][:3]
    top_predictions = []
    for i in top3:
        lbl  = CLASS_NAMES[i]
        info = DISEASE_INFO.get(lbl, {})
        top_predictions.append({
            "name":       info.get("name", lbl.replace('___', ' - ').replace('_', ' ')),
            "confidence": round(float(preds[i]) * 100, 1)
        })

    return {
        "disease":         disease,
        "confidence":      round(confidence, 1),
        "top_predictions": top_predictions
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type. Allowed: PNG, JPG, JPEG, WEBP"}), 400

    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        img = Image.open(filepath)
        img.verify()
    except Exception:
        os.remove(filepath)
        return jsonify({"error": "Not a valid image"}), 400

    try:
        result = classify_image(filepath)
        result["image_url"] = f"/static/uploads/{filename}"
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)