import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import json
import cv2
import base64
from pathlib import Path
import requests

# Page configuration
st.set_page_config(
    page_title="Rice Leaf Disease Classifier",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS with modern design
st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .stApp {
        background: #0B1D26;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Hero Section */
    .hero-section {
        position: relative;
        width: 115%;
        height: 100vh;
        min-height: 800px;
        overflow: hidden;
        background: linear-gradient(330.24deg, rgba(11, 29, 38, 0.4) 31.06%, #0B1D26 108.93%),
                    url('https://images.unsplash.com/photo-1574943320219-553eb213f72d?w=1920&h=1200&fit=crop') center/cover;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: -6rem -5rem 2rem -5rem;
        padding: 0;
    }
    
    .hero-overlay {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: linear-gradient(180deg, rgba(11, 29, 38, 0.3) 0%, rgba(11, 29, 38, 0.8) 100%);
        z-index: 1;
    }
    
    .hero-content {
        position: relative;
        z-index: 2;
        max-width: 950px;
        padding: 0 2rem;
        animation: fadeInUp 1s ease-out;
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .hero-badge {
        display: inline-block;
        padding: 8px 20px;
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 50px;
        color: #FBD784;
        font-size: 0.9rem;
        font-weight: 600;
        letter-spacing: 3px;
        text-transform: uppercase;
        margin-bottom: 2rem;
        border: 1px solid rgba(251, 215, 132, 0.3);
    }
    
    .hero-title {
        font-family: 'Playfair Display', serif;
        font-size: 5rem;
        font-weight: 900;
        color: #FFFFFF;
        line-height: 1.1;
        margin: 0 0 1.5rem 0;
        text-shadow: 2px 4px 20px rgba(0, 0, 0, 0.5);
    }
    
    .hero-subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 1.3rem;
        font-weight: 300;
        color: rgba(255, 255, 255, 0.8);
        line-height: 1.6;
        margin-bottom: 2.5rem;
        max-width: 700px;
    }
    
    .hero-button {
        display: inline-block;
        padding: 16px 40px;
        background: linear-gradient(135deg, #FBD784 0%, #F4A261 100%);
        color: #0B1D26;
        font-family: 'Inter', sans-serif;
        font-size: 1.1rem;
        font-weight: 700;
        text-decoration: none;
        border-radius: 8px;
        transition: all 0.3s ease;
        box-shadow: 0 10px 30px rgba(251, 215, 132, 0.3);
        cursor: pointer;
        border: none;
    }
    
    .hero-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 15px 40px rgba(251, 215, 132, 0.5);
    }
    
    /* Scroll Indicator */
    .scroll-indicator {
        position: absolute;
        bottom: 40px;
        left: 50%;
        transform: translateX(-50%);
        z-index: 3;
        animation: bounce 2s infinite;
    }
    
    @keyframes bounce {
        0%, 20%, 50%, 80%, 100% {
            transform: translateX(-50%) translateY(0);
        }
        40% {
            transform: translateX(-50%) translateY(-10px);
        }
        60% {
            transform: translateX(-50%) translateY(-5px);
        }
    }
    
    .scroll-text {
        color: rgba(255, 255, 255, 0.6);
        font-size: 0.8rem;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 10px;
    }
    
    /* Main Content Section */
    .main-content {
        max-width: 1400px;
        margin: 0 auto;
        padding: 4rem 2rem;
    }
    
    .section-title {
        font-family: 'Playfair Display', serif;
        font-size: 3.5rem;
        font-weight: 700;
        color: #FFFFFF;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .section-subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 1.2rem;
        color: rgba(255, 255, 255, 0.6);
        text-align: center;
        margin-bottom: 4rem;
    }
    
    /* Upload Section */
    .upload-container {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 3rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
        margin-bottom: 3rem;
    }
    
    /* Prediction Card */
    .prediction-card {
        background: linear-gradient(135deg, rgba(251, 215, 132, 0.1) 0%, rgba(244, 162, 97, 0.1) 100%);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 2.5rem;
        border: 1px solid rgba(251, 215, 132, 0.2);
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
        margin: 2rem 0;
    }
    
    .prediction-label {
        font-family: 'Inter', sans-serif;
        font-size: 0.9rem;
        font-weight: 600;
        color: #FBD784;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 1rem;
    }
    
    .prediction-disease {
        font-family: 'Playfair Display', serif;
        font-size: 2.5rem;
        font-weight: 700;
        color: #FFFFFF;
        margin-bottom: 1rem;
    }
    
    .confidence-badge {
        display: inline-block;
        padding: 8px 20px;
        background: rgba(251, 215, 132, 0.2);
        border-radius: 50px;
        color: #FBD784;
        font-size: 1.1rem;
        font-weight: 600;
    }
    
    
    /* Stats Grid */
.stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 2.5rem; /* Increased gap for extra space between items */
    margin: 3rem 0;
}

.stat-card {
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(20px);
    border-radius: 15px;
    padding: 2rem;
    border: 1px solid rgba(255, 255, 255, 0.1);
    text-align: center;
    transition: all 0.3s ease;
}

.stat-card:hover {
    transform: translateY(-5px);
    border-color: rgba(251, 215, 132, 0.5);
}

.stat-number {
    font-family: 'Playfair Display', serif;
    font-size: 3rem;
    font-weight: 700;
    color: #FBD784;
    margin-bottom: 0.5rem;
}

.stat-label {
    font-family: 'Inter', sans-serif;
    font-size: 1rem;
    color: rgba(255, 255, 255, 0.7);
}

    /* Disease Cards */
    .disease-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    row-gap: 2rem; /* Space between rows */
    column-gap: 2rem; /* Space between columns */
    margin: 2rem 0;
}

    
    .disease-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(20px);
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .disease-card:hover {
        border-color: #FBD784;
        transform: scale(1.05);
    }
    
    .disease-name {
        font-family: 'Inter', sans-serif;
        font-size: 1rem;
        font-weight: 600;
        color: #FFFFFF;
        margin-bottom: 0.5rem;
    }
    
    .disease-prob {
        font-size: 0.9rem;
        color: #FBD784;
        font-weight: 600;
    }
    
    /* Progress Bar */
    .custom-progress {
        width: 100%;
        height: 8px;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        overflow: hidden;
        margin-top: 0.5rem;
    }
    
    .progress-fill {
        height: 100%;
        background: linear-gradient(90deg, #FBD784 0%, #F4A261 100%);
        border-radius: 10px;
        transition: width 1s ease;
    }
    
    /* Streamlit Override */
    .stButton button {
        background: linear-gradient(135deg, #FBD784 0%, #F4A261 100%);
        color: #0B1D26;
        font-family: 'Inter', sans-serif;
        font-size: 1.1rem;
        font-weight: 700;
        border: none;
        border-radius: 8px;
        padding: 16px 40px;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 10px 30px rgba(251, 215, 132, 0.3);
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 15px 40px rgba(251, 215, 132, 0.5);
    }
    
    /* File Uploader */
    [data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.05);
        border: 2px dashed rgba(251, 215, 132, 0.3);
        border-radius: 15px;
        padding: 2rem;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #FBD784;
    }
    
    /* Alert Boxes */
    .stAlert {
        background: rgba(251, 215, 132, 0.1);
        border: 1px solid rgba(251, 215, 132, 0.3);
        border-radius: 12px;
        color: #FFFFFF;
    }
    
    /* Responsive */
    @media (max-width: 768px) {
        .hero-title {
            font-size: 3rem;
        }
        .hero-subtitle {
            font-size: 1.1rem;
        }
        .section-title {
            font-size: 2.5rem;
        }
    }
    /* Disease link hover */
    .disease-link:hover h5 {
        text-decoration: underline;
        color: #F4A261;
    }
    </style>
""", unsafe_allow_html=True)

# Load model and class names
@st.cache_resource
def load_model_and_classes():
    try:
        model = keras.models.load_model('rice_disease_classifier_final.keras')
        with open('class_names.json', 'r') as f:
            class_names = json.load(f)
        return model, class_names
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

def preprocess_image(image, target_size=(224, 224)):
    """Preprocess uploaded image for prediction"""
    img_array = np.array(image)
    img_resized = cv2.resize(img_array, target_size)
    img_normalized = img_resized / 255.0
    img_batch = np.expand_dims(img_normalized, axis=0)
    return img_batch

def predict_disease(model, image, class_names):
    """Make prediction on uploaded image"""
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image, verbose=0)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_idx] * 100
    
    # Get all predictions sorted
    all_predictions = [(class_names[i], predictions[0][i] * 100) for i in range(len(class_names))]
    all_predictions.sort(key=lambda x: x[1], reverse=True)
    
    return class_names[predicted_class_idx], confidence, all_predictions

# Main app
def main():
    # Hero Section
    st.markdown("""
        <div class="hero-section">
            <div class="hero-overlay"></div>
            <div class="hero-content">
                <div class="hero-badge">🌾 AI-Powered Agriculture</div>
                <h1 class="hero-title">Rice Leaf Disease<br/>Classifier</h1>
                <h5><p class="hero-subtitle">
                    Advanced deep learning technology to identify rice plant diseases instantly. 
                    Upload a leaf image and get accurate diagnosis in seconds, helping farmers 
                    protect their crops and increase yield.
                </p></h5>
                <a href="#upload-section" class="hero-button">Start Diagnosis →</a>
            </div>
            <div class="scroll-indicator">
                <div class="scroll-text">Scroll Down</div>
                <div style="text-align: center; font-size: 1.5rem; color: rgba(255,255,255,0.6);">↓</div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Stats Section
    st.markdown('<div class="main-content">', unsafe_allow_html=True)
    
    st.markdown("""
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-number">7</div>
                <div class="stat-label">Disease Types Detected</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">95%+</div>
                <div class="stat-label">Accuracy Rate</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">&lt;2s</div>
                <div class="stat-label">Analysis Time</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">24/7</div>
                <div class="stat-label">Available</div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Main Content Section
    st.markdown('<div id="upload-section"></div>', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">Upload & Analyze</h2>', unsafe_allow_html=True)
    st.markdown('<p class="section-subtitle">Upload a clear image of a rice leaf to get instant disease detection</p>', unsafe_allow_html=True)
    
    # Load model
    model, class_names = load_model_and_classes()
    
    if model is None:
        st.error("⚠️ Model not found. Please ensure 'rice_disease_classifier_final.keras' is in the current directory.")
        st.stop()
    
    # Upload Section
    st.markdown('<div class="upload-container">', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose a rice leaf image",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a clear, well-lit image of a rice leaf"
    )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    if uploaded_file is not None:
        col1, col2 = st.columns([1, 1], gap="large")
        
        with col1:
            st.markdown("### 📸 Uploaded Image")
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True)
        
        with col2:
            st.markdown("### 🔬 Analysis Results")
            
            with st.spinner('🔄 Analyzing image...'):
                predicted_class, confidence, all_predictions = predict_disease(
                    model, image, class_names
                )
            
            # Main Prediction Card
            st.markdown(f"""
                <div class="prediction-card">
                    <div class="prediction-label">Detected Disease</div>
                    <div class="prediction-disease">{predicted_class}</div>
                    <span class="confidence-badge">{confidence:.1f}% Confidence</span>
                </div>
            """, unsafe_allow_html=True)
            
            # Confidence indicator
            if confidence > 80:
                st.success("✅ High confidence detection")
            elif confidence > 60:
                st.warning("⚠️ Moderate confidence - consider retaking image")
            else:
                st.info("ℹ️ Low confidence - please upload a clearer image")
        
        # All Predictions
        st.markdown("---")
        st.markdown("### 📊 Detailed Analysis")
        
        st.markdown('<div class="disease-grid">', unsafe_allow_html=True)
        for disease, prob in all_predictions:
            st.markdown(f"""
                <div class="disease-card">
                    <div class="disease-name">{disease}</div>
                    <div class="disease-prob">{prob:.1f}%</div>
                    <div class="custom-progress">
                        <div class="progress-fill" style="width: {prob}%"></div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Recommendations
        st.markdown("---")
        st.markdown("### 💡 Recommendations")
        
        if predicted_class == "Healthy Rice Leaf":
            st.success("""
            **Great News! Your rice plant appears healthy.**
            
            ✅ Continue regular monitoring  
            ✅ Maintain proper irrigation  
            ✅ Follow standard fertilization schedule  
            ✅ Keep the field clean and weed-free
            """)
        else:
            st.warning(f"""
            **Action Required: {predicted_class} Detected**
            
            ⚠️ **Immediate Steps:**
            - Isolate affected plants if possible
            - Consult with a local agricultural expert
            - Consider appropriate fungicide/pesticide treatment
            - Improve field drainage and ventilation
            - Monitor surrounding plants closely
            
            📞 **Need Help?** Contact your local agricultural extension office for specific treatment recommendations.
            """)
    
    # Disease Information Section
    st.markdown("---")
    st.markdown('<h2 class="section-title">Disease Information</h2>', unsafe_allow_html=True)
    st.markdown('<p class="section-subtitle">Learn about the diseases we can detect</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2, gap="large")
    
    diseases_info = {
        "Bacterial Blight": {
            "description": "Caused by Xanthomonas oryzae. Symptoms include water-soaked lesions on leaves.",
            "url": "https://en.wikipedia.org/wiki/Xanthomonas_oryzae_pv._oryzae"
        },
        "Brown Spot": {
            "description": "Fungal disease causing brown spots with gray centers on leaves and grains.",
            "url": "https://en.wikipedia.org/wiki/Cochliobolus_miyabeanus"
        },
        "Leaf Smut": {
            "description": "Fungal infection producing black powdery spores on leaves.",
            "url": "https://en.wikipedia.org/wiki/Eballistra_oryzae"
        },
        "Leaf Blast": {
            "description": "Most destructive rice disease, causing diamond-shaped lesions.",
            "url": "https://en.wikipedia.org/wiki/Magnaporthe_oryzae"
        },
        "Leaf Scald": {
            "description": "Bacterial disease causing scalded appearance on leaf tips.",
            "url": "https://en.wikipedia.org/wiki/Monographella_albescens"
        },
        "Sheath Blight": {
            "description": "Fungal disease affecting leaf sheaths near water line.",
            "url": "https://en.wikipedia.org/wiki/Sheath_blight"
        }
    }

    for i, (name, data) in enumerate(diseases_info.items()):
        with col1 if i % 2 == 0 else col2:
            icon = "✅" if name == "Healthy Rice Leaf" else "⚠️"
            url_attr = f'href="{data["url"]}" target="_blank" rel="noopener noreferrer"' if data.get("url") else ""
            st.markdown(f"""
                <div class="stat-card" style="text-align: left;">
                    <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">{icon}</div>
                    <a {url_attr} class="disease-link" style="text-decoration:none;">
                        <h5 style="margin:0; color:#FBD784;">{name}</h5>
                    </a>
                    <p style="color: rgba(255,255,255,0.6); font-size: 0.9rem; margin-top:0.5rem;">{data['description']}</p>
                </div>
            """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

    # 🌤 Local Weather (below Disease Information)
    st.markdown('<h2 style="text-align:center;">🌤 Local Weather</h2>', unsafe_allow_html=True)
    import streamlit.components.v1 as components
    weather_widget = """
    <div id="weatherapi-weather-widget-2"></div>
    <script type="text/javascript"
        src="https://www.weatherapi.com/weather/widget.ashx?loc=2850955&wid=2&tu=2&div=weatherapi-weather-widget-2"></script>
    <noscript>
        <a href="https://www.weatherapi.com/weather/q/polonnaruwa-2850955"
        alt="Hour by hour Polonnaruwa weather">
        10 day hour by hour Polonnaruwa weather
        </a>
    </noscript>
    """
    components.html(weather_widget, height=400, scrolling=False)

    # Footer (moved to the very end of the page)
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; padding: 2rem; color: rgba(255,255,255,0.5);">
            <p style="font-size: 0.9rem;">🌾 Rice Leaf Disease Classifier | Powered by Deep Learning & TensorFlow</p>
            <p style="font-size: 0.8rem; margin-top: 0.5rem;">© 2024 All Rights Reserved</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()