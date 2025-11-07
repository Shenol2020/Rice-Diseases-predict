import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import json
import cv2

# Page configuration
st.set_page_config(
    page_title="Rice Leaf Disease Classifier",
    page_icon="🌾",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
        .main-header {
            font-size: 3rem;
            color: #2E7D32;
            text-align: center;
            padding: 1rem;
        }
        .sub-header {
            font-size: 1.5rem;
            color: #558B2F;
            text-align: center;
            padding-bottom: 2rem;
        }
        .prediction-box {
            padding: 2rem;
            border-radius: 10px;
            background-color: #F1F8E9;
            margin: 1rem 0;
        }
    </style>
""", unsafe_allow_html=True)

# Load model and class names
@st.cache_resource
def load_model_and_classes():
    model = keras.models.load_model('rice_disease_classifier_final.keras')
    with open('class_names.json', 'r') as f:
        class_names = json.load(f)
    return model, class_names

def preprocess_image(image, target_size=(224, 224)):
    """Preprocess uploaded image for prediction"""
    # Convert PIL Image to numpy array
    img_array = np.array(image)
    
    # Resize image
    img_resized = cv2.resize(img_array, target_size)
    
    # Normalize pixel values
    img_normalized = img_resized / 255.0
    
    # Add batch dimension
    img_batch = np.expand_dims(img_normalized, axis=0)
    
    return img_batch

def predict_disease(model, image, class_names):
    """Make prediction on uploaded image"""
    # Preprocess image
    processed_image = preprocess_image(image)
    
    # Make prediction
    predictions = model.predict(processed_image, verbose=0)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_idx] * 100
    
    # Get top 3 predictions
    top_3_idx = np.argsort(predictions[0])[-3:][::-1]
    top_3_predictions = [(class_names[i], predictions[0][i] * 100) for i in top_3_idx]
    
    return class_names[predicted_class_idx], confidence, top_3_predictions, predictions[0]

# Main app
def main():
    # Header
    st.markdown('<div class="main-header">🌾 Rice Leaf Disease Classifier</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Upload a rice leaf image to detect diseases</div>', unsafe_allow_html=True)
    
    # Load model
    try:
        model, class_names = load_model_and_classes()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()
    
    # Sidebar with information
    with st.sidebar:
        st.header("📋 About")
        st.write("""
        This application uses deep learning to classify rice leaf diseases.
        
        **Detectable Diseases:**
        - Bacterial Blight
        - Brown Spot
        - Leaf Smut
        - Leaf Blast
        - Leaf Scald
        - Sheath Blight
        - Healthy Rice Leaf
        """)
        
        st.header("📸 How to Use")
        st.write("""
        1. Upload a clear image of a rice leaf
        2. Wait for the model to process
        3. View the prediction results
        """)
        
        st.header("⚠️ Tips")
        st.write("""
        - Use well-lit, clear images
        - Focus on the affected area
        - Avoid blurry or dark images
        """)
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a rice leaf image...",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a clear image of a rice leaf"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📷 Uploaded Image")
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True)
        
        with col2:
            st.subheader("🔍 Prediction Results")
            
            # Make prediction
            with st.spinner('Analyzing image...'):
                predicted_class, confidence, top_3, all_predictions = predict_disease(
                    model, image, class_names
                )
            
            # Display main prediction
            st.markdown('', unsafe_allow_html=True)
            
            if confidence > 80:
                st.success(f"**Detected Disease:** {predicted_class}")
                st.metric("Confidence", f"{confidence:.2f}%")
            elif confidence > 60:
                st.warning(f"**Likely Disease:** {predicted_class}")
                st.metric("Confidence", f"{confidence:.2f}%")
            else:
                st.info(f"**Possible Disease:** {predicted_class}")
                st.metric("Confidence", f"{confidence:.2f}%")
                st.caption("⚠️ Low confidence - please upload a clearer image")
            
            st.markdown('', unsafe_allow_html=True)
            
            # Display top 3 predictions
            st.subheader("📊 Top 3 Predictions")
            for i, (disease, prob) in enumerate(top_3, 1):
                st.write(f"{i}. **{disease}**: {prob:.2f}%")
                st.progress(round(prob) / 100)

        
        # Display all class probabilities
        st.subheader("📈 All Class Probabilities")
        import pandas as pd
        prob_df = pd.DataFrame({
            'Disease': class_names,
            'Probability (%)': all_predictions * 100
        }).sort_values('Probability (%)', ascending=False)
        
        st.bar_chart(prob_df.set_index('Disease'))
        
        # Recommendations based on prediction
        st.subheader("💡 Recommendations")
        if predicted_class == "Healthy Rice Leaf":
            st.success("✅ The leaf appears healthy! Continue regular monitoring.")
        else:
            st.warning(f"""
            ⚠️ **{predicted_class}** detected!
            
            **Recommended Actions:**
            - Consult with an agricultural expert
            - Consider appropriate treatment methods
            - Monitor surrounding plants for spread
            - Maintain proper field hygiene
            """)

if __name__ == '__main__':
    main()
