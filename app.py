import os
import tempfile
from pathlib import Path

import streamlit as st
import predict
from predict import load_model, predict_image, MODEL_PATH, CLASS_NAMES
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
TEMP_DIR = BASE_DIR / "temp_uploads"
TEMP_DIR.mkdir(exist_ok=True)

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="Skin Disease Diagnosis",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM STYLING ====================
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #f4f8ff 0%, #eef4ff 100%);
        color: #163b63;
    }
    .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }
    .hero-card {
        background: linear-gradient(135deg, #0f3d7a 0%, #2d72d8 100%);
        color: white;
        padding: 1.25rem 1.4rem;
        border-radius: 20px;
        box-shadow: 0 10px 25px rgba(15, 61, 122, 0.18);
        margin-bottom: 1rem;
    }
    .hero-card h2 {
        margin: 0 0 0.3rem 0;
        font-size: 1.7rem;
    }
    .hero-card p {
        margin: 0;
        font-size: 0.98rem;
        opacity: 0.95;
    }
    .section-card {
        background: white;
        border: 1px solid #dfe8f6;
        border-radius: 16px;
        padding: 1rem 1.15rem;
        box-shadow: 0 4px 14px rgba(10, 48, 102, 0.06);
        margin-bottom: 0.9rem;
    }
    .section-card h3, .section-card h2 {
        margin-top: 0;
        margin-bottom: 0.5rem;
        color: #1b4a7b;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #ecf4ff 0%, #f8fbff 100%);
    }
    .upload-card {
        background: linear-gradient(180deg, #f8fbff 0%, #edf5ff 100%);
        border: 2px dashed #67a8ff;
        border-radius: 16px;
        padding: 1rem;
        text-align: center;
    }
    .class-list {
        display: flex;
        flex-wrap: wrap;
        gap: 0.45rem;
        margin-top: 0.35rem;
    }
    .class-pill {
        display: inline-block;
        background: #eaf4ff;
        color: #1c4c89;
        padding: 0.45rem 0.7rem;
        border-radius: 999px;
        font-size: 0.92rem;
        font-weight: 600;
    }
    .image-preview-container {
        background: white;
        border-radius: 16px;
        padding: 0.6rem;
        border: 1px solid #e0eaf8;
        box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.03);
    }
    .result-box {
        background: linear-gradient(135deg, #f1fbf4 0%, #f8fff9 100%);
        border-left: 6px solid #2f9e44;
        border-radius: 16px;
        padding: 1rem 1.1rem;
        margin-top: 0.7rem;
    }
    .result-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #17683c;
        margin-bottom: 0.3rem;
    }
    .diagnosis-name {
        font-size: 1.35rem;
        font-weight: 800;
        color: #103f6d;
        margin-bottom: 0.25rem;
    }
    .confidence-score {
        font-size: 1rem;
        color: #294c6b;
    }
    .metric-card {
        background: white;
        border: 1px solid #dfe8f6;
        border-radius: 14px;
        padding: 0.85rem;
        text-align: center;
        box-shadow: 0 3px 10px rgba(10, 48, 102, 0.05);
        margin-bottom: 0.6rem;
    }
    .metric-label {
        font-size: 0.76rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #6f7f96;
        margin-bottom: 0.2rem;
    }
    .metric-value {
        font-size: 1.2rem;
        font-weight: 700;
        color: #133d6e;
    }
    .alert-success, .alert-warning, .alert-error {
        border-radius: 14px;
        padding: 0.8rem 1rem;
        margin-top: 0.6rem;
        font-size: 0.95rem;
    }
    .alert-success {
        background: #ebf9ee;
        color: #1d673a;
        border: 1px solid #bfe6c6;
    }
    .alert-warning {
        background: #fff8e8;
        color: #8a5a00;
        border: 1px solid #f3d08f;
    }
    .alert-error {
        background: #fff0f0;
        color: #a12e2e;
        border: 1px solid #f1b5b5;
    }
    .medical-footer {
        text-align: center;
        color: #5b6b83;
        padding-top: 0.3rem;
        font-size: 0.95rem;
    }
    .footer-disclaimer {
        margin-top: 0.6rem;
        padding: 0.8rem 1rem;
        border-radius: 12px;
        background: #f7faff;
        border: 1px solid #e0eaf8;
        color: #55657a;
    }
</style>
""", unsafe_allow_html=True)

# ==================== LOAD MODEL ====================
@st.cache_resource
def get_model():
    if MODEL_PATH is None:
        return None
    return load_model(MODEL_PATH)

model = get_model()

if model is None:
    if MODEL_PATH is None:
        st.error(f"❌ Could not download model from Hugging Face Hub.\n\n**Details:** {predict.DOWNLOAD_ERROR}")
    else:
        st.error(f"❌ Model file downloaded but failed to load.\n\n**Details:** {predict.LAST_LOAD_ERROR}")
    st.stop()

# ==================== HEADER ====================
st.markdown("""
<div class="hero-card">
    <h2>🏥 Skin Disease Diagnosis</h2>
    <p>Advanced dermatological screening powered by AI and the PASSION dataset.</p>
</div>
""", unsafe_allow_html=True)

# ==================== SIDEBAR ====================
with st.sidebar:
    st.subheader("System information")
    st.write(f"Model state: active")
    st.write(f"Dataset classes: {len(CLASS_NAMES)}")
    st.write("Model specs: 224×224, PyTorch, ImageNet normalization")
    st.divider()
    st.subheader("Detected classes")
    for cls in CLASS_NAMES:
        st.write(f"• {cls}")
    st.divider()
    st.subheader("Instructions")
    st.write("1. Upload a clear image")
    st.write("2. Wait for analysis")
    st.write("3. Review the result")
    st.write("4. Consult a professional")

# ==================== MAIN CONTENT ====================
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("""
        <div class="section-card">
            <h3>📤 Upload medical image</h3>
            <p style="margin:0; color:#576b84;">Upload a clear image of a skin lesion for preliminary analysis.</p>
        </div>
    """, unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Select a dermatological image",
        type=["jpg", "jpeg", "png", "gif"],
        help="Upload clear images of skin lesions for accurate diagnosis",
    )

with col2:
    st.markdown("""
        <div class="section-card">
            <h3>✓ System capabilities</h3>
            <div class="class-list">
    """, unsafe_allow_html=True)

    for cls in CLASS_NAMES:
        st.markdown(f'<span class="class-pill">{cls}</span>', unsafe_allow_html=True)

    st.markdown('</div></div>', unsafe_allow_html=True)

st.divider()

# ==================== ANALYSIS SECTION ====================
if uploaded_file is not None:
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown("""
            <div class="section-box">
                <h2>🖼️ Uploaded Image</h2>
            </div>
        """, unsafe_allow_html=True)
        
        image = Image.open(uploaded_file)
        st.markdown('<div class="image-preview-container">', unsafe_allow_html=True)
        st.image(image, use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Image info
        st.write(f"**File Name:** {uploaded_file.name}")
        st.write(f"**File Size:** {uploaded_file.size / 1024:.2f} KB")
        st.write(f"**Image Size:** {image.size[0]} × {image.size[1]} px")
    
    with col2:
        st.markdown("""
            <div class="section-box">
                <h2>🔬 Analysis Progress</h2>
            </div>
        """, unsafe_allow_html=True)
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Save and analyze using a temporary file in the project folder
        temp_file = tempfile.NamedTemporaryFile(suffix=".jpg", dir=str(TEMP_DIR), delete=False)
        temp_file.close()
        temp_path = temp_file.name
        image.save(temp_path)
        
        try:
            status_text.write("⏳ **Initializing analysis...**")
            progress_bar.progress(25)
            
            status_text.write("🔍 **Processing image...**")
            progress_bar.progress(50)
            
            prediction, confidence = predict_image(temp_path, model)
            
            status_text.write("✓ **Analysis complete!**")
            progress_bar.progress(100)
            
            # Display detailed results
            st.divider()
            st.markdown("""
                <div class="result-box-success">
                    <h2>🩺 DIAGNOSIS RESULT</h2>
                    <div class="diagnosis-name">{}</div>
                    <div class="confidence-score">
                        Confidence: <strong>{:.2f}%</strong>
                        <div class="confidence-bar">
                            <div class="confidence-fill" style="width: {}%"></div>
                        </div>
                    </div>
                </div>
            """.format(prediction, confidence, confidence), unsafe_allow_html=True)
            
            # Risk Assessment
            st.markdown("""
                <div class="section-box">
                    <h2>⚠️ Risk Assessment</h2>
                </div>
            """, unsafe_allow_html=True)
            
            if confidence >= 85:
                st.markdown("""
                    <div class="alert-success">
                        <strong>✓ HIGH CONFIDENCE RESULT</strong><br>
                        The model shows strong confidence in this diagnosis.
                    </div>
                """, unsafe_allow_html=True)
            elif confidence >= 70:
                st.markdown("""
                    <div class="alert-warning">
                        <strong>⚠ MODERATE CONFIDENCE</strong><br>
                        Consider professional medical evaluation for confirmation.
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                    <div class="alert-warning">
                        <strong>! LOW CONFIDENCE RESULT</strong><br>
                        Please consult a dermatologist for accurate diagnosis.
                    </div>
                """, unsafe_allow_html=True)
            
            # Detailed Metrics
            st.markdown("""
                <div class="section-box">
                    <h2>📊 Detailed Metrics</h2>
                </div>
            """, unsafe_allow_html=True)
            
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            
            with metric_col1:
                st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-card-label">Diagnosis</div>
                        <div class="metric-card-value">{prediction}</div>
                    </div>
                """, unsafe_allow_html=True)
            
            with metric_col2:
                st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-card-label">Confidence</div>
                        <div class="metric-card-value">{confidence:.1f}%</div>
                    </div>
                """, unsafe_allow_html=True)
            
            with metric_col3:
                confidence_level = "High" if confidence >= 85 else "Moderate" if confidence >= 70 else "Low"
                st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-card-label">Reliability</div>
                        <div class="metric-card-value">{confidence_level}</div>
                    </div>
                """, unsafe_allow_html=True)
            
            # Recommendations
            st.markdown("""
                <div class="section-box">
                    <h2>💡 Clinical Recommendations</h2>
                </div>
            """, unsafe_allow_html=True)
            
            recommendations = {
                'Eczema': '• Recommend dermatology consultation\n• Consider patch testing\n• Evaluate for triggers\n• Review medications\n• Assess skin barrier function',
                'Fungal': '• Fungal culture recommended\n• Topical antifungal therapy consideration\n• Hygiene assessment\n• Monitor treatment response\n• Check for secondary infections',
                'Scabies': '• Recommend scrapings/dermoscopy confirmation\n• Treatment for patient and all close contacts\n• Environmental decontamination required\n• Close follow-up after 2 weeks\n• Educate on transmission prevention',
                'Others': '• Consult dermatologist for accurate classification\n• Consider additional diagnostic tests\n• Document clinical findings and history\n• Plan follow-up care as advised\n• Consider dermoscopy if available'
            }
            
            for rec in (recommendations.get(prediction, '• Consult dermatologist for confirmation\n• Consider additional diagnostic tests\n• Document clinical findings\n• Plan follow-up care')).split('\n'):
                if rec.strip():
                    st.write(rec)
        
        except Exception as e:
            st.markdown(f"""
                <div class="alert-error">
                    <strong>❌ Analysis Error</strong><br>
                    {str(e)}
                </div>
            """, unsafe_allow_html=True)
        
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

else:
    st.markdown("""
        <div style="text-align: center; padding: 3rem; color: #999;">
            <p style="font-size: 1.2rem;">👆 Upload an image to begin diagnosis</p>
            <p style="font-size: 0.9rem;">Supported formats: JPG, JPEG, PNG, GIF</p>
        </div>
    """, unsafe_allow_html=True)

# ==================== FOOTER ====================
st.divider()
st.markdown("""
    <div class="medical-footer">
        <p><strong>Skin Disease Diagnosis</strong> - Advanced Dermatological Analysis System</p>
        <p>Powered by Artificial Intelligence & PASSION Dataset</p>
        <div class="footer-disclaimer">
            <strong>⚠️ MEDICAL DISCLAIMER</strong><br>
            This application is designed for educational and preliminary screening purposes only. 
            It should NOT be used as a substitute for professional medical advice, diagnosis, or treatment. 
            Always consult a qualified dermatologist or healthcare provider for accurate diagnosis and treatment. 
            The creators assume no liability for outcomes resulting from the use of this system.
        </div>
    </div>
""", unsafe_allow_html=True)