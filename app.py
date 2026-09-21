import html
import io
import traceback
from urllib.parse import quote_plus

import streamlit as st
from PIL import Image, ImageOps

from predict import CLASS_NAMES, get_model_path, load_model, predict_image

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="Skin Disease Diagnosis",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)


def render_html(markup: str) -> None:
    """Render HTML through st.markdown.

    Lines are stripped and joined so Markdown never treats indented HTML as a
    code block and blank lines never split a block in two.
    """
    flat = "".join(line.strip() for line in markup.splitlines())
    st.markdown(flat, unsafe_allow_html=True)


def show_image(img: Image.Image) -> None:
    # Newer Streamlit uses width="stretch"; older versions use use_container_width.
    try:
        st.image(img, width="stretch")
    except Exception:
        st.image(img, use_container_width=True)


# ==================== CUSTOM STYLING ====================
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #f4f8ff 0%, #eef4ff 100%);
        color: #163b63;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ecf4ff 0%, #f8fbff 100%);
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
        color: white;
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
    .section-card h3 {
        margin-top: 0;
        margin-bottom: 0.5rem;
        color: #1b4a7b;
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
    }
    .result-box {
        background: linear-gradient(135deg, #f1fbf4 0%, #f8fff9 100%);
        border-left: 6px solid #2f9e44;
        border-radius: 16px;
        padding: 1rem 1.1rem;
        margin-top: 0.7rem;
        margin-bottom: 0.9rem;
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
    .confidence-bar {
        background: #dbe7f5;
        border-radius: 999px;
        height: 10px;
        margin-top: 0.5rem;
        overflow: hidden;
    }
    .confidence-fill {
        background: linear-gradient(90deg, #2d72d8 0%, #2f9e44 100%);
        height: 100%;
        border-radius: 999px;
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
    .alert-success, .alert-warning {
        border-radius: 14px;
        padding: 0.8rem 1rem;
        margin-top: 0.6rem;
        margin-bottom: 0.6rem;
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
    .result-disclaimer {
        padding: 0.7rem 1rem;
        border-radius: 12px;
        background: #f7faff;
        border: 1px solid #e0eaf8;
        color: #55657a;
        font-size: 0.9rem;
        margin-bottom: 0.9rem;
    }
    .referral-list {
        display: flex;
        flex-direction: column;
        gap: 0.6rem;
        margin: 0.5rem 0 0.6rem 0;
    }
    .hospital-card {
        background: white;
        border: 1px solid #dfe8f6;
        border-left: 5px solid #2d72d8;
        border-radius: 12px;
        padding: 0.75rem 1rem;
        box-shadow: 0 3px 10px rgba(10, 48, 102, 0.05);
    }
    .hospital-name {
        font-weight: 700;
        color: #133d6e;
        margin-bottom: 0.3rem;
    }
    .map-link {
        font-size: 0.9rem;
        color: #1c6fd1;
        text-decoration: none;
        font-weight: 600;
    }
    .map-link:hover {
        text-decoration: underline;
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
# Exceptions are not cached by st.cache_resource, so a failed load is retried on
# the next run instead of being stored as a permanent None.
@st.cache_resource(show_spinner="Loading model...")
def get_model():
    path = get_model_path()
    return load_model(path)


try:
    model = get_model()
except Exception as e:
    st.error(f"❌ Model could not be loaded.\n\n**{type(e).__name__}:** {e}")
    with st.expander("Technical details"):
        st.code(traceback.format_exc())
    st.stop()


# Prediction is cached per image so that changing the state dropdown (which
# reruns the script) does not run the model again.
@st.cache_data(show_spinner=False, max_entries=32)
def run_prediction(image_bytes: bytes):
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(image_bytes))).convert("RGB")
    return predict_image(img, model)


# ==================== HEADER ====================
render_html("""
<div class="hero-card">
    <h2>🏥 Skin Disease Diagnosis</h2>
    <p>AI-assisted dermatological screening trained on the PASSION dataset.</p>
</div>
""")

# ==================== SIDEBAR ====================
with st.sidebar:
    st.subheader("System information")
    st.write("Model state: active")
    st.write(f"Dataset classes: {len(CLASS_NAMES)}")
    st.write("Model specs: ResNet50, 224×224 input, PyTorch, ImageNet normalization")
    st.divider()
    st.subheader("Detected classes")
    for cls in CLASS_NAMES:
        st.write(f"• {cls}")
    st.divider()
    st.subheader("Instructions")
    st.write("1. Upload a clear, well-lit image")
    st.write("2. Wait for analysis")
    st.write("3. Review the result")
    st.write("4. Select your state to see suggested hospitals")
    st.write("5. Consult a dermatology professional")


# ==================== MAIN CONTENT ====================
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    render_html("""
    <div class="section-card">
        <h3>📤 Upload skin image</h3>
        <p style="margin:0; color:#576b84;">Upload a clear image of a skin lesion for preliminary screening.</p>
    </div>
    """)
    uploaded_file = st.file_uploader(
        "Select a dermatological image",
        type=["jpg", "jpeg", "png", "gif"],
        help="Clear, well-lit, in-focus images give the most reliable results",
    )

with col2:
    pills = "".join(f'<span class="class-pill">{c}</span>' for c in CLASS_NAMES)
    render_html(f"""
    <div class="section-card">
        <h3>✓ System capabilities</h3>
        <div class="class-list">{pills}</div>
    </div>
    """)

st.divider()


# ==================== REFERRAL DATA ====================
HOSPITALS_BY_STATE = {
    "Abia": ["Abia State University Teaching Hospital", "Federal Medical Centre Umuahia"],
    "Adamawa": ["Federal Medical Centre Yola", "Specialist Hospital Yola"],
    "Akwa Ibom": ["University of Uyo Teaching Hospital", "Ibom Specialist Hospital"],
    "Anambra": ["Nnamdi Azikiwe University Teaching Hospital", "Chukwuemeka Odumegwu Ojukwu University Teaching Hospital"],
    "Bauchi": ["Abubakar Tafawa Balewa University Teaching Hospital", "Specialist Hospital Bauchi"],
    "Bayelsa": ["Niger Delta University Teaching Hospital", "Federal Medical Centre Yenagoa"],
    "Benue": ["Benue State University Teaching Hospital", "Federal Medical Centre Makurdi"],
    "Borno": ["University of Maiduguri Teaching Hospital", "State Specialist Hospital Maiduguri"],
    "Cross River": ["University of Calabar Teaching Hospital", "General Hospital Calabar"],
    "Delta": ["Delta State University Teaching Hospital", "Central Hospital Warri"],
    "Ebonyi": ["Alex Ekwueme Federal University Teaching Hospital", "David Umahi Federal University Teaching Hospital"],
    "Edo": ["University of Benin Teaching Hospital", "Irrua Specialist Teaching Hospital"],
    "Ekiti": ["Federal Teaching Hospital Ido-Ekiti", "Ekiti State University Teaching Hospital"],
    "Enugu": ["University of Nigeria Teaching Hospital", "Enugu State University Teaching Hospital"],
    "FCT Abuja": ["University of Abuja Teaching Hospital", "Supreme Dermatology and Specialist Hospital"],
    "Gombe": ["Federal Teaching Hospital Gombe", "Specialist Hospital Gombe"],
    "Imo": ["Imo State University Teaching Hospital", "Federal Medical Centre Owerri"],
    "Jigawa": ["Federal Medical Centre Birnin Kudu", "Rasheed Shekoni Teaching Hospital"],
    "Kaduna": ["Ahmadu Bello University Teaching Hospital", "Barau Dikko Teaching Hospital"],
    "Kano": ["Aminu Kano Teaching Hospital", "Murtala Mohammed Specialist Hospital"],
    "Katsina": ["Federal Medical Centre Katsina", "General Amadi Rimi Specialist Hospital"],
    "Kebbi": ["Federal Teaching Hospital Birnin Kebbi", "Sir Yahaya Memorial Hospital"],
    "Kogi": ["Federal Medical Centre Lokoja", "Kogi State Specialist Hospital"],
    "Kwara": ["University of Ilorin Teaching Hospital", "Sobi Specialist Hospital"],
    "Lagos": ["Lagos University Teaching Hospital", "Lagos State University Teaching Hospital (LASUTH)"],
    "Nasarawa": ["Dalhatu Araf Specialist Hospital", "Federal University Teaching Hospital Lafia"],
    "Niger": ["General Hospital Minna", "Ibrahim Badamasi Babangida Specialist Hospital"],
    "Ogun": ["Olabisi Onabanjo University Teaching Hospital", "Federal Medical Centre Abeokuta"],
    "Ondo": ["University of Medical Sciences Teaching Hospital", "Federal Medical Centre Owo"],
    "Osun": ["Obafemi Awolowo University Teaching Hospitals Complex", "LAUTECH Teaching Hospital"],
    "Oyo": ["University College Hospital", "Bowen University Teaching Hospital"],
    "Plateau": ["Jos University Teaching Hospital", "Plateau Specialist Hospital"],
    "Rivers": ["University of Port Harcourt Teaching Hospital", "Rivers State University Teaching Hospital"],
    "Sokoto": ["Usmanu Danfodiyo University Teaching Hospital", "Specialist Hospital Sokoto"],
    "Taraba": ["Federal Medical Centre Jalingo", "Specialist Hospital Jalingo"],
    "Yobe": ["Federal Medical Centre Nguru", "Yobe State Specialist Hospital"],
    "Zamfara": ["Federal Medical Centre Gusau", "Yariman Bakura Specialist Hospital"],
}


def hospital_card(name: str, state: str) -> str:
    query = quote_plus(f"{name}, {state}, Nigeria")
    url = f"https://www.google.com/maps/search/?api=1&query={query}"
    return (
        '<div class="hospital-card">'
        f'<div class="hospital-name">🏥 {html.escape(name)}</div>'
        f'<a class="map-link" href="{url}" target="_blank" rel="noopener noreferrer">📍 Open in Google Maps</a>'
        "</div>"
    )


# ==================== ANALYSIS SECTION ====================
GENERAL_ADVICE = (
    "Seek medical care sooner if there is fever, spreading redness, pain, pus, "
    "or rapid change in the skin."
)

GUIDANCE = {
    "Eczema": "The image pattern resembles eczema. A clinician can confirm the type and advise on suitable care.",
    "Fungal": "The image pattern resembles a fungal skin infection. Confirmation usually needs a clinical examination and sometimes a lab test.",
    "Scabies": "The image pattern resembles scabies, which is contagious. See a clinician promptly, since close contacts may also need assessment.",
    "Dermatitis": "The image pattern resembles dermatitis. A clinician can identify the cause and advise on care.",
    "Others": "The image did not match a specific listed condition. This is not a sign that nothing is wrong. Please have it checked by a clinician.",
}

if uploaded_file is not None:
    try:
        image = ImageOps.exif_transpose(Image.open(uploaded_file)).convert("RGB")
    except Exception as e:
        st.error(f"❌ Could not read this image: {e}")
        st.stop()

    left, right = st.columns([1, 1], gap="large")

    with left:
        render_html('<div class="section-card"><h3>🖼️ Uploaded image</h3></div>')
        show_image(image)
        st.write(f"**File name:** {uploaded_file.name}")
        st.write(f"**File size:** {uploaded_file.size / 1024:.2f} KB")
        st.write(f"**Image size:** {image.size[0]} × {image.size[1]} px")

    with right:
        render_html('<div class="section-card"><h3>🔬 Analysis</h3></div>')

        try:
            with st.spinner("Analyzing image..."):
                prediction, confidence = run_prediction(uploaded_file.getvalue())

            bar_width = max(0.0, min(confidence, 100.0))
            render_html(f"""
            <div class="result-box">
                <div class="result-title">🩺 Screening result</div>
                <div class="diagnosis-name">{prediction}</div>
                <div class="confidence-score">Model confidence: <strong>{confidence:.2f}%</strong></div>
                <div class="confidence-bar"><div class="confidence-fill" style="width: {bar_width:.1f}%"></div></div>
            </div>
            """)
            render_html("""
            <div class="result-disclaimer">
                This is an automated screening result, not a diagnosis.
                Only a qualified clinician can confirm a skin condition.
            </div>
            """)

            # Risk assessment
            st.markdown("#### ⚠️ Reliability")
            if confidence >= 85:
                level = "High"
                render_html("""
                <div class="alert-success">
                    <strong>✓ HIGH MODEL CONFIDENCE</strong><br>
                    The model scores this label highly. Confidence is not the same as certainty, so professional confirmation is still needed.
                </div>
                """)
            elif confidence >= 70:
                level = "Moderate"
                render_html("""
                <div class="alert-warning">
                    <strong>⚠ MODERATE CONFIDENCE</strong><br>
                    Professional medical evaluation is recommended to confirm this result.
                </div>
                """)
            else:
                level = "Low"
                render_html("""
                <div class="alert-warning">
                    <strong>! LOW CONFIDENCE RESULT</strong><br>
                    Please consult a dermatologist. Try a clearer, better-lit image if you upload again.
                </div>
                """)

            # Metrics
            st.markdown("#### 📊 Detailed metrics")
            m1, m2, m3 = st.columns(3)
            for column, label, value in (
                (m1, "Prediction", prediction),
                (m2, "Confidence", f"{confidence:.1f}%"),
                (m3, "Reliability", level),
            ):
                with column:
                    render_html(f"""
                    <div class="metric-card">
                        <div class="metric-label">{label}</div>
                        <div class="metric-value">{value}</div>
                    </div>
                    """)

            # General guidance
            st.markdown("#### 💡 What to do next")
            st.write(GUIDANCE.get(prediction, "Please consult a dermatologist for confirmation."))
            st.write(GENERAL_ADVICE)

            # Referral to a dermatology facility by state
            st.markdown("#### 🏥 Find a dermatology clinic")
            state = st.selectbox(
                "Select your state or region in Nigeria",
                options=sorted(HOSPITALS_BY_STATE),
                index=None,
                placeholder="Choose your state",
                key="referral_state",
            )
            if state is None:
                st.caption("Select your state to see suggested hospitals for a dermatology review.")
            else:
                cards = "".join(hospital_card(name, state) for name in HOSPITALS_BY_STATE[state])
                st.write(f"Suggested facilities in **{state}**:")
                render_html(f'<div class="referral-list">{cards}</div>')
                st.caption(
                    "Please call ahead to confirm that the hospital runs a dermatology clinic "
                    "and on which days. In an emergency, go to the nearest hospital emergency unit."
                )

        except Exception as e:
            st.error(f"❌ Analysis error: {type(e).__name__}: {e}")

else:
    render_html("""
    <div style="text-align: center; padding: 3rem; color: #999;">
        <p style="font-size: 1.2rem;">👆 Upload an image to begin</p>
        <p style="font-size: 0.9rem;">Supported formats: JPG, JPEG, PNG, GIF</p>
    </div>
    """)


# ==================== FOOTER ====================
st.divider()
render_html("""
<div class="medical-footer">
    <p><strong>Skin Disease Diagnosis</strong> - AI-Assisted Dermatological Screening</p>
    <p>Powered by Artificial Intelligence and the PASSION Dataset</p>
    <div class="footer-disclaimer">
        <strong>⚠️ MEDICAL DISCLAIMER</strong><br>
        This application is designed for educational and preliminary screening purposes only.
        It should NOT be used as a substitute for professional medical advice, diagnosis, or treatment.
        Always consult a qualified dermatologist or healthcare provider for accurate diagnosis and treatment.
        The creators assume no liability for outcomes resulting from the use of this system.
    </div>
</div>
""")
