import streamlit as st
from PIL import Image

from detector import WildlifeDetector
from utils import calculate_threat, get_detected_classes


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Wildlife Monitoring & Poaching Detection",
    page_icon="W",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* --------------------------------------------------------
       GLOBAL PAGE
    -------------------------------------------------------- */

    .stApp {
        background:
            linear-gradient(
                rgba(8, 30, 18, 0.82),
                rgba(8, 30, 18, 0.88)
            ),
            url("https://images.unsplash.com/photo-1448375240586-882707db888b?auto=format&fit=crop&w=2400&q=85");

        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }


    /* --------------------------------------------------------
       MAIN CONTENT
    -------------------------------------------------------- */

    .main .block-container {
        max-width: 1200px;
        padding-top: 3rem;
        padding-bottom: 4rem;
    }


    /* --------------------------------------------------------
       SIDEBAR
    -------------------------------------------------------- */

    section[data-testid="stSidebar"] {
        background:
            linear-gradient(
                rgba(13, 39, 24, 0.96),
                rgba(7, 27, 17, 0.96)
            );

        border-right: 1px solid rgba(255,255,255,0.08);
    }

    section[data-testid="stSidebar"] h2 {
        color: #e8f5e9;
        font-size: 1.15rem;
        letter-spacing: 0.5px;
    }


    /* --------------------------------------------------------
       TITLE
    -------------------------------------------------------- */

    .title {
        font-size: 3.1rem;
        font-weight: 750;
        line-height: 1.1;
        color: #f1f8f3;
        margin-bottom: 0.7rem;
        letter-spacing: -1px;
    }

    .subtitle {
        font-size: 1.15rem;
        color: #b9d5c0;
        margin-bottom: 2rem;
    }


    /* --------------------------------------------------------
       DASHBOARD CARD
    -------------------------------------------------------- */

    .dashboard-card {
        background: rgba(12, 35, 23, 0.72);
        border: 1px solid rgba(174, 213, 181, 0.16);
        border-radius: 18px;
        padding: 2rem;
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        box-shadow: 0 12px 40px rgba(0,0,0,0.25);
    }


    /* --------------------------------------------------------
       SECTION HEADERS
    -------------------------------------------------------- */

    .section-title {
        font-size: 1.15rem;
        font-weight: 650;
        color: #e8f5e9;
        margin-bottom: 0.8rem;
    }


    /* --------------------------------------------------------
       INFO BOX
    -------------------------------------------------------- */

    .info-box {
        background: rgba(35, 74, 49, 0.72);
        border: 1px solid rgba(144, 202, 149, 0.22);
        border-radius: 12px;
        padding: 1rem 1.1rem;
        color: #d9eadc;
        font-size: 0.92rem;
        line-height: 1.55;
    }


    /* --------------------------------------------------------
       UPLOAD AREA
    -------------------------------------------------------- */

    [data-testid="stFileUploader"] {
        background: rgba(238, 246, 239, 0.94);
        border-radius: 15px;
        padding: 0.5rem;
        border: 1px dashed #71977a;
    }

    [data-testid="stFileUploader"] section {
        border: none;
    }


    /* --------------------------------------------------------
       RESULT IMAGE
    -------------------------------------------------------- */

    .image-container {
        background: rgba(5, 24, 14, 0.7);
        border-radius: 14px;
        padding: 0.6rem;
        border: 1px solid rgba(255,255,255,0.08);
    }


    /* --------------------------------------------------------
       THREAT STATUS
    -------------------------------------------------------- */

    .threat-high {
        background: rgba(120, 30, 30, 0.75);
        border: 1px solid rgba(255, 120, 120, 0.35);
        border-radius: 14px;
        padding: 1.25rem;
        color: #ffe5e5;
    }

    .threat-medium {
        background: rgba(125, 88, 20, 0.75);
        border: 1px solid rgba(255, 210, 100, 0.35);
        border-radius: 14px;
        padding: 1.25rem;
        color: #fff1c7;
    }

    .threat-low {
        background: rgba(30, 100, 55, 0.75);
        border: 1px solid rgba(130, 230, 150, 0.30);
        border-radius: 14px;
        padding: 1.25rem;
        color: #e1f5e5;
    }

    .threat-title {
        font-size: 1.3rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }


    /* --------------------------------------------------------
       METRICS
    -------------------------------------------------------- */

    [data-testid="stMetric"] {
        background: rgba(19, 49, 31, 0.75);
        border: 1px solid rgba(180, 220, 185, 0.12);
        border-radius: 12px;
        padding: 0.8rem;
    }


    /* --------------------------------------------------------
       BUTTON
    -------------------------------------------------------- */

    .stButton > button {
        background: #416b4b;
        color: white;
        border: none;
        border-radius: 9px;
    }

    .stButton > button:hover {
        background: #527e5c;
        color: white;
    }


    /* --------------------------------------------------------
       FOOTER
    -------------------------------------------------------- */

    .footer {
        text-align: center;
        color: rgba(220,235,222,0.55);
        font-size: 0.8rem;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid rgba(255,255,255,0.08);
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown("## Detection Settings")

confidence = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.10,
    max_value=0.90,
    value=0.25,
    step=0.05,
)

st.sidebar.markdown("---")

st.sidebar.markdown(
    """
    <div class="info-box">
    <strong>Threat Assessment</strong><br><br>
    Threat levels are calculated using a rule-based
    assessment layer on top of YOLO object detections.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_detector():
    return WildlifeDetector()


try:
    detector = load_detector()

except Exception as e:
    st.error(f"Unable to load detection model: {e}")
    st.stop()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="title">
        Wildlife Monitoring &<br>
        Poaching Detection
    </div>

    <div class="subtitle">
        AI-powered wildlife object detection and threat assessment
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# INTRODUCTION
# ============================================================

st.markdown(
    """
    <div class="dashboard-card">

    <div class="section-title">
        Wildlife Scene Analysis
    </div>

    <p style="color:#c5dac9; line-height:1.6;">
    Upload an image captured from a wildlife monitoring camera.
    The trained YOLO11 model will identify wildlife, humans,
    vehicles, weapons and other relevant objects.
    </p>

    </div>
    """,
    unsafe_allow_html=True,
)


st.write("")


# ============================================================
# IMAGE UPLOAD
# ============================================================

st.markdown(
    '<div class="section-title">Upload Monitoring Image</div>',
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
)


# ============================================================
# DETECTION
# ============================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    with st.spinner("Analysing wildlife scene..."):

        results = detector.predict(
            image,
            confidence=confidence,
        )

    result = results[0]

    annotated_image = result.plot()

    detected_classes = get_detected_classes(result)

    threat_level, threat_message = calculate_threat(
        detected_classes
    )


    # ========================================================
    # IMAGE RESULTS
    # ========================================================

    st.write("")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            '<div class="section-title">Input Image</div>',
            unsafe_allow_html=True,
        )

        st.image(
            image,
            use_container_width=True,
        )


    with col2:

        st.markdown(
            '<div class="section-title">Detection Output</div>',
            unsafe_allow_html=True,
        )

        st.image(
            annotated_image,
            channels="BGR",
            use_container_width=True,
        )


    # ========================================================
    # THREAT ASSESSMENT
    # ========================================================

    st.write("")

    st.markdown(
        '<div class="section-title">Threat Assessment</div>',
        unsafe_allow_html=True,
    )


    if threat_level == "HIGH":

        st.markdown(
            f"""
            <div class="threat-high">
                <div class="threat-title">
                    HIGH RISK
                </div>
                {threat_message}
            </div>
            """,
            unsafe_allow_html=True,
        )

    elif threat_level == "MEDIUM":

        st.markdown(
            f"""
            <div class="threat-medium">
                <div class="threat-title">
                    MEDIUM RISK
                </div>
                {threat_message}
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:

        st.markdown(
            f"""
            <div class="threat-low">
                <div class="threat-title">
                    LOW RISK
                </div>
                {threat_message}
            </div>
            """,
            unsafe_allow_html=True,
        )


    # ========================================================
    # DETECTION SUMMARY
    # ========================================================

    st.write("")

    st.markdown(
        '<div class="section-title">Detection Summary</div>',
        unsafe_allow_html=True,
    )


    if detected_classes:

        counts = {}

        for cls in detected_classes:
            counts[cls] = counts.get(cls, 0) + 1


        metric_columns = st.columns(
            min(len(counts), 4)
        )


        for index, (cls, count) in enumerate(
            counts.items()
        ):

            with metric_columns[
                index % len(metric_columns)
            ]:

                st.metric(
                    label=cls,
                    value=count,
                )

    else:

        st.info(
            "No objects were detected above the selected confidence threshold."
        )


else:

    st.markdown(
        """
        <div class="info-box" style="margin-top:1rem;">
        Upload a monitoring image to begin object detection
        and threat assessment.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        Wildlife Monitoring & Poaching Detection System
        &nbsp;|&nbsp;
        YOLO11 Object Detection
    </div>
    """,
    unsafe_allow_html=True,
)