"""
ArchitectXpert – Streamlit Client
Run with:
    streamlit run web/client_streamlit.py
"""
import io
import requests
import streamlit as st
from PIL import Image

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="ArchitectXpert", page_icon="🏠", layout="centered")
st.title("🏗️ ArchitectXpert – Floorplan Generator")
st.markdown("### Enter your dimensions below to generate a floorplan:")

col1, col2, col3 = st.columns(3)
with col1:
    width = st.number_input("Width (ft)", min_value=5.0, max_value=200.0, value=25.0)
with col2:
    depth = st.number_input("Depth (ft)", min_value=5.0, max_value=200.0, value=45.0)
with col3:
    px_per_ft = st.number_input("Pixels per foot", min_value=5, max_value=50, value=18)

st.write("---")

if st.button("✨ Generate Floorplan"):
    st.info("Generating floorplan... please wait ⏳")
    payload = {"width": width, "depth": depth, "px_per_ft": px_per_ft}
    try:
        response = requests.post(API_URL, json=payload, timeout=120)
        if response.status_code == 200:
            img = Image.open(io.BytesIO(response.content))
            st.success("✅ Generated successfully!")
            st.image(img, caption=f"{width}ft x {depth}ft", use_container_width=True)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            st.download_button("⬇️ Download Image", buf.getvalue(), file_name="floorplan.png", mime="image/png")
        else:
            st.error(f"❌ API returned error: {response.status_code} – {response.text}")
    except requests.exceptions.RequestException as e:
        st.error(f"⚠️ Could not connect to backend: {e}")
