import streamlit as st
import time
import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# -----------------------------------------
# Page Config
# -----------------------------------------
st.set_page_config(
    page_title="User Stories Generator",
    page_icon="📋",
    layout="wide"
)

st.title("📋 AI User Stories Generator")
st.markdown("Convert plain requirements into User Stories + Module Breakdown using your fine-tuned FLAN-T5 model.")

# -----------------------------------------
# Check Files
# -----------------------------------------
st.sidebar.header("📁 Model Files Status")

required_files = [
    "adapter_config.json",
    "adapter_model.safetensors",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "tokenizer.json"
]

missing = False
for file in required_files:
    if os.path.exists(file):
        st.sidebar.success(f"✅ {file}")
    else:
        st.sidebar.error(f"❌ {file}")
        missing = True

# -----------------------------------------
# Load Model
# -----------------------------------------
@st.cache_resource
def load_model():
    base_model = "google/flan-t5-base"   # you can change if needed

    tokenizer = AutoTokenizer.from_pretrained("./")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        base_model,
        device_map="auto"
    )

    # Load LoRA adapter
    model.load_adapter("./", "lora")
    model.set_active_adapters("lora")

    return tokenizer, model

if not missing:
    tokenizer, model = load_model()
else:
    st.warning("⚠️ Missing model files. Generation disabled.")

# -----------------------------------------
# Input Box
# -----------------------------------------
requirement = st.text_area(
    "Enter requirement:",
    height=150,
    placeholder="Example: As a restaurant owner, I want a mobile app for online ordering..."
)

# -----------------------------------------
# Generate Button
# -----------------------------------------
if st.button("🚀 Generate", type="primary"):

    if missing:
        st.error("❌ Cannot generate because model files are missing.")
    elif not requirement:
        st.warning("⚠️ Please enter a requirement first.")
    else:
        with st.spinner("Generating... please wait"):

            # Build prompt
            prompt = f"""
Requirement: {requirement}

Generate:
1. 5 User Stories (As a ___, I want ___ so that ___)
2. Module Breakdown with bullet points
"""

            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            output = model.generate(
                **inputs,
                max_length=512,
                num_beams=4,
                temperature=0.7
            )

            decoded = tokenizer.decode(output[0], skip_special_tokens=True)

        # ------------------------------
        # Display Output
        # ------------------------------
        st.markdown("### 📖 Generated Output")
        st.write(decoded)

        # Download button
        st.download_button(
            "📥 Download Result",
            decoded,
            file_name="user_stories_output.txt",
            mime="text/plain"
        )

# -----------------------------------------
# Examples
# -----------------------------------------
with st.expander("💡 Examples"):
    examples = [
        "E-commerce platform with user reviews and ratings",
        "Fitness app with workout tracking and nutrition plans",
        "Hotel booking system with real-time availability",
        "LMS with quizzes, assignments, and progress tracking"
    ]
    
    for i, example in enumerate(examples, 1):
        st.code(f"{example}", language="text")

st.markdown("---")
st.caption("AI User Story Generator is running! 🚀")
