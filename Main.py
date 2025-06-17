import streamlit as st
import tempfile
import os
from datetime import datetime
from prediction import get_summary, okt_preprocess
from pages.Recording_Audio import uploading_file
import joblib 
from konlpy.tag import Okt

import whisper
tokenizer = Okt()
def pre_tokenizer(text):
    return tokenizer.morphs(text)
loaded_vectorizer = joblib.load("korean_tfidf_vectorizer.pkl")

def extract_keywords(loaded_vectorizer, text, top_n=3):
    tfidf_vector = loaded_vectorizer.transform([okt_preprocess(text)])
    feature_names = loaded_vectorizer.get_feature_names_out()
    sorted_indices = tfidf_vector.toarray().flatten().argsort()[::-1]
    top_keywords = [feature_names[i] for i in sorted_indices[:top_n]]
    return top_keywords
    
    
def transcribe_korean_audio(file_path):
    model = whisper.load_model("base")  # You can try 'small', 'medium', or 'large' for better results
    result = model.transcribe(file_path, language="ko")  # 'ko' specifies Korean
    return result["text"]
def dummy_summarize(text):
    #keywords = extract_keywords(loaded_vectorizer, text)
    result = 'Summarized:  \n' + get_summary(text) #+ '  \nTop 3 Keywords: ' + ', '.join(keywords)
    return result

def save_summary(summary, date_key):
    if "history" not in st.session_state:
        st.session_state.history = {}
    if date_key not in st.session_state.history:
        st.session_state.history[date_key] = []
    st.session_state.history[date_key].append(summary)

def display_history():
    st.subheader("Summarization History")
    if "history" in st.session_state:
        for date_key in sorted(st.session_state.history.keys(), reverse=True):
            st.markdown(f"### {date_key}")
            for idx, summary in enumerate(st.session_state.history[date_key]):
                st.markdown(f"**Summary {idx + 1}:** {summary}")
    else:
        st.info("No history yet.")

# --- Streamlit UI Starts Here ---
st.title("🎙️ Voice-to-Text Summarizer (Korean)")

# Section 1: Input
st.header("Input Section: Voice Recording / Upload")
col1, col2 = st.columns(2)

# Record Button (Note: Streamlit does not support real-time recording natively)

# Upload Button
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None
if "file_changed" not in st.session_state:
    st.session_state.file_changed = False
    
with col1:
    st.write("### 📁 Upload Audio File")
    uploaded = st.file_uploader("Please upload a .wav or .mp3 file", type=["wav", "mp3"])
    if uploaded is not None:
        st.session_state.uploaded_file = uploaded.read()
        st.session_state.file_changed = True  # ✅ Mark file changed
        st.session_state.transcribed_text = ""

with col2:
    st.write("### 🎙️ Record Audio")
    if st.button("Go to Recorder"):
        st.switch_page("pages/Recording_Audio.py")
    if st.button("Recorded Audio"):
        try:
            with open("recorded_audio.wav", "rb") as f:
                st.session_state.uploaded_file = f.read()
            st.audio(st.session_state.uploaded_file)
            st.session_state.file_changed = True  # ✅ Mark file changed
            st.session_state.transcribed_text = "" 
        except FileNotFoundError:
            st.error("No recorded audio file found. Please record first.")
        
if st.session_state.uploaded_file and st.session_state.file_changed:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(st.session_state.uploaded_file)
        tmp_path = tmp_file.name

    st.session_state.transcribed_text = transcribe_korean_audio(tmp_path)
    os.remove(tmp_path)
    st.session_state.file_changed = False

# Section 2: Transcription
#edited_text = None
if "transcribed_text" in st.session_state and st.session_state.transcribed_text:
    st.header("📜 Transcription")
    st.session_state.edited_text = st.text_area(
        "Edit the transcribed text if needed:",
        value=st.session_state.transcribed_text,
        height=200
    )
    
# Section 3: Summarization
st.header("🧾 Summarization")
if st.button("Generate Summary"):
    if "edited_text" in st.session_state and st.session_state.edited_text:
        summary = dummy_summarize(st.session_state.edited_text)
        st.session_state.latest_summary = summary

        today = datetime.today().strftime("%Y-%m-%d")
        save_summary(summary, today)
# Display summarized output if available
if "latest_summary" in st.session_state:
    st.subheader("🧍 Summarized / Keywords Output (in Korean)")
    st.success(st.session_state.latest_summary)
       
        
# Initialize the session state variable
if "show_history" not in st.session_state:
    st.session_state.show_history = False

# Show or hide content based on state
if st.session_state.show_history:
    display_history()
    if st.button("⬆️ Hide Summary"):
        st.session_state.show_history = False
else:
    if st.button("Summary History"):
        st.session_state.show_history = True
