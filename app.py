import streamlit as st
import numpy as np
import pickle as pkl
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Next Word Predictor",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Next Word Prediction")
st.caption("LSTM / GRU based Language Model")

# -----------------------------
# Sidebar – Model Selection
# -----------------------------
st.sidebar.header("⚙️ Model Settings")

model_choice = st.sidebar.selectbox(
    "Choose a model",
    ("LSTM Model", "GRU Model")
)

MODEL_PATHS = {
    "LSTM Model": "models/next_word_LSTM.h5",
    "GRU Model": "models/next_word_GRU.h5"
}

# -----------------------------
# Load tokenizer (cached)
# -----------------------------
@st.cache_resource
def load_tokenizer():
    with open("tokenizers/tokenizer.pickle", "rb") as handle:
        return pkl.load(handle)

tokenizer = load_tokenizer()

# -----------------------------
# Load model dynamically (cached)
# -----------------------------
@st.cache_resource
def load_selected_model(model_path):
    return load_model(model_path)

model = load_selected_model(MODEL_PATHS[model_choice])

# -----------------------------
# Prediction Function
# -----------------------------
def predict_next(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]

    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len - 1):]

    token_list = pad_sequences(
        [token_list],
        maxlen=max_sequence_len,
        padding="pre"
    )

    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)[0]

    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word

    return None

# -----------------------------
# Main UI
# -----------------------------
st.subheader("✍️ Enter your text")

input_text = st.text_input(
    "Input sequence",
    "To be or not to be"
)

if st.button("🔮 Predict Next Word"):
    max_sequence_len = model.input_shape[1]
    next_word = predict_next(model, tokenizer, input_text, max_sequence_len)

    if next_word:
        st.success(f"**Predicted Next Word:** `{next_word}`")
    else:
        st.error("Could not predict the next word.")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("Built with TensorFlow & Streamlit By Karamjodh Singh 🚀")