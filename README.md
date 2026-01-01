# 🔮 Next Word Prediction using LSTM & GRU

This project is a **Next Word Prediction system** built using **Deep Learning (LSTM & GRU)** and deployed with **Streamlit**. The model predicts the most probable next word based on a given input text sequence, similar to how language models autocomplete sentences.

---

## 🚀 Features
- 📚 Trained on sequential text data
- 🧠 Supports **LSTM** and **GRU** models
- 🔄 Dynamic model selection in the UI
- 🎯 Predicts the most likely next word using Softmax
- 🌐 Interactive **Streamlit Web App**
- 🎨 Clean and user-friendly interface

---

## 🧠 How It Works
1. User inputs a sequence of words  
2. Text is tokenized and padded  
3. Selected model (LSTM / GRU) processes the sequence  
4. Model outputs probability distribution over vocabulary  
5. Word with highest probability is returned as prediction  

---

## 🛠 Tech Stack
- Python
- TensorFlow / Keras
- NumPy
- Streamlit
- Pickle (Tokenizer)

---

## 📂 Project Structure
Next-Word-Predictor/
├── app.py
├── models/
│   ├── next_word_LSTM.h5
│   └── next_word_GRU.h5
├── tokenizers/
│   └── tokenizer.pickle
├── requirements.txt
└── README.md


---

## ▶️ How to Run
```bash
pip install -r requirements.txt
streamlit run app.py
✨ Example

Input:
To be or not to

Prediction:
be
```

---

## 📌 Future Improvements

Add Beam Search

Train on larger corpus

Support sentence generation

Add attention mechanism

---

## 👤 Author

Karamjodh Singh
Aspiring Machine Learning Engineer | Deep Learning Enthusiast

---

⭐ If you like this project, consider giving it a star on GitHub!
