# ⚖️ LawMate – AI-Powered Legal Advisor for Indian Laws  

LawMate is an **AI-driven legal assistant** designed to answer queries based on **Indian laws**. It can process **PDFs and images**, extract legal information, build a **semantic search knowledge base**, and let users **chat with an AI attorney** that cites relevant sections of law.  

---

## 🚀 Key Features  

- 📄 **Upload PDF & Image Files** (PNG, JPG, JPEG)  
- 🔎 **Text Extraction** from documents using **PyPDF2 & Tesseract OCR**  
- 📚 **Semantic Search** with **FAISS Vector Store**  
- 🤖 **AI-Powered Q&A** using **Google Gemini** with legal context  
- ⚖️ **Contextual Legal Advice** with relevant section citations  
- 💬 **Persistent Chat History**  
- 📂 **Fallback Local Dataset** (if no files are uploaded)  

---

## 🛠️ Tech Stack

- Python
- Streamlit
- LangChain
- FAISS (Vector Store)
- Google Generative AI (Gemini)
- Tesseract OCR
- PyPDF2
- PIL (Pillow) 

---

## ⚡ Getting Started  

### 1️⃣ Clone the Repository  
```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
````

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Set Up Environment Variables

* Create a `.env` file in the root directory
* Add your **Google API Key**:

```plaintext
GOOGLE_API_KEY=your_google_api_key_here
```

---

## 🔑 Example `.env` File

```plaintext
# Rename this file to ".env" and add your Google API Key
GOOGLE_API_KEY=your_google_api_key_here
```

---

## 📜 Requirements

```plaintext
streamlit
pytesseract
Pillow
PyPDF2
python-dotenv
langchain
langchain-google-genai
langchain-community
faiss-cpu
google-generativeai
```

---

## 🔧 Tesseract OCR Setup

1. Download & Install **Tesseract OCR**:
   👉 [Tesseract OCR GitHub Releases](https://github.com/tesseract-ocr/tesseract)

2. Update the path in `app.py` if needed:

```python
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe"
```

---

## ▶️ Run the App

```bash
streamlit run app.py
```

---

## 📂 Project Structure

```plaintext
your-project/
│
├── app.py
├── README.md
├── requirements.txt
├── .env.example
└── dataset/   # optional - keep default PDFs here
```

---

## 📜 License

This project is licensed under the **MIT License**.

---

## ❤️ Acknowledgments

* **Google Generative AI** (Gemini)
* **LangChain**
* **Streamlit Community**
* **Tesseract OCR Project**

---

## ⚡ GitHub Commands to Push

```bash
# Initialize Git
git init

# Add all files
git add .

# Commit changes
git commit -m "Initial commit - LawMate AI Legal Advisor"

# Create GitHub repo (from GitHub UI)

# Link local repo to remote
git remote add origin https://github.com/<your-username>/<your-repo-name>.git

# Push code
git branch -M main
git push -u origin main
