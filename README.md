# HealthExplain Pro 🏥

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B.svg)](https://streamlit.io/)

A powerful Streamlit application that simplifies complex medical text using LangChain and Groq AI, grounded in medical knowledge. Perfect for patients, medical students, and healthcare professionals who need to understand or explain complex medical concepts.

## 🌟 Features

- 📄 Upload medical PDFs for context
- ✍️ Enter complex medical text
- 🎯 Get simplified, patient-friendly explanations
- 📚 View source documents used for context
- ⚡ Fast generation using Groq's Qwen model
- 🔍 Vector search using FAISS and Sentence Transformers

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Groq API key ([Get one here](https://console.groq.com))
- Git

### Installation

1. Clone the repository:
```bash
git clone https://github.com/shoaibms27/healthexplain-pro.git
cd healthexplain-pro
```

2. Create and activate a virtual environment:
```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up your environment:
```bash
# Create .env file and add your Groq API key
echo "GROQ_API_KEY=your_key_here" > .env
```

## 💻 Usage

1. Start the Streamlit app:
```bash
streamlit run src/app.py
```

2. Open your browser to the displayed URL (usually http://localhost:8501)

3. Use the app:
   - Upload medical PDFs for additional context (optional)
   - Enter medical text in the input box
   - Click "Generate Explanation"
   - View the simplified explanation and sources

## 🏗️ Project Structure

```
.
├── src/
│   ├── app.py                 # Streamlit application
│   └── utils/
│       ├── pdf_processor.py   # PDF text extraction
│       ├── vector_store.py    # FAISS vector store
│       └── llm_chain.py       # LangChain + Groq integration
├── requirements.txt           # Python dependencies
├── .env.example              # Example environment variables
├── .gitignore               # Git ignore rules
├── LICENSE                  # MIT license
└── README.md               # This file
```

## 🛠️ Technologies Used

- [LangChain](https://python.langchain.com/) - Framework for LLM applications
- [Groq](https://groq.com/) - Fast LLM inference
- [Sentence Transformers](https://www.sbert.net/) - Text embeddings
- [FAISS](https://github.com/facebookresearch/faiss) - Vector similarity search
- [PyMuPDF](https://pymupdf.readthedocs.io/) - PDF processing
- [Streamlit](https://streamlit.io/) - Web interface

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Thanks to Groq for providing fast LLM inference
- Thanks to the LangChain team for their excellent framework
- Thanks to the medical community for their valuable feedback

## 📧 Contact

For questions and support, please [open an issue](https://github.com/shoaibms27/healthexplain-pro/issues) in the GitHub repository. 