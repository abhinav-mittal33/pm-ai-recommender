# PM-internAI — Internship Recommender

## Summary
A Flask web app that recommends internships from `data/internships.csv`.  
- Uses `sentence-transformers` embeddings when available (for semantic matching).  
- Falls back to keyword-based ranking if embeddings are unavailable.  
- Multi-language support: UI labels are pre-translated, internship text is translated using `deep-translator` (preferred) with `googletrans` as fallback.  

---

## Recommended Environment
- **Preferred Python version:** 3.11 or 3.12 for smoothest install.  
- Python 3.13 also works but may trigger builds (Rust/Cargo). If simplicity matters (e.g. for judges), use 3.11.  

---

## Project Structure
pm-ai-recommender/
├── app.py
├── requirements.txt
├── README.md
├── data/
│ └── internships.csv
├── templates/
│ ├── index.html
│ └── admin.html
├── static/ (optional)
└── models/ (optional)
└── internships_emb.npy # precomputed embeddings (ship this for faster runs)

yaml
Copy code

---

## Installation (Windows PowerShell)
```powershell
cd "C:\path\to\pm-ai-recommender"

# 1) Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2) Upgrade pip tooling
python -m pip install --upgrade pip setuptools wheel

# 3) Install CPU-only torch (easiest for demo)
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.8.0

# 4) Install project dependencies
pip install -r requirements.txt

# 5) Run app
python app.py
# → open http://127.0.0.1:5000 in your browser
Installation (macOS / Linux)
bash
Copy code
cd /path/to/pm-ai-recommender

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.8.0
pip install -r requirements.txt

python app.py
# → open http://127.0.0.1:5000
Requirements
requirements.txt should contain:

ini
Copy code
flask==3.1.2
pandas==2.3.2
numpy==2.3.2
scikit-learn==1.7.1
sentence-transformers==2.2.2
transformers==4.56.1
huggingface-hub==0.34.4
tokenizers==0.22.0
torch==2.8.0
python-docx==1.2.0
docx2txt==0.9
PyMuPDF==1.26.4
deep-translator==1.11.4
googletrans==4.0.0rc1
langdetect==1.0.9
tqdm==4.67.1
Notes:

deep-translator is the primary translator, googletrans is fallback.

safetensors is excluded to avoid Rust build issues on Windows.

Precompute embeddings and include models/internships_emb.npy if you want faster demo startup.

Running the App
With venv active:

bash
Copy code
python app.py
Then visit http://127.0.0.1:5000.

Fixing VS Code "Import not resolved"
Press Ctrl+Shift+P → Python: Select Interpreter → pick .venv\Scripts\python.exe.

Restart VS Code window.

(Optional) Add .vscode/settings.json:

json
Copy code
{
  "python.defaultInterpreterPath": ".venv\\Scripts\\python.exe",
  "python.analysis.extraPaths": [
    ".venv\\Lib\\site-packages"
  ]
}
Troubleshooting
1) ImportError: cannot import name 'cached_download' from 'huggingface_hub'
Mismatch between sentence-transformers and huggingface-hub. Fix:

bash
Copy code
pip uninstall -y huggingface-hub sentence-transformers transformers tokenizers
pip install "huggingface-hub==0.34.4" "transformers==4.56.1" "sentence-transformers==2.2.2" "tokenizers==0.22.0"
2) safetensors or tokenizers fail with Rust errors
Avoid installing safetensors. If you must: install Rust via https://rustup.rs.

3) No recommendations after CV upload

Check Flask logs in terminal (your app prints debug info on uploaded files).

debug_text also shows resume length.

Use curl -F "resume=@resume.pdf" http://127.0.0.1:5000 to test file upload.

4) Import errors in VS Code
Ensure the correct interpreter (.venv) is selected.

Minimal Demo (No embeddings)
If judges face install issues with ML libs, run without embeddings:

bash
Copy code
pip install flask pandas python-docx docx2txt PyMuPDF deep-translator googletrans
python app.py
App will warn: sentence-transformers not available; embeddings disabled. but still works (keyword-only matching).

Tips for Judges
Prefer Python 3.11 (most compatible).

Always run inside a virtual environment.

If install is slow, skip embeddings and run in minimal demo mode.

If you include models/internships_emb.npy in the zip, startup will be instant (no need to encode).