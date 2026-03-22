# DocTree AI — Hierarchical Document Clusterer

A full-stack AI-powered document clustering web app built with Python, Flask, scikit-learn, and Llama 3.3 70b.

## Features
- 🔐 User authentication (register / login / sessions)
- 📁 Upload TXT, CSV, PDF files
- 🧠 TF-IDF vectorization + hierarchical clustering (scikit-learn + scipy)
- 🤖 AI cluster labeling via Groq (Llama 3.3 70b) — FREE
- 🌲 Interactive zoomable D3.js dendrogram tree
- 📊 Cards view + Statistics view with coherence scores
- ✏️ Rename clusters inline
- 🔍 Search across clusters
- 📄 Export PDF report (print dialog)
- ↓ Export SVG tree
- 📧 Email results (optional, requires Gmail setup)
- 💾 Full history dashboard — all runs saved per user
- 🌓 Dark / Light mode with preference saved per user

## Stack
- **Backend**: Python, Flask, Flask-Login, Flask-SQLAlchemy, SQLite
- **ML**: scikit-learn (TF-IDF), scipy (hierarchical clustering)
- **AI**: Groq API (Llama 3.3 70b) — free tier
- **Frontend**: Vanilla HTML/CSS/JS, D3.js

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure environment
```bash
cp .env.example .env
# Edit .env — set GROQ_API_KEY and SECRET_KEY
```

### 3. Run locally
```bash
python app.py
```
Open http://localhost:3000

## Deploy to Render.com (free)
1. Push to GitHub
2. New Web Service on render.com
3. Build command: `pip install -r requirements.txt`
4. Start command: `gunicorn app:app`
5. Add environment variables: GROQ_API_KEY, SECRET_KEY
6. Deploy → get live URL!
