import os, json, re, smtplib
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv
import requests as http_requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
import numpy as np

load_dotenv()

app = Flask(__name__, static_folder='public')
app.secret_key = os.getenv('SECRET_KEY', 'doctree-secret-change-in-production')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///doctree.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024

CORS(app)
db = SQLAlchemy(app)
login_manager = LoginManager(app)

GROQ_API_KEY = os.getenv('GROQ_API_KEY')
GROQ_MODEL   = 'llama-3.3-70b-versatile'
GROQ_URL     = 'https://api.groq.com/openai/v1/chat/completions'

# ── Database Models ────────────────────────────────────────────────────────────
class User(UserMixin, db.Model):
    id            = db.Column(db.Integer, primary_key=True)
    name          = db.Column(db.String(100), nullable=False)
    email         = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    theme         = db.Column(db.String(10), default='dark')
    created_at    = db.Column(db.DateTime, default=datetime.utcnow)
    runs          = db.relationship('ClusterRun', backref='user', lazy=True, cascade='all, delete-orphan')

    def set_password(self, pw):   self.password_hash = generate_password_hash(pw)
    def check_password(self, pw): return check_password_hash(self.password_hash, pw)

class ClusterRun(db.Model):
    id           = db.Column(db.Integer, primary_key=True)
    user_id      = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    title        = db.Column(db.String(200), nullable=False)
    num_docs     = db.Column(db.Integer)
    num_clusters = db.Column(db.Integer)
    method       = db.Column(db.String(20))
    mode         = db.Column(db.String(20), default='text')
    result_json  = db.Column(db.Text)
    created_at   = db.Column(db.DateTime, default=datetime.utcnow)

@login_manager.user_loader
def load_user(uid): return User.query.get(int(uid))

with app.app_context():
    db.create_all()

# ── Helpers ────────────────────────────────────────────────────────────────────
def groq_chat(system_msg, user_msg, max_tokens=1500):
    resp = http_requests.post(GROQ_URL, headers={
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {GROQ_API_KEY}'
    }, json={
        'model': GROQ_MODEL, 'temperature': 0.3, 'max_tokens': max_tokens,
        'messages': [{'role':'system','content':system_msg},{'role':'user','content':user_msg}]
    }, timeout=30)
    resp.raise_for_status()
    return resp.json()['choices'][0]['message']['content'].strip()

def build_tree(linkage_matrix, node_id, n_leaves):
    if node_id < n_leaves:
        return {'id': f'leaf_{node_id}', 'docIdx': node_id, 'leaves': [node_id], 'children': None, 'distance': 0.0}
    row   = linkage_matrix[int(node_id) - n_leaves]
    left  = build_tree(linkage_matrix, int(row[0]), n_leaves)
    right = build_tree(linkage_matrix, int(row[1]), n_leaves)
    return {'id': f'node_{int(node_id)}', 'docIdx': None,
            'leaves': left['leaves'] + right['leaves'],
            'children': [left, right], 'distance': float(row[2])}

def tag_nodes(node, top_sets, labels):
    if not node: return
    leaf_set = set(node.get('leaves') or [])
    for i, ts in enumerate(top_sets):
        if ts['size'] == len(leaf_set) and all(l in ts['set'] for l in leaf_set):
            node['isTop'] = True; node['clusterIdx'] = i; node['aiLabel'] = labels[i]; break
    if node.get('children'):
        for c in node['children']: tag_nodes(c, top_sets, labels)

def compute_coherence(vectors, indices):
    if len(indices) < 2: return 1.0
    vecs  = np.array([vectors[i] for i in indices])
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1
    nvecs = vecs / norms
    sim   = nvecs @ nvecs.T
    n     = len(indices)
    return float(np.clip((sim.sum() - n) / (n * (n - 1)), 0, 1))

# ── SMART DATA DETECTOR ────────────────────────────────────────────────────────
def detect_numeric_data(texts):
    """
    Detect if texts contain structured numeric data (like student marks).
    Returns (is_numeric, parsed_records) where parsed_records is a list of dicts.
    """
    numeric_pattern = re.compile(r'scored\s+(\d+)\s+in\s+([\w\s]+?)(?:,|\.)')
    name_pattern    = re.compile(r'^([A-Za-z\s]+?)\s+scored')
    total_pattern   = re.compile(r'Total score is (\d+)')

    records = []
    for text in texts:
        subjects = numeric_pattern.findall(text)
        name_m   = name_pattern.match(text.strip())
        total_m  = total_pattern.search(text)
        if len(subjects) >= 3 and name_m:
            record = {
                'name':    name_m.group(1).strip(),
                'total':   int(total_m.group(1)) if total_m else sum(int(s[0]) for s in subjects),
                'subjects': {s[1].strip(): int(s[0]) for s in subjects}
            }
            records.append(record)

    return len(records) >= len(texts) * 0.7, records

def extract_numeric_vectors(records):
    """Convert student records to numpy feature vectors."""
    if not records: return np.array([])
    subject_keys = list(records[0]['subjects'].keys())
    vectors = []
    for r in records:
        vec = [r['subjects'].get(k, 0) for k in subject_keys] + [r['total']]
        vectors.append(vec)
    return np.array(vectors, dtype=float), subject_keys

def get_performance_tier(total, max_total):
    pct = (total / max_total) * 100 if max_total > 0 else 0
    if pct >= 70:   return 'High Performer'
    if pct >= 55:   return 'Good Performer'
    if pct >= 40:   return 'Average Performer'
    return 'Needs Support'

def find_weak_subjects(record, subject_keys, passing_pcts):
    weak = []
    for sk in subject_keys:
        score   = record['subjects'].get(sk, 0)
        passing = passing_pcts.get(sk, 40)
        if score < passing:
            weak.append(sk)
    return weak

# ── AUTH routes ────────────────────────────────────────────────────────────────
@app.route('/api/auth/register', methods=['POST'])
def register():
    d = request.get_json()
    name = d.get('name','').strip(); email = d.get('email','').strip().lower(); pw = d.get('password','')
    if not name or not email or not pw: return jsonify({'error': 'All fields are required'}), 400
    if len(pw) < 6: return jsonify({'error': 'Password must be at least 6 characters'}), 400
    if User.query.filter_by(email=email).first(): return jsonify({'error': 'Email already registered'}), 409
    user = User(name=name, email=email); user.set_password(pw)
    db.session.add(user); db.session.commit()
    login_user(user, remember=True)
    return jsonify({'ok': True, 'user': {'id': user.id, 'name': user.name, 'email': user.email, 'theme': user.theme}})

@app.route('/api/auth/login', methods=['POST'])
def login():
    d = request.get_json()
    email = d.get('email','').strip().lower(); pw = d.get('password','')
    user = User.query.filter_by(email=email).first()
    if not user or not user.check_password(pw): return jsonify({'error': 'Invalid email or password'}), 401
    login_user(user, remember=True)
    return jsonify({'ok': True, 'user': {'id': user.id, 'name': user.name, 'email': user.email, 'theme': user.theme}})

@app.route('/api/auth/logout', methods=['POST'])
@login_required
def logout():
    logout_user(); return jsonify({'ok': True})

@app.route('/api/auth/me')
def me():
    if current_user.is_authenticated:
        return jsonify({'ok': True, 'user': {'id': current_user.id, 'name': current_user.name, 'email': current_user.email, 'theme': current_user.theme}})
    return jsonify({'ok': False}), 401

@app.route('/api/auth/theme', methods=['POST'])
@login_required
def set_theme():
    current_user.theme = request.get_json().get('theme', 'dark')
    db.session.commit(); return jsonify({'ok': True})

# ── HISTORY ────────────────────────────────────────────────────────────────────
@app.route('/api/history')
@login_required
def history():
    runs = ClusterRun.query.filter_by(user_id=current_user.id).order_by(ClusterRun.created_at.desc()).limit(50).all()
    return jsonify({'runs': [{'id':r.id,'title':r.title,'num_docs':r.num_docs,'num_clusters':r.num_clusters,'method':r.method,'mode':r.mode,'created_at':r.created_at.isoformat()} for r in runs]})

@app.route('/api/history/<int:run_id>')
@login_required
def history_detail(run_id):
    run = ClusterRun.query.filter_by(id=run_id, user_id=current_user.id).first()
    if not run: return jsonify({'error': 'Not found'}), 404
    return jsonify({'run': json.loads(run.result_json)})

@app.route('/api/history/<int:run_id>', methods=['DELETE'])
@login_required
def delete_run(run_id):
    run = ClusterRun.query.filter_by(id=run_id, user_id=current_user.id).first()
    if not run: return jsonify({'error': 'Not found'}), 404
    db.session.delete(run); db.session.commit(); return jsonify({'ok': True})

@app.route('/api/history/<int:run_id>/rename-cluster', methods=['POST'])
@login_required
def rename_cluster(run_id):
    run = ClusterRun.query.filter_by(id=run_id, user_id=current_user.id).first()
    if not run: return jsonify({'error': 'Not found'}), 404
    d = request.get_json(); idx = d.get('clusterIdx'); new_label = d.get('label','').strip()
    if not new_label: return jsonify({'error': 'Label required'}), 400
    result = json.loads(run.result_json)
    if 0 <= idx < len(result.get('labels', [])):
        result['labels'][idx]['label'] = new_label
        run.result_json = json.dumps(result); db.session.commit()
    return jsonify({'ok': True})

# ── FILE UPLOAD ────────────────────────────────────────────────────────────────
@app.route('/api/upload', methods=['POST'])
@login_required
def upload_file():
    if 'file' not in request.files: return jsonify({'error': 'No file provided'}), 400
    f = request.files['file']; name = f.filename.lower()
    try:
        if name.endswith('.txt'):
            text = f.read().decode('utf-8', errors='ignore')
        elif name.endswith('.csv'):
            import csv, io
            content = f.read().decode('utf-8', errors='ignore')
            reader  = csv.reader(io.StringIO(content))
            rows    = [' '.join(row) for row in reader if any(row)]
            text    = '\n\n'.join(rows)
        elif name.endswith('.pdf'):
            import io
            pdf_bytes = f.read()

            # ── Try pdfplumber first (smart table reader) ──────────────────
            try:
                import pdfplumber

                texts_out = []
                with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                    for page in pdf.pages:
                        tables = page.extract_tables()
                        if tables:
                            for table in tables:
                                # Find header row (contains subject names)
                                header_row = None
                                subject_cols = {}
                                for ri, row in enumerate(table):
                                    if row and any(cell and ('math' in str(cell).lower() or
                                                             'algorithm' in str(cell).lower() or
                                                             'discrete' in str(cell).lower() or
                                                             'computer' in str(cell).lower() or
                                                             'managerial' in str(cell).lower() or
                                                             'science' in str(cell).lower() or
                                                             'english' in str(cell).lower() or
                                                             'physics' in str(cell).lower())
                                                   for cell in row):
                                        header_row = ri
                                        # Map column index to subject name
                                        for ci, cell in enumerate(row):
                                            if cell and len(str(cell).strip()) > 3:
                                                subject_cols[ci] = str(cell).strip().replace('\n', ' ')
                                        break

                                if header_row is not None and subject_cols:
                                    # Find name column (look for column with names)
                                    name_col = None
                                    for ri in range(header_row + 1, min(header_row + 5, len(table))):
                                        row = table[ri]
                                        if not row: continue
                                        for ci, cell in enumerate(row):
                                            if cell and re.search(r'[A-Za-z]{3,}', str(cell)) and ci not in subject_cols:
                                                name_col = ci
                                                break
                                        if name_col is not None: break

                                    # Extract each student row
                                    for row in table[header_row + 1:]:
                                        if not row or all(not c for c in row): continue

                                        # Get student name
                                        student_name = None
                                        if name_col is not None and name_col < len(row):
                                            cell = str(row[name_col] or '').strip()
                                            # Extract name from enrollment format like "MU123 [Name Here]"
                                            bracket_match = re.search(r'\[([^\]]+)\]', cell)
                                            if bracket_match:
                                                student_name = bracket_match.group(1).strip()
                                            elif re.search(r'[A-Za-z]{3,}', cell):
                                                student_name = cell

                                        if not student_name:
                                            # Try to find name in any cell
                                            for ci, cell in enumerate(row):
                                                if cell and ci not in subject_cols:
                                                    bracket_match = re.search(r'\[([^\]]+)\]', str(cell))
                                                    if bracket_match:
                                                        student_name = bracket_match.group(1).strip()
                                                        break

                                        if not student_name: continue

                                        # Get scores for each subject
                                        score_parts = []
                                        total = 0
                                        valid = True
                                        for ci, subj in subject_cols.items():
                                            if ci < len(row):
                                                val = str(row[ci] or '').strip()
                                                try:
                                                    score = int(float(val))
                                                    score_parts.append(f'{score} in {subj}')
                                                    total += score
                                                except:
                                                    valid = False
                                                    break

                                        if valid and score_parts and total > 0:
                                            desc = f'{student_name} scored ' + ', '.join(score_parts) + f'. Total score is {total}.'
                                            texts_out.append(desc)

                # If table extraction found students, use them
                if len(texts_out) >= 3:
                    return jsonify({'ok': True, 'texts': texts_out[:150], 'filename': f.filename,
                                    'message': f'Extracted {len(texts_out)} student records from PDF table'})

                # Fallback: extract plain text from PDF
                with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                    all_text = '\n\n'.join(page.extract_text() or '' for page in pdf.pages)
                paragraphs = [p.strip() for p in re.split(r'\n{2,}', all_text) if len(p.strip()) > 40]
                if paragraphs:
                    return jsonify({'ok': True, 'texts': paragraphs[:150], 'filename': f.filename})

            except ImportError:
                pass  # Fall through to pypdf

            # ── Fallback to pypdf ──────────────────────────────────────────
            try:
                import pypdf
                reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
                text   = '\n\n'.join(page.extract_text() or '' for page in reader.pages)
                paragraphs = [p.strip() for p in re.split(r'\n{2,}', text) if len(p.strip()) > 40]
                if not paragraphs:
                    sentences  = re.split(r'(?<=[.!?])\s+', text)
                    paragraphs = [s.strip() for s in sentences if len(s.strip()) > 40]
                return jsonify({'ok': True, 'texts': paragraphs[:150], 'filename': f.filename})
            except ImportError:
                return jsonify({'error': 'PDF support requires pdfplumber. Run: pip install pdfplumber'}), 400

        else:
            return jsonify({'error': 'Only .txt, .csv, and .pdf files supported'}), 400

        paragraphs = [p.strip() for p in re.split(r'\n{2,}', text) if len(p.strip()) > 40]
        if not paragraphs:
            sentences  = re.split(r'(?<=[.!?])\s+', text)
            paragraphs = [s.strip() for s in sentences if len(s.strip()) > 40]

        return jsonify({'ok': True, 'texts': paragraphs[:150], 'filename': f.filename})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ── CLUSTER endpoint (handles both text + numeric) ────────────────────────────
@app.route('/api/cluster', methods=['POST'])
@login_required
def cluster():
    data   = request.get_json()
    texts  = data.get('texts', [])
    k      = min(int(data.get('k', 3)), len(texts))
    method = data.get('method', 'average')

    if len(texts) < 2: return jsonify({'error': 'Need at least 2 documents'}), 400

    # ── Detect if data is numeric (student marks etc.) ──────────────────────
    is_numeric, records = detect_numeric_data(texts)

    if is_numeric and len(records) >= 5:
        # ── NUMERIC MODE: K-Means on actual scores ───────────────────────────
        vectors_raw, subject_keys = extract_numeric_vectors(records)
        scaler  = StandardScaler()
        vectors = scaler.fit_transform(vectors_raw)

        # K-Means clustering
        kmeans    = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels_arr = kmeans.fit_predict(vectors)

        # Sort clusters by average total score (highest first)
        cluster_totals = {}
        for idx, label in enumerate(labels_arr):
            cluster_totals.setdefault(int(label), []).append(records[idx]['total'])
        cluster_avgs   = {c: np.mean(t) for c, t in cluster_totals.items()}
        sorted_clusters = sorted(cluster_avgs.keys(), key=lambda c: -cluster_avgs[c])
        remap = {old: new for new, old in enumerate(sorted_clusters)}
        labels_arr = np.array([remap[l] for l in labels_arr])

        # Build cluster groups
        clusters = {}
        for idx, label in enumerate(labels_arr):
            clusters.setdefault(int(label), []).append(idx)

        # Max possible total
        max_total = max(r['total'] for r in records) if records else 270

        # Compute passing thresholds per subject (40% of max per subject)
        subject_maxes = {}
        for sk in subject_keys:
            scores = [r['subjects'].get(sk, 0) for r in records]
            subject_maxes[sk] = max(scores) if scores else 60
        passing_pcts = {sk: subject_maxes[sk] * 0.4 for sk in subject_keys}

        # Build top_clusters with rich stats
        top_clusters = []
        for cid in sorted(clusters.keys()):
            idxs = clusters[cid]
            cluster_records  = [records[i] for i in idxs]
            cluster_totals_v = [r['total'] for r in cluster_records]
            avg_total        = float(np.mean(cluster_totals_v))
            avg_pct          = (avg_total / max_total) * 100

            # Per-subject averages
            subject_avgs = {}
            for sk in subject_keys:
                subject_avgs[sk] = float(np.mean([r['subjects'].get(sk,0) for r in cluster_records]))

            # Find weakest subject
            weakest_subject = min(subject_avgs, key=subject_avgs.get) if subject_avgs else ''

            # At-risk students (below passing in 2+ subjects)
            at_risk = []
            for i in idxs:
                weak = find_weak_subjects(records[i], subject_keys, passing_pcts)
                if len(weak) >= 2:
                    at_risk.append({'name': records[i]['name'], 'weak_subjects': weak, 'total': records[i]['total']})

            # Top performers in cluster
            top_in_cluster = sorted(idxs, key=lambda i: records[i]['total'], reverse=True)[:3]
            top_names = [records[i]['name'] for i in top_in_cluster]

            top_clusters.append({
                'clusterLabel':   cid,
                'docIndices':     idxs,
                'coherence':      round(compute_coherence(vectors, idxs), 3),
                'avgTotal':       round(avg_total, 1),
                'avgPct':         round(avg_pct, 1),
                'subjectAvgs':    {k: round(v,1) for k,v in subject_avgs.items()},
                'weakestSubject': weakest_subject,
                'atRisk':         at_risk,
                'topStudents':    top_names,
                'minTotal':       int(min(cluster_totals_v)),
                'maxTotal':       int(max(cluster_totals_v)),
            })

        # Build hierarchical tree from K-Means results using linkage
        dist_matrix    = pdist(vectors, metric='euclidean')
        linkage_matrix = linkage(dist_matrix, method=method)
        n              = len(texts)
        root           = build_tree(linkage_matrix, 2 * n - 2, n)

        return jsonify({
            'root':        root,
            'topClusters': top_clusters,
            'mode':        'numeric',
            'subjectKeys': subject_keys,
            'records':     records
        })

    else:
        # ── TEXT MODE: TF-IDF + Hierarchical Clustering ──────────────────────
        vectorizer   = TfidfVectorizer(stop_words='english', max_features=5000, sublinear_tf=True, min_df=1)
        tfidf_matrix = vectorizer.fit_transform(texts)
        vectors      = normalize(tfidf_matrix, norm='l2').toarray()

        dist_matrix    = pdist(vectors, metric='cosine')
        linkage_matrix = linkage(dist_matrix, method=method)
        labels_arr     = fcluster(linkage_matrix, k, criterion='maxclust')

        clusters = {}
        for idx, label in enumerate(labels_arr):
            clusters.setdefault(int(label), []).append(idx)

        top_clusters = [{'clusterLabel': cid, 'docIndices': idxs,
                         'coherence': round(compute_coherence(vectors, idxs), 3)}
                        for cid, idxs in clusters.items()]

        n    = len(texts)
        root = build_tree(linkage_matrix, 2 * n - 2, n)

        return jsonify({'root': root, 'topClusters': top_clusters, 'mode': 'text'})

# ── LABEL endpoint ─────────────────────────────────────────────────────────────
@app.route('/api/label-clusters', methods=['POST'])
@login_required
def label_clusters():
    if not GROQ_API_KEY: return jsonify({'error': 'GROQ_API_KEY not set'}), 500

    data         = request.get_json()
    clusters     = data.get('clusters', [])
    texts        = data.get('texts', [])
    title        = data.get('title', f'Run {datetime.utcnow().strftime("%b %d %H:%M")}')
    root         = data.get('root')
    top_clusters = data.get('topClusters', [])
    method       = data.get('method', 'average')
    mode         = data.get('mode', 'text')
    records      = data.get('records', [])
    subject_keys = data.get('subjectKeys', [])

    labels = []

    if mode == 'numeric' and top_clusters:
        # ── Generate smart labels for numeric/student clusters ───────────────
        for i, c in enumerate(top_clusters):
            avg_pct  = c.get('avgPct', 0)
            avg_tot  = c.get('avgTotal', 0)
            n_docs   = len(c.get('docIndices', []))
            weakest  = c.get('weakestSubject', '')
            at_risk  = c.get('atRisk', [])
            top_stu  = c.get('topStudents', [])
            min_t    = c.get('minTotal', 0)
            max_t    = c.get('maxTotal', 0)

            # Determine tier
            if avg_pct >= 68:
                tier    = 'High Performers'
                summary = f'Students consistently excelling across all subjects with strong performance in every area.'
            elif avg_pct >= 52:
                tier    = 'Good Performers'
                summary = f'Students performing above average with solid understanding of core subjects.'
            elif avg_pct >= 38:
                tier    = 'Average Performers'
                summary = f'Students performing at a moderate level with room for improvement especially in {weakest}.'
            else:
                tier    = 'Needs Academic Support'
                summary = f'Students struggling significantly and requiring immediate academic intervention and support.'

            keywords = [tier.split()[0], 'score', weakest.split()[0] if weakest else 'performance',
                       f'{round(avg_pct)}%', f'{n_docs} students']

            labels.append({
                'label':    tier,
                'summary':  summary,
                'keywords': keywords[:4],
                'avgPct':   round(avg_pct, 1),
                'avgTotal': round(avg_tot, 1),
                'atRisk':   at_risk,
                'topStudents': top_stu,
                'minTotal': min_t,
                'maxTotal': max_t,
                'subjectAvgs': c.get('subjectAvgs', {}),
                'weakestSubject': weakest
            })

    else:
        # ── AI labels for text clusters ──────────────────────────────────────
        prompt_parts = []
        for i, c in enumerate(clusters):
            snippets = '\n---\n'.join(t[:300] for t in c['docTexts'])
            prompt_parts.append(f"Cluster {i+1} ({len(c['docTexts'])} documents):\n{snippets}")

        system_msg = (f"You are a document clustering expert. Respond ONLY with a JSON array, no markdown. "
                      f"Each element: {{\"label\":\"2-4 word topic\",\"summary\":\"one sentence\","
                      f"\"keywords\":[\"w1\",\"w2\",\"w3\",\"w4\"]}} Return exactly {len(clusters)} objects.")
        raw    = groq_chat(system_msg, f"Label these {len(clusters)} clusters:\n\n" + '\n\n===\n\n'.join(prompt_parts))
        labels = json.loads(raw.replace('```json','').replace('```','').strip())

    # Tag tree nodes
    top_sets = [{'set': set(c['docIndices']), 'size': len(c['docIndices'])} for c in top_clusters]
    tag_nodes(root, top_sets, labels)

    # Save to database
    result = {
        'title': title, 'texts': texts, 'root': root,
        'topClusters': top_clusters, 'labels': labels,
        'method': method, 'mode': mode,
        'records': records, 'subjectKeys': subject_keys,
        'created_at': datetime.utcnow().isoformat()
    }
    run = ClusterRun(user_id=current_user.id, title=title,
                     num_docs=len(texts), num_clusters=len(clusters or top_clusters),
                     method=method, mode=mode, result_json=json.dumps(result))
    db.session.add(run); db.session.commit()

    return jsonify({'labels': labels, 'root': root, 'run_id': run.id, 'mode': mode})

# ── EMAIL ──────────────────────────────────────────────────────────────────────
@app.route('/api/email-results', methods=['POST'])
@login_required
def email_results():
    d        = request.get_json()
    to_email = d.get('email', current_user.email)
    labels   = d.get('labels', [])
    title    = d.get('title', 'DocTree Results')
    smtp_user = os.getenv('SMTP_USER'); smtp_pass = os.getenv('SMTP_PASS')
    if not smtp_user or not smtp_pass:
        return jsonify({'error': 'Email not configured. Add SMTP_USER and SMTP_PASS to .env'}), 400
    body_lines = [f"<h2>{title}</h2><p>Your clustering results:</p><hr>"]
    for i, lbl in enumerate(labels):
        body_lines.append(f"<h3 style='color:#2d7d32'>Cluster {i+1}: {lbl.get('label','')}</h3>"
                          f"<p>{lbl.get('summary','')}</p><hr>")
    msg = MIMEMultipart('alternative')
    msg['Subject'] = f'DocTree Results: {title}'; msg['From'] = smtp_user; msg['To'] = to_email
    msg.attach(MIMEText('\n'.join(body_lines), 'html'))
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as s:
            s.login(smtp_user, smtp_pass); s.sendmail(smtp_user, to_email, msg.as_string())
        return jsonify({'ok': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ── HEALTH ─────────────────────────────────────────────────────────────────────
@app.route('/api/health')
def health():
    return jsonify({'ok': True, 'apiKeyConfigured': bool(GROQ_API_KEY)})

# ── SERVE FRONTEND ─────────────────────────────────────────────────────────────
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    port = int(os.getenv('PORT', 3000))
    print(f'\n🌿 DocTree running at http://localhost:{port}')
    print(f'{"✅ Groq key loaded" if GROQ_API_KEY else "⚠️  GROQ_API_KEY missing"}\n')
    app.run(host='0.0.0.0', port=port, debug=False)
