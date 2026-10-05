# JobMatch AI — ML-Powered Job Recommendation System

A full-stack job recommendation web application that trains a **Random Forest classifier**
on 10,000 real user skill profiles and ranks 11 job categories by how well they match
any given user's hard and soft skills.

```
┌─────────────────────────────────────────────────────────────────┐
│  Stack:   Python 3.10+  ·  Flask  ·  scikit-learn  ·  pandas   │
│  ML:      TF-IDF (5,000 features)  +  Random Forest (200 trees) │
│  Data:    User-data-10000.csv  +  jobs_data.csv                  │
│  UI:      Responsive HTML/CSS/JS — mobile, tablet, desktop       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Datasets](#3-datasets)
4. [Quick Start](#4-quick-start)
5. [Detailed Setup](#5-detailed-setup)
6. [Machine Learning Pipeline](#6-machine-learning-pipeline)
7. [Scoring and Ranking Algorithm](#7-scoring-and-ranking-algorithm)
8. [Flask Application and API Reference](#8-flask-application-and-api-reference)
9. [Frontend Architecture](#9-frontend-architecture)
10. [Responsive Design System](#10-responsive-design-system)
11. [Complete Request Response Workflow](#11-complete-request-response-workflow)
12. [Model Performance](#12-model-performance)
13. [Configuration and Customisation](#13-configuration-and-customisation)
14. [Troubleshooting](#14-troubleshooting)
15. [Extending the System](#15-extending-the-system)

---

## 1. Project Overview

### What It Does

JobMatch AI accepts a user's **hard skills** (technical abilities) and **soft skills**
(interpersonal traits), then uses a machine-learning model to predict which of 11 job
categories is the best match. Every category is given three independent scores:

| Score | Source | Weight |
|-------|--------|--------|
| **RF Probability** | Random Forest class probability | 60% |
| **Cosine Similarity** | TF-IDF vector similarity to job profile | 40% |
| **Blended Score** | Weighted combination of above | Final rank |

Beyond the score, the UI also shows which of the user's skills exactly match each job's
requirements, and which skills the user would need to develop to fully qualify.

### Why Two Scores?

The **Random Forest** has learned patterns from 10,000 real user profiles. It understands
which skill *combinations* historically correlate with each job field. **Cosine similarity**
measures direct vocabulary overlap between the user's skills and the job's listed requirements
— it rewards exact keyword matches. Blending both captures complementary signals: the RF
generalises from context while cosine rewards specificity.

---

## 2. Repository Structure

```
jobmatch2/
│
├── app.py                  ← Flask web server + all API routes
├── ml_engine.py            ← Full ML pipeline (train, infer, stats)
│
├── User-data-10000.csv     ← Training data: 10,000 user skill profiles
├── jobs_data.csv           ← Job catalogue: 11 categories with skill requirements
│
├── requirements.txt        ← Python dependencies
├── README.md               ← This file
│
└── templates/
    └── index.html          ← Single-page frontend (HTML + CSS + JS)
```

No additional directories, build steps, or compiled assets are needed.
The application is entirely self-contained in these six files.

---

## 3. Datasets

### 3.1 `jobs_data.csv` — Job Catalogue

**11 rows x 4 columns.** One row per job category.

| Column | Type | Description |
|--------|------|-------------|
| `Job ID` | int | Unique job identifier (1-11) |
| `Major` | string | Job category name (used as the class label) |
| `Hard Skills` | string | Python list literal of required technical skills |
| `Soft Skills` | string | Python list literal of required interpersonal skills |

**All 11 job categories:**

| ID | Category | Key Hard Skills |
|----|----------|-----------------|
| 1 | accounting | finance, excel, tax, erp, xero, reconciliation |
| 2 | administration & office support | myob, databases, booking, accounts payable |
| 3 | advertising, arts & media | digital media, publishing, journalism, analytics |
| 4 | healthcare & medical | nursing, mental health, rehabilitation, surgery |
| 5 | it jobs | java, python, sql, javascript, aws, linux |
| 6 | marketing | crm, google analytics, seo, copywriting |
| 7 | recruitment consultancy jobs | executive recruitment, supply chain, procurement |
| 8 | retail & consumer products | visual merchandising, inventory, stock management |
| 9 | sales | account management, business development, crm |
| 10 | sport & recreation | cpr, outdoor education, functional training |
| 11 | telecomunication | networking, windows, testing, project management |

Each job profile contains exactly **20 hard skills** and **10 soft skills**.

**Sample row (accounting):**
```
Hard Skills: ['business', 'finance', 'excel', 'tax', 'service', 'data', 'system',
              'accounts payable', 'application', 'sales', 'forecasting', ...]
Soft Skills: ['management', 'planning', 'operations', 'leadership', 'customer service',
              'time management', 'data entry', 'verbal communication skills', ...]
```

---

### 3.2 `User-data-10000.csv` — Training Data

**10,000 rows x 4 columns.** One row per user profile.

| Column | Type | Description |
|--------|------|-------------|
| `hard_skill` | string | Python list literal of the user's hard skills |
| `soft_skill` | string | Python list literal of the user's soft skills |
| `label` | int | Binary flag (0 = general, 1 = telecom-specific match) |
| `candidate_field` | string | Job category this user belongs to — the prediction target |

> **Note on the `label` column:** Label `1` exclusively marks users in the
> `telecomunication` field. The ML model does **not** use this binary label;
> it predicts `candidate_field` directly (9-class classification).

**Class distribution in training data:**

| Category | Users | Percentage |
|----------|-------|-----------|
| telecomunication | 2,382 | 23.8% |
| healthcare & medical | 1,831 | 18.3% |
| administration & office support | 1,526 | 15.3% |
| accounting | 1,477 | 14.8% |
| sales | 1,242 | 12.4% |
| retail & consumer products | 674 | 6.7% |
| marketing | 609 | 6.1% |
| advertising, arts & media | 140 | 1.4% |
| sport & recreation | 119 | 1.2% |

> **Important note on coverage:** The training data contains **9** distinct
> `candidate_field` values. The jobs catalogue has **11** entries.
> Two categories — `it jobs` and `recruitment consultancy jobs` — exist only
> in the catalogue. They receive zero RF probability but are still ranked
> using cosine similarity alone.

**Sample row:**
```
hard_skill:      ['business', 'merchandising', 'sales', 'service']
soft_skill:      ['customer service']
label:           0
candidate_field: retail & consumer products
```

---

## 4. Quick Start

```bash
# 1. Enter the project directory
cd jobmatch2

# 2. Install Python dependencies (Python 3.10+ required)
pip install -r requirements.txt

# 3. Run the server
python app.py

# 4. Open in browser
# Navigate to: http://localhost:5000
```

The application shows a loading screen while the model trains (approximately 15 seconds).
Once complete, the interface becomes fully interactive.

---

## 5. Detailed Setup

### 5.1 Prerequisites

| Requirement | Minimum Version | Notes |
|-------------|-----------------|-------|
| Python | 3.10 | Required for `list[str]` type hints |
| pip | 21.0 | Dependency resolution |

Verify your Python version:
```bash
python --version
# Python 3.10.x or higher
```

### 5.2 Install Dependencies

```bash
pip install -r requirements.txt
```

**`requirements.txt`:**
```
flask>=2.3.0
pandas>=2.0.0
scikit-learn>=1.3.0
numpy>=1.24.0
```

All four are standard PyPI packages with no system-level prerequisites.
Installation typically takes under 60 seconds on a broadband connection.

### 5.3 Verify Data Files

Both CSV files must be in the **same directory as `app.py`**:

```bash
ls -lh
# -rw-r--r--  2.0M  User-data-10000.csv
# -rw-r--r--  5.0K  jobs_data.csv
# -rw-r--r--  2.5K  app.py
# -rw-r--r--  11K   ml_engine.py
# -rw-r--r--  512   requirements.txt
# drwxr-xr-x        templates/
```

The engine resolves paths using `os.path.dirname(__file__)` — the folder containing
`ml_engine.py`. If you move the CSVs elsewhere, update the paths in `ml_engine.py`:

```python
def train(self,
          users_path: str = None,
          jobs_path:  str = None) -> dict:
    users_path = users_path or os.path.join(BASE, "User-data-10000.csv")
    jobs_path  = jobs_path  or os.path.join(BASE, "jobs_data.csv")
```

### 5.4 Development Mode

```bash
python app.py
```

The server starts on `http://0.0.0.0:5000` with `debug=False` by default.
Debug mode is disabled because Flask's auto-reloader would retrain the model
on every code change.

To enable debug mode and accept the reload delay:
```python
# Bottom of app.py — change to:
app.run(debug=True, port=5000)
```

### 5.5 Production Deployment

Use **Gunicorn** with a single worker to avoid multiple independent model instances:

```bash
pip install gunicorn
gunicorn -w 1 -b 0.0.0.0:5000 app:app
```

**Why exactly 1 worker?**  
Each worker is a separate OS process. With multiple workers, each process would load and
train its own model independently — tripling memory use and startup time. A single worker
handles all requests sequentially; for a recommendation tool this is perfectly adequate.

### 5.6 Docker Deployment

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 5000
CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:5000", "app:app"]
```

```bash
docker build -t jobmatch-ai .
docker run -p 5000:5000 jobmatch-ai
```

---

## 6. Machine Learning Pipeline

All ML logic lives in `ml_engine.py` inside the `RecommenderEngine` class.
Training is triggered once on startup via a background daemon thread in `app.py`.

### 6.1 Startup Sequence

```
python app.py
    │
    ├── Flask app object created
    ├── threading.Thread(target=_train, daemon=True).start()
    │       (non-blocking — Flask is ready immediately)
    │
    └── Flask begins serving requests
             │
             └── _train() runs concurrently in background:
                     └── engine.train()
                              ├── Load CSVs
                              ├── Parse skills
                              ├── Fit TF-IDF
                              ├── Train Random Forest
                              ├── Evaluate model
                              ├── Vectorise jobs
                              └── engine.ready = True
```

The `/api/status` endpoint returns `{"ready": false}` during training and
`{"ready": true, ...}` once complete. The browser polls this endpoint every 1.2 seconds.

### 6.2 Step-by-Step Training Process

#### Step 1 — Load CSVs

```python
users_df = pd.read_csv("User-data-10000.csv")   # 10,000 rows
jobs_df  = pd.read_csv("jobs_data.csv")          # 11 rows
```

#### Step 2 — Parse Skill Lists

Skills are stored as stringified Python lists (e.g. `"['python', 'sql']"`).
The `_parse()` helper handles them safely with two fallback strategies:

```python
def _parse(raw) -> list[str]:
    if pd.isna(raw):
        return []
    try:
        out = ast.literal_eval(str(raw))   # strategy 1: proper Python eval
        if isinstance(out, list):
            return [s.strip().lower() for s in out]
    except Exception:
        pass
    # strategy 2: regex split on common delimiters
    return [s.strip().lower()
            for s in re.split(r"[,;\n|]", str(raw)) if s.strip()]
```

All skills are normalised to **lowercase** to ensure case-insensitive matching
throughout the pipeline.

#### Step 3 — Build Skill Documents

Each user profile is converted into a single text string for TF-IDF processing.
Hard skills are **repeated twice** to give them double the statistical weight:

```python
def _skills_to_doc(hard: list, soft: list) -> str:
    return " ".join(hard * 2 + soft)
    # Example: "python python sql sql excel excel leadership communication"
```

This deliberate weighting reflects that hard skills are more discriminative
for job-field classification than soft skills.

#### Step 4 — Collect Unique Skills

All unique skills from the training corpus are sorted into two lists for autocomplete:

```python
hard_all, soft_all = set(), set()
for h in df["_hard"]: hard_all.update(h)
for s in df["_soft"]: soft_all.update(s)
self.all_hard = sorted(hard_all)   # 4,474 unique hard skills
self.all_soft = sorted(soft_all)   # 214 unique soft skills
```

#### Step 5 — Fit TF-IDF Vectoriser

```python
self.tfidf = TfidfVectorizer(
    ngram_range=(1, 2),    # capture both unigrams AND bigrams
                           # e.g. "project" AND "project management" as features
    max_features=5000,     # keep the 5,000 most informative terms by TF-IDF score
    sublinear_tf=True,     # replace raw TF with log(1 + TF) to dampen
                           # high-frequency terms like "business" or "service"
    min_df=2,              # discard terms appearing in fewer than 2 documents
                           # (eliminates typos and one-off skills)
)
self.tfidf.fit(df["_doc"])   # learn vocabulary + IDF weights from all 10,000 docs
```

**What TF-IDF does:**
- **TF (Term Frequency):** How often a skill appears in *this* user's profile
- **IDF (Inverse Document Frequency):** How rare that skill is *across all profiles*
- **TF-IDF score:** TF × IDF — high for skills that are specific to a user and rare overall

A skill like "business" appears in nearly every profile, so its IDF is near zero and
it contributes very little. A skill like "xero" is rare and highly specific to accounting,
so it has a high IDF and strongly influences the accounting prediction.

#### Step 6 — Transform Training Data

```python
X = self.tfidf.transform(df["_doc"])
# Shape: (10000, 5000) — sparse CSR matrix
# Each row is a user; each column is one of the 5,000 TF-IDF features
```

The matrix is sparse (most values are zero) because most users only have 5-15 skills
out of 5,000 possible features. scikit-learn handles sparse matrices efficiently.

#### Step 7 — Encode Class Labels

```python
self.le = LabelEncoder()
y = self.le.fit_transform(df["candidate_field"])
# Maps 9 category strings to integers 0-8
# Example: "accounting" → 0, "administration & office support" → 1, ...
```

#### Step 8 — Stratified Train/Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,       # 20% held out = 2,000 test samples
    random_state=42,     # fixed seed for fully reproducible results
    stratify=y           # preserve class proportions in both splits
                         # ensures rare classes like "sport" appear in both sets
)
# X_train shape: (8,000, 5,000)
# X_test  shape: (2,000, 5,000)
```

**Why stratify?** Without stratification, the 119 sport & recreation samples might all
end up in the training set by chance, giving the model no test examples to evaluate on.
`stratify=y` guarantees ~95 training samples and ~24 test samples for sport.

#### Step 9 — Train Random Forest Classifier

```python
self.rf = RandomForestClassifier(
    n_estimators=200,         # build 200 independent decision trees
    max_depth=None,           # trees grow until all leaves contain pure classes
                              # or fewer than min_samples_split samples
    min_samples_split=4,      # a node must have 4+ samples to be split further
                              # prevents overfitting on tiny leaf nodes
    class_weight="balanced",  # automatically compensate for class imbalance
                              # rare classes (sport=119) get higher per-sample weight
                              # than common classes (telecom=2382)
    n_jobs=-1,                # use all available CPU cores (parallel tree building)
    random_state=42,          # reproducible ensemble
)
self.rf.fit(X_train, y_train)
```

**How Random Forest works:**
1. Each of the 200 trees is trained on a random subset of the training data (bootstrap sample)
2. At each node split, a random subset of features (typically sqrt(5000) ≈ 70) is considered
3. The best split among those 70 features is chosen using Gini impurity
4. Prediction is the **majority vote** across all 200 trees
5. Class probability is the **fraction of trees** voting for each class — this is what
   `predict_proba()` returns

**Why Random Forest over other models?**

| Alternative | Limitation |
|-------------|-----------|
| Logistic Regression | Assumes linear decision boundary; skill interactions are non-linear |
| SVM | Does not natively output calibrated probabilities |
| Naive Bayes | Assumes feature independence; skills are correlated |
| Neural Network | Requires much more data and tuning to outperform RF here |
| Decision Tree (single) | High variance; 77% accuracy drops to ~65% with one tree |

#### Step 10 — Evaluate on Held-Out Test Set

```python
y_pred = self.rf.predict(X_test)
acc    = accuracy_score(y_test, y_pred)       # 77.15%
report = classification_report(
    y_test, y_pred,
    target_names=self.le.classes_,
    output_dict=True,    # return dict instead of string for programmatic access
)
self.metrics = {
    "accuracy": round(float(acc) * 100, 2),
    "report":   report,
    "n_train":  int(X_train.shape[0]),
    "n_test":   int(X_test.shape[0]),
}
```

The full per-class report is stored and exposed at `/api/metrics`, rendered as a
table in the UI's Model Performance section.

#### Step 11 — Vectorise Job Catalogue

The same fitted TF-IDF is applied to all 11 job profiles so they live in the
**same 5,000-dimensional vector space** as user queries:

```python
self.job_vecs = self.tfidf.transform(self.jobs_df["_doc"])
# Shape: (11, 5000)
```

This must happen **after** fitting so the vocabulary is locked. The 11 job vectors
are stored in memory and reused on every recommendation request.

---

## 7. Scoring and Ranking Algorithm

When a user submits their skills, `engine.recommend()` runs three scoring stages.

### 7.1 Build User Vector

```python
user_doc = _skills_to_doc(hard_skills, soft_skills)
# Same 2x weighting as training: "python python sql sql leadership communication"

user_vec = self.tfidf.transform([user_doc])
# Shape: (1, 5000) — same vocabulary as training
```

### 7.2 Random Forest Probability Score

```python
proba = self.rf.predict_proba(user_vec)[0]
# Array of length 9 (one float per training class, sums to 1.0)
# e.g. [0.02, 0.15, 0.01, 0.08, 0.03, 0.04, 0.12, 0.03, 0.52]
#       acc   adm   adv   hlt   mkt   ret   sal   spt   tel

class_prob = {
    self.le.classes_[i]: float(proba[i])
    for i in range(len(self.le.classes_))
}
# {"accounting": 0.02, "telecomunication": 0.52, ...}
```

Jobs not in training data (`it jobs`, `recruitment`) map to a probability of `0.0`
because they have no corresponding class in `self.le.classes_`.

### 7.3 Cosine Similarity Score

```python
cos_scores = cosine_similarity(user_vec, self.job_vecs).flatten()
# Shape: (11,) — one score per job catalogue entry
```

Cosine similarity is the dot product of two unit-normalised vectors:

```
cosine(A, B) = (A · B) / (|A| × |B|)
```

It measures the angle between vectors, not their magnitude. Two profiles with identical
skill proportions score `1.0` regardless of profile length. Two profiles with no shared
vocabulary score `0.0`.

Unlike RF probability, cosine similarity applies to **all 11** jobs, covering the two
categories absent from training.

### 7.4 Blended Score

```python
blended = 0.60 * rf_prob + 0.40 * cos_score
```

| Component | Weight | Role |
|-----------|--------|------|
| RF probability | 60% | Learned contextual patterns from 10,000 real profiles |
| Cosine similarity | 40% | Direct keyword match; covers training-absent categories |

Both inputs are on `[0, 1]` before multiplication by 100 for display.

### 7.5 Skill Overlap Calculation

This is independent of ML — pure set arithmetic on the normalised skill strings:

```python
user_set = set(hard_skills + soft_skills)

for each job:
    job_set    = set(job_hard + job_soft)
    matched    = sorted(user_set & job_set)        # intersection
    missing    = sorted(job_set - user_set)        # what user lacks
    match_pct  = round(len(matched) / len(job_set) * 100)
```

**Example:**
```
User skills:  {"python", "sql", "leadership"}
Job skills:   {"python", "java", "sql", "testing", "leadership", "management"}

matched:   ["leadership", "python", "sql"]   ← 3 of 6 = 50% match
missing:   ["java", "management", "testing"]
match_pct: 50
```

### 7.6 Final Output Per Job

```json
{
  "job_id":         5,
  "category":       "it jobs",
  "hard_skills":    ["java", "python", "sql", "javascript", ...],
  "soft_skills":    ["management", "leadership", ...],
  "matched_skills": ["python", "sql", "leadership"],
  "missing_skills": ["java", "javascript", "testing", ...],
  "match_pct":      16,
  "rf_score":       0.0,
  "cos_score":      22.4,
  "score":          8.96
}
```

All 11 jobs are computed, then sorted descending by `score`, and the top N are returned.

---

## 8. Flask Application and API Reference

### 8.1 Application Bootstrap

```python
# app.py
app = Flask(__name__)

def _train():
    try:
        info = engine.train()
        print(f"Model ready — accuracy {info['accuracy']}%")
    except Exception as e:
        print(f"Training failed: {e}")

threading.Thread(target=_train, daemon=True).start()
# Flask begins serving before training completes
```

`daemon=True` ensures the training thread is automatically killed when the main
Python process exits (e.g. Ctrl+C), preventing zombie threads.

### 8.2 Complete API Reference

---

#### `GET /`
Returns the main HTML page.

**Response:** `200 text/html` — full single-page application.

---

#### `GET /api/status`
Poll training state. Call repeatedly until `ready` is `true`.

**While training:**
```json
{"ready": false, "message": "Training in progress…"}
```

**When ready:**
```json
{
  "ready":       true,
  "users":       10000,
  "jobs":        11,
  "hard_skills": 4474,
  "soft_skills": 214,
  "accuracy":    77.15,
  "categories":  ["accounting", "administration & office support", "..."]
}
```

---

#### `POST /api/recommend`
Returns ranked job recommendations.

**Request `Content-Type: application/json`:**
```json
{
  "hard_skills": "python, sql, data analysis, excel",
  "soft_skills": "leadership, communication",
  "experience":  "mid",
  "top_n":       8
}
```

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `hard_skills` | string | Yes* | Comma-separated |
| `soft_skills` | string | Yes* | Comma-separated |
| `experience` | string | No | `"entry"`, `"mid"`, `"senior"`, or `""` |
| `top_n` | int | No | Default 11, max 11 |

*At least one skill field must be non-empty.

**Response `200 OK`:**
```json
{
  "results": [
    {
      "job_id":         5,
      "category":       "it jobs",
      "hard_skills":    ["java", "python", "sql", "javascript", "aws", "..."],
      "soft_skills":    ["management", "leadership", "problem solving", "..."],
      "matched_skills": ["python", "sql"],
      "missing_skills": ["java", "javascript", "testing", "linux", "aws"],
      "match_pct":      10,
      "rf_score":       0.0,
      "cos_score":      18.4,
      "score":          7.4
    }
  ],
  "total": 8
}
```

**Error responses:**

| Code | Condition | Body |
|------|-----------|------|
| `400` | Both skill fields empty | `{"error": "Enter at least one skill"}` |
| `503` | Model still training | `{"error": "Model still training, please wait…"}` |

---

#### `GET /api/autocomplete/hard?q=<query>`
Returns up to 14 hard skill suggestions.

**Example:** `/api/autocomplete/hard?q=py`
```json
["python", "physical therapy", "payroll processing", "pyomo"]
```
Returns `[]` if `q` is fewer than 2 characters.

---

#### `GET /api/autocomplete/soft?q=<query>`
Returns up to 14 soft skill suggestions.

**Example:** `/api/autocomplete/soft?q=lead`
```json
["leadership", "lead generation", "leading teams"]
```

---

#### `GET /api/metrics`
Returns the full model evaluation report.

```json
{
  "accuracy":    77.15,
  "total_users": 10000,
  "total_hard":  4474,
  "total_soft":  214,
  "categories":  ["accounting", "..."],
  "rf_report": {
    "accounting": {
      "precision": 0.882,
      "recall":    0.853,
      "f1-score":  0.867,
      "support":   295
    },
    "...": {}
  }
}
```

---

## 9. Frontend Architecture

The entire frontend is a **single HTML file** (`templates/index.html`, ~60 KB).
No JavaScript framework, no build tool, no npm. Everything is vanilla.

### 9.1 Technology Choices

| Concern | Solution | Rationale |
|---------|----------|-----------|
| Fonts | Google Fonts CDN | Plus Jakarta Sans + JetBrains Mono |
| Styling | Vanilla CSS with custom properties | No framework dependency |
| JavaScript | Vanilla ES2022 | async/await, Set, template literals |
| SVG charts | Pure inline SVG | Score rings via `stroke-dasharray` |
| HTTP calls | `fetch()` API | Native, no library needed |
| State | JavaScript `Set` + `let` | Minimal and explicit |

### 9.2 Application State Variables

```javascript
const hardSet = new Set();     // currently selected hard skills
const softSet = new Set();     // currently selected soft skills

let allResults     = [];       // raw results from server (original sort order)
let displayResults = [];       // currently rendered order (after user sorting)
let curTab         = 'hard';   // active skill tab: 'hard' or 'soft'
let expLevel       = '';       // '' | 'entry' | 'mid' | 'senior'
```

`allResults` and `displayResults` are kept deliberately separate.
`sortBy()` modifies only `displayResults`. `openModal(idx)` always reads from
`displayResults` so clicking card #3 always opens the job ranked #3 on screen —
regardless of whether the user has re-sorted the list.

### 9.3 Startup Polling Loop

```javascript
async function pollStatus() {
  advanceStep();                           // start loading animation
  const interval = setInterval(advanceStep, 3500);  // step every 3.5s
  while (true) {
    await sleep(1200);                     // check server every 1.2s
    try {
      const r = await fetch('/api/status');
      const d = await r.json();
      if (d.ready) {
        clearInterval(interval);
        onReady(d);                        // unlock UI, populate KPIs
        setTimeout(dismissOverlay, 1400); // fade out loading screen
        return;
      }
    } catch(e) {}                          // silently retry on failure
  }
}
```

The overlay shows a 5-step progress list. Steps are visually advanced by a timer
(not tied to real backend events) to give visual feedback during the ~15s wait.

### 9.4 Skill Tab System

```javascript
function switchTab(type) {
  curTab = type;
  // Toggle pane visibility
  document.getElementById('pane-hard').classList.toggle('active', type === 'hard');
  document.getElementById('pane-soft').classList.toggle('active', type === 'soft');
  // Toggle tab button colours
  document.getElementById('tab-hard').className =
    'skill-tab' + (type === 'hard' ? ' active-hard' : '');
  document.getElementById('tab-soft').className =
    'skill-tab' + (type === 'soft' ? ' active-soft' : '');
}
```

When `addTag()` is called from any source (autocomplete, quick chips, Enter key),
it calls `switchTab(type)` at the end so the user always sees the tag they just added.

### 9.5 Autocomplete System

```javascript
function makeAC(inputId, dropId, apiPath, type) {
  // Fires on every keystroke (de-facto debounced by network round-trip)
  inp.addEventListener('input', async () => {
    if (q.length < 2) { drop.classList.remove('open'); return; }
    items = await fetch(`${apiPath}?q=...`).json();
    // Render matched portion in bold using RegExp replace
    // Escape special regex chars to prevent injection
    const safe = q.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    drop.innerHTML = items.map(s =>
      s.replace(new RegExp(safe, 'gi'), m => `<b>${m}</b>`)
    );
    drop.classList.add('open');
  });
  // Keyboard: Enter commits, arrows navigate, Escape dismisses
  // Mouse: clicking a row also commits via window[`pickAC_${inputId}`]
}
```

Each call to `makeAC()` creates an independent closure per input, so the two
inputs (hard and soft) never share state.

### 9.6 SVG Score Donut Ring

```javascript
function donut(score) {
  const R = 24;                            // radius in px
  const C = 2 * Math.PI * R;              // full circumference = 150.8px
  const offset = C - (score / 100) * C;  // unfilled gap length

  // stroke-dasharray="150.8 150.8"  → one full dash = full circle
  // stroke-dashoffset="105.6"       → shift start point clockwise
  // After rotating SVG -90deg, filled arc starts at 12 o'clock
}
```

Score-to-colour mapping:
- **Teal** (`#00c2a8`): score >= 60 — strong match
- **Gold** (`#f5a623`): score >= 35 — moderate match  
- **Muted** (`#8898bb`): score < 35 — weak match

---

## 10. Responsive Design System

### 10.1 Breakpoints

```
xs: < 480px    Small phones — base styles, everything stacked
sm: 480px+     Large phones — KPI strip in header, corner toast
md: 640px+     Tablets portrait — 4-column score boxes, 2-col modal grid
lg: 900px+     Tablets landscape / laptops — two-column sidebar + results layout
xl: 1200px+    Desktop — wider sidebar (380px), increased page padding
```

### 10.2 Two-Layout Strategy

**Mobile (< 900px) — Bottom Sheet Pattern:**
```
┌─────────────────────┐
│ Header              │
├─────────────────────┤
│ Hero                │
├─────────────────────┤
│ Results (full width)│
│                     │
│                     │
└─────────────────────┘
         [⚙️ FAB]   ← tapping opens sidebar as bottom sheet
```

**Desktop (900px+) — Sidebar + Results:**
```
┌──────────────────────────────────────────┐
│ Header                                   │
├─────────────┬────────────────────────────┤
│ Hero (full width)                        │
├─────────────┬────────────────────────────┤
│             │ Results                    │
│  Sidebar    │                            │
│ (sticky)    │  [Card 1]                  │
│             │  [Card 2]                  │
│             │  [Card 3]                  │
└─────────────┴────────────────────────────┘
```

### 10.3 DOM-Based Layout Switching

Because the sidebar needs to be a bottom-sheet on mobile and a sticky column on
desktop, it needs different DOM parents in each layout. This is handled with JavaScript:

```javascript
function positionSidebar() {
  const slot = document.querySelector('.sidebar-slot'); // placeholder in grid
  const sb   = document.getElementById('panel');

  if (window.innerWidth >= 900) {
    slot.style.display = 'block';
    slot.appendChild(sb);   // move into CSS grid
    // CSS applies: position:sticky; top:70px; border-radius:16px
  } else {
    if (sb.parentElement !== document.body) {
      document.body.appendChild(sb); // move back to body for bottom-sheet
    }
    slot.style.display = 'none';
    // CSS applies: position:fixed; bottom:0; transform:translateY(100%)
  }
}
positionSidebar();
window.addEventListener('resize', positionSidebar);
```

### 10.4 iOS Compatibility

| Issue | Fix |
|-------|-----|
| Input zoom on focus | `font-size: 16px` on all `<input>` elements |
| iPhone notch overlap | `padding-bottom: env(safe-area-inset-bottom)` |
| Edge-to-edge display | `viewport-fit=cover` in `<meta name="viewport">` |
| Momentum scrolling | `-webkit-overflow-scrolling: touch` on scrollable containers |
| Tap flash | `-webkit-tap-highlight-color: transparent` on all interactive elements |
| Overscroll bounce | `overscroll-behavior: contain` on drawer and modal |

### 10.5 Touch Targets

All interactive elements meet the 44px minimum touch target size:

```css
:root { --touch: 44px; }

.seg-opt    { min-height: var(--touch); }
.ac-row     { min-height: var(--touch); display: flex; align-items: center; }
.panel-close{ width: var(--touch); height: var(--touch); }
.m-close    { width: var(--touch); height: var(--touch); }
.search-btn { min-height: var(--touch); }
```

### 10.6 Colour System

```
Surface hierarchy:
  --bg   #0a0c14  ← page background
  --s1   #0e1120  ← cards, panels
  --s2   #13172a  ← inputs, chips, nested surfaces
  --s3   #181d33  ← progress tracks, hover states
  --s4   #1e2540  ← subtle dividers

Brand palette:
  --gold  #f5a623  ← hard skills, RF probability, CTA gradient, strong scores
  --teal  #00c2a8  ← soft skills, cosine similarity, success, live indicators

Semantic:
  --green  #22c55e  ← matched skills
  --red    #ef4444  ← errors, remove actions
  --rose   #f43f5e  ← FAB badge, critical counts
  --sub    #8898bb  ← secondary text, labels
  --dim    #2a3550  ← placeholder text, very low emphasis
```

---

## 11. Complete Request Response Workflow

### 11.1 Application Startup

```
1. User opens http://localhost:5000

2. Browser → GET /
   Server  → 200 HTML (index.html, ~60 KB)

3. Browser renders HTML; JavaScript starts executing

4. pollStatus() begins:
   Browser → GET /api/status (every 1.2 seconds)
   Server  → {"ready": false}  (while training)

5. After ~15 seconds:
   Server  → {"ready": true, "accuracy": 77.15, "users": 10000, ...}

6. onReady(d) fires:
   - Header KPIs updated (10,000 users, 11 jobs, 4,474 hard, 214 soft)
   - Status dot turns teal (live)
   - Accuracy badge appears (77.15%)
   - Model stats card revealed in sidebar
   - CTA button enabled
   - Quick-add chips rendered
   - loadMetrics() fetches /api/metrics → renders performance table

7. setTimeout(dismissOverlay, 1400) → loading screen fades and is removed from DOM
```

---

### 11.2 Adding Skills

```
User types "py" in Hard Skills input

1. Browser → GET /api/autocomplete/hard?q=py
   Server  → ["python", "physical therapy", "payroll processing", ...]

2. Dropdown renders with "py" portion bolded in each suggestion

3. User presses Enter (or clicks "python")

4. addTag("python", "hard"):
   - hardSet.add("python")
   - New <span class="stag hard"> appended to #hard-tags
   - switchTab("hard") (no-op if already on hard tab)
   - updateBadge():
       tab-hard count: 1
       FAB badge: 1  (becomes visible)
       "python" quick chip: .added class applied (greyed out)

5. User taps FAB on mobile → openDrawer() → sidebar slides up from bottom

6. User adds 3 more hard skills, 2 soft skills

7. FAB badge shows: 5
   Hard tab badge: 3
   Soft tab badge: 2
```

---

### 11.3 Running a Recommendation

```
User clicks "Find Matching Jobs"

1. doSearch() validates: hardSet.size + softSet.size > 0  ✓

2. CTA button disabled, text → "Analysing…" with spinner

3. If mobile: closeDrawer() → sidebar slides down

4. Browser → POST /api/recommend
   Body: {
     "hard_skills": "python, sql, data analysis",
     "soft_skills": "leadership, communication",
     "experience":  "mid",
     "top_n":       8
   }

5. server: engine.recommend(...)
   a. Build user doc: "python python sql sql data analysis data analysis leadership communication"
   b. TF-IDF transform → user_vec (1×5000 sparse)
   c. RF predict_proba → [0.02, 0.15, 0.01, 0.08, 0.03, 0.04, 0.12, 0.03, 0.52]
   d. cosine_similarity(user_vec, job_vecs) → [0.06, 0.07, 0.03, 0.04, 0.18, ...]
   e. For each of 11 jobs:
        blended = 0.60 × rf_prob + 0.40 × cos_score
        matched = user_skills ∩ job_skills
        missing = job_skills − user_skills
        match_pct = |matched| / |job_skills| × 100
   f. Sort by blended score (descending)
   g. Return top 8

6. Browser: renderCards(displayResults)
   - #rpill updated: "8 categories"
   - Sort bar appears
   - 8 job cards rendered with animation-delay staggering (0, 40, 80, 120, ... ms)
   - Each card shows: donut ring, RF/Cosine/Blended chips, matched skills,
     missing skills, skill columns, match progress bar

7. Toast: "8 job categories ranked ✓"

8. CTA re-enabled

9. If mobile: page scrolls to #resbox
```

---

### 11.4 Sorting Results

```
User clicks "↑ Skill Match" sort button

sortBy("match", buttonElement):
  - All sort buttons have .on removed
  - "Skill Match" button gets .on class
  - displayResults = [...allResults].sort((a,b) => b.match_pct - a.match_pct)
  - renderCards(displayResults)
  - Cards re-render with new order (same animation)
  - allResults unchanged (original order preserved for future re-sorts)
```

---

### 11.5 Viewing Job Details Modal

```
User clicks job card #3 (currently in rank #3 after sorting)

openModal(2):  ← idx=2 (zero-indexed)
  j = displayResults[2]    ← correct: reads display order, not original order

  Modal populates:
  - Title: "accounting"
  - Badges: category badge + Job ID badge
  - Score boxes (2×2 grid):
      RF Probability: 45.2% (gold)
      Cosine Sim:     12.3% (teal)
      Blended Score:  32.0% (gold, score 32-59)
      Skill Match:    23%   (gold)
  - Matched skills section: green chips for each matched skill
  - Skills to develop: red/pink chips for missing skills
  - Full requirements grid:
      Hard Skills column: all 20 required hard skills
        → user's matched skills highlighted in teal
      Soft Skills column: all 10 required soft skills
        → user's matched skills highlighted in teal

  document.body.style.overflow = "hidden"  ← prevent background scroll

Close modal (3 ways):
  - Click backdrop outside modal → maybeClose(event)
  - Press Escape key → keydown listener
  - Swipe down on modal (mobile) → touchstart/touchend listener
  - Click ✕ button → closeModal()

  document.body.style.overflow = ""   ← restore scroll
```

---

## 12. Model Performance

### 12.1 Overall Metrics

| Metric | Value |
|--------|-------|
| **Overall test accuracy** | **77.15%** |
| Training samples | 8,000 (80% of 10,000) |
| Test samples | 2,000 (20% of 10,000) |
| Number of predicted classes | 9 |
| Majority class baseline | ~23.8% (always predict telecomunication) |
| **Improvement over baseline** | **3.2× better** |

### 12.2 Per-Class Results

| Category | Precision | Recall | F1-Score | Test Samples |
|----------|-----------|--------|----------|-------------|
| accounting | 88.2% | 85.3% | **86.7%** | 295 |
| administration & office support | 67.1% | 61.6% | **64.3%** | 305 |
| advertising, arts & media | 50.0% | 10.7% | **18.2%** | 28 |
| healthcare & medical | 81.2% | 87.2% | **84.1%** | 366 |
| marketing | 62.0% | 72.1% | **66.7%** | 122 |
| retail & consumer products | 65.8% | 56.3% | **60.5%** | 135 |
| sales | 74.5% | 67.3% | **70.7%** | 248 |
| sport & recreation | 47.6% | 66.7% | **55.8%** | 24 |
| telecomunication | 86.4% | 88.5% | **87.4%** | 477 |

### 12.3 Interpreting the Scores

**High-performing classes (F1 > 80%):**
`accounting`, `healthcare & medical`, `telecomunication` — these fields have large,
distinctive skill vocabularies with minimal overlap. "Xero", "erp", "reconciliation"
appear almost exclusively in accounting profiles; "nursing", "rehabilitation" in healthcare.

**Lower-performing classes (F1 < 65%):**
`advertising, arts & media` (28 test samples) and `sport & recreation` (24 test samples)
suffer from insufficient training data. The model struggles to learn stable boundaries
when only ~120 training examples exist for a class.

**Classes with zero RF probability:**
`it jobs` and `recruitment consultancy jobs` are not in the training data.
They still appear in recommendations via cosine similarity when the user's skills
match their keyword requirements (e.g. "java", "python", "testing" for IT).

---

## 13. Configuration and Customisation

### 13.1 Change RF Hyperparameters

In `ml_engine.py`, `_train_rf()` method:

```python
self.rf = RandomForestClassifier(
    n_estimators=200,      # increase to 500 for ~1% more accuracy (slower)
    max_depth=None,        # set to 20 to reduce overfitting
    min_samples_split=4,   # increase to 8 for simpler trees
    class_weight="balanced",
    n_jobs=-1,
    random_state=42,
)
```

### 13.2 Change TF-IDF Parameters

```python
self.tfidf = TfidfVectorizer(
    ngram_range=(1, 2),   # (1,1) unigrams only; (1,3) adds trigrams
    max_features=5000,    # increase to 10000 for more precision (more memory)
    sublinear_tf=True,    # set False for raw term frequency
    min_df=2,             # increase to 5 to ignore less common skills
)
```

### 13.3 Change the Score Blend

In `ml_engine.py`, `recommend()` method:

```python
# Current: 60% RF, 40% cosine
blended = 0.60 * rf_prob + 0.40 * cos_score

# More weight on exact keyword matching:
blended = 0.40 * rf_prob + 0.60 * cos_score

# RF only (disable cosine):
blended = 1.00 * rf_prob + 0.00 * cos_score
```

### 13.4 Add New Job Categories

1. Add a row to `jobs_data.csv`
2. Add representative user profiles to `User-data-10000.csv` with the new `candidate_field`
3. Restart the app — model retrains automatically

Minimum recommended: 100+ new user profiles for a reliable F1-score.

### 13.5 Swap the Training Algorithm

Replace Random Forest with any scikit-learn classifier that supports `predict_proba`:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# LogisticRegression — much faster training, slightly lower accuracy
self.rf = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    C=1.0,
    random_state=42,
)

# SVM with probability (much slower)
self.rf = SVC(
    kernel="linear",
    probability=True,
    class_weight="balanced",
    random_state=42,
)
```

### 13.6 Customise Quick-Add Chips

Edit the arrays at the top of the `<script>` section in `index.html`:

```javascript
const HARD_QUICK = [
  'python', 'sql', 'excel', 'javascript', 'java',
  'data analysis', 'financial modeling', 'accounting',
  'project management', 'marketing'
];

const SOFT_QUICK = [
  'leadership', 'communication', 'teamwork', 'problem solving',
  'time management', 'customer service', 'management', 'planning'
];
```

### 13.7 Change the Application Port

```python
# Bottom of app.py
app.run(debug=False, port=8080)   # change 5000 to any available port
```

---

## 14. Troubleshooting

### Problem: "Model still training" persists after 60 seconds

**Cause:** Training exception silently failed in the background thread.

**Fix:** Check the terminal output. `app.py` prints the error:
```
✗ Training failed: <error message>
```

Common causes:
- Missing CSV files (verify both CSVs are in the project root)
- Insufficient RAM (model requires ~500 MB peak during training)
- Incompatible package versions (run `pip install -r requirements.txt` again)

---

### Problem: `FileNotFoundError` on startup

```
FileNotFoundError: [Errno 2] No such file or directory:
  '/path/to/jobmatch2/User-data-10000.csv'
```

**Fix:** Ensure both CSV files are in the **same directory** as `app.py` and `ml_engine.py`.

---

### Problem: Port already in use

```
OSError: [Errno 98] Address already in use: ('0.0.0.0', 5000)
```

**Fix:**
```bash
# Find the process using port 5000
lsof -ti :5000 | xargs kill -9    # macOS/Linux: kill it
# or change port in app.py to 5001
```

---

### Problem: Low recommendation scores (all scores under 10%)

**Cause:** Very generic skills like "sales", "management", "service" have
near-zero IDF because they appear in almost every training document.

**Fix:** Add more specific skills. Instead of just "sales", add:
"account management", "business development", "crm", "lead generation", "closing".
The model performs best with 8+ domain-specific skills.

---

### Problem: Autocomplete returns no suggestions

**Cause:** Query is fewer than 2 characters, or the skill is not in the training vocabulary.

**Fix:** Type at least 2 characters. Suggested skills come only from `all_hard` and
`all_soft` which are built from the training data. Any skill can still be manually
typed and added — it just won't appear in suggestions.

---

### Problem: Modal opens wrong job after sorting

This was a bug in an earlier version (fixed). `openModal(idx)` now reads from
`displayResults` (the currently rendered order) not `allResults` (original order).
If you see this issue, ensure you are using the latest version of `index.html`.

---

### Problem: Page looks broken on mobile

**Cause:** iOS Safari zoom triggered by inputs with `font-size < 16px`.

**Fix:** All `<input>` elements in `index.html` use `font-size: 16px` which prevents
the automatic zoom. If you modify the CSS and reduce input font size, Safari will
zoom in on focus — restore `font-size: 16px` to fix.

---

## 15. Extending the System

### Add Model Persistence (avoid retraining on restart)

```python
import joblib

# After training — save to disk:
joblib.dump({
    'rf':    self.rf,
    'tfidf': self.tfidf,
    'le':    self.le,
}, 'model.pkl')

# On startup — check for saved model:
MODEL_PATH = os.path.join(BASE, 'model.pkl')
if os.path.exists(MODEL_PATH):
    saved = joblib.load(MODEL_PATH)
    self.rf, self.tfidf, self.le = saved['rf'], saved['tfidf'], saved['le']
    users_df = pd.read_csv(users_path)
    self._preprocess_users(users_df)
    self._preprocess_jobs()
    self._vectorise_jobs()
    self.ready = True
else:
    self.train()  # train from scratch and save
    joblib.dump({'rf': self.rf, 'tfidf': self.tfidf, 'le': self.le}, MODEL_PATH)
```

Benefit: restart time drops from ~15 seconds to ~1 second.

---

### Add a Second Classifier (ensemble)

```python
from sklearn.ensemble import GradientBoostingClassifier

self.gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
self.gb.fit(X_train, y_train)

# In recommend():
rf_prob  = class_prob_rf.get(major, 0.0)
gb_prob  = class_prob_gb.get(major, 0.0)
cos_sc   = float(cos_scores[idx])
blended  = 0.35 * rf_prob + 0.25 * gb_prob + 0.40 * cos_sc
```

---

### Add User Profile Persistence

```javascript
// Save to localStorage when skills change (in addTag / removeTag)
function saveProfile() {
  localStorage.setItem('hardSkills', JSON.stringify([...hardSet]));
  localStorage.setItem('softSkills', JSON.stringify([...softSet]));
}

// Restore on page load
function loadProfile() {
  const hard = JSON.parse(localStorage.getItem('hardSkills') || '[]');
  const soft = JSON.parse(localStorage.getItem('softSkills') || '[]');
  hard.forEach(s => addTag(s, 'hard'));
  soft.forEach(s => addTag(s, 'soft'));
}
```

---

### Deploy to Cloud

**Render.com (free tier):**
```yaml
# render.yaml
services:
  - type: web
    name: jobmatch-ai
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: gunicorn -w 1 -b 0.0.0.0:$PORT app:app
```

**Docker:**
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:5000", "app:app"]
```

```bash
docker build -t jobmatch-ai .
docker run -p 5000:5000 jobmatch-ai
```

---

## License

This project is provided for educational and personal use.

**Dataset:** [batuhanmtl/job-skill-set](https://huggingface.co/datasets/batuhanmtl/job-skill-set)
by Batuhan Mutlu (2024). Skill sets extracted using RecAI APIs.