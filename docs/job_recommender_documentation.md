# JobMatch AI - Comprehensive Technical Documentation

## 1. Executive Summary
JobMatch AI is an intelligent, machine-learning-powered job recommendation system. It evaluates a candidate's profile—consisting of specific skills, desired experience level, and preferred job categories—and matches it against a dataset of job listings. The system utilizes Natural Language Processing (NLP) techniques, namely Term Frequency-Inverse Document Frequency (TF-IDF) combined with Cosine Similarity, to compute relevance scores. An experience-based heuristic modifier further refines the ranking.

The application uses an ephemeral, memory-based data ingestion pipeline. Instead of storing complex databases locally, the application ingests `.parquet` dataset files dynamically (e.g., fetched directly from the HuggingFace CDN by the client browser).

## 2. System Architecture

The application is built on a standard Client-Server architecture with an embedded Machine Learning singleton.

- **Frontend Interface (`templates/index.html`)**: The user-facing component built with HTML, CSS, and JavaScript. It provides an interactive UI with animated cards, a dark theme, real-time skill autocomplete, and filtering options. Data from HuggingFace is fetched here and dispatched to the backend.
- **RESTful API Backend (`app.py`)**: A Flask-based web server routing API requests, parsing JSON/Parquet payloads, and acting as the mediator between the UI and the ML core.
- **Machine Learning Engine (`ml_engine.py`)**: A discrete, stateful Python singleton that maintains the pre-computed sparse matrices, vocabulary arrays, and performs all vector transformations and cosine similarity calculations.

## 3. Machine Learning Engine (`ml_engine.py`) Detailed Breakdown

The `MLEngine` class encapsulates the sequence of data processing and recommendation. It acts as a singleton (`engine = MLEngine()`) initialized when Flask starts, maintaining the vocabulary and TF-IDF matrix in memory.

### 3.1 Data Ingestion (`ingest_parquet` & `_load`)
1. The engine receives raw bytes (representing a `.parquet` file).
2. It uses `io.BytesIO` to stream these bytes into `pandas.read_parquet()`, instantiating a DataFrame without writing to the disk.
3. It triggers preprocessing and fitting functions.

### 3.2 NLP Preprocessing (`_preprocess`)
The dataset requires heavy cleaning, specifically parsing irregular skill array formats.
1. **Skill Parsing:** The system encounters skills as stringified Python lists (e.g., `"['Python', 'SQL']"`) or delimiter-separated strings (e.g., `"Python; SQL|AWS"`). The `parse_skills` nested function utilizes `ast.literal_eval` safely evaluate strings into native Python lists. Fallbacks use regular expressions (`r"[,;\n|]"`) to split raw strings. All skills are aggressively stripped of whitespace and converted to lowercase.
2. **Document Synthesis:** To enable TF-IDF to find semantic overlaps, the engine constructs a combined string field called `_doc`. This aggregates the strings of `job_title`, `category`, and the aggregated parsed skills (`_skills_str`). This `_doc` becomes the corpus on which the vectorizer operates.
3. **Vocabulary Extraction:** The engine iterates over all parsed skills to generate a sorted deduplicated universe of skills (`all_skills`) and categories (`categories`), empowering the frontend's autocomplete and dropdown UI.

### 3.3 Text Vectorization (`_fit`)
The engine uses `sklearn.feature_extraction.text.TfidfVectorizer` to map the text documents to a vector space.
- **N-Grams:** Set to `ngram_range=(1, 2)`, meaning it evaluates individual words ("Unigrams") and two-word pairings ("Bigrams"). This catches context like "data science" or "machine learning".
- **Vocabulary Size:** Hard-capped at `max_features=10_000` to manage memory efficiency and mitigate the curse of dimensionality.
- **Sublinear TF:** Enabled (`sublinear_tf=True`), applying logarithmic scaling to term frequencies (e.g., 20 occurrences of a word doesn't carry 20 times the weight of 1 occurrence).

Once configured, it runs `fit_transform` on the entire corpus's `_doc` items, generating the base `matrix`.

### 3.4 Recommendation Algorithm (`recommend`)

The `recommend` function dynamically calculates matches for a specific candidate query:

#### Phase A: Query Construction & Vectorization
1. The user's input skills are lowered, stripped, and joined into a single space-separated string.
2. If `categories` are specified, the category strings are appended to the query *multiple times* (`" ".join(categories) * 3`). This artificially inflates the TF-IDF weight of the category terms, forcing the cosine similarity to strongly favor jobs falling under those exact categories.
3. The query is vectorized (`q_vec = self.vectorizer.transform([query])`).

#### Phase B: Cosine Similarity Scoring
1. The `cosine_similarity` function measures the angular distance between the query vector `q_vec` and the pre-computed corpus `self.matrix`.
2. This creates a flat array of scores ranging from 0.0 (no overlap) to 1.0 (exact match) for all jobs in the dataset.

#### Phase C: Experience Modifier Heuristics
A static scoring boost is applied using keyword inspection on the job's title and description.
- **Senior Keywords:** `{"senior", "sr", "director", "lead", "head", "chief", "vp", "manager", "principal", "staff"}`
- **Junior/Entry Keywords:** `{"junior", "jr", "entry", "associate", "intern", "graduate"}`

**Modifier Logic:**
- If user requests **`entry`**: +0.12 boost if junior keywords are found. -0.08 penalty if senior keywords are found.
- If user requests **`mid`**: +0.04 boost if senior/lead keywords are found. -0.04 penalty if executive (director/vp) keywords are found.
- If user requests **`senior`**: +0.12 boost if senior keywords are found.

#### Phase D: Targeted Category Boosting
Exact matches between the user's requested `categories` and a job's `category` column apply a fixed `+0.06` deterministic boost.

#### Phase E: Ranking & Data Transformation
1. Jobs with a final score below `0.005` are filtered out to remove extreme outliers.
2. The remaining jobs are sorted descending by score, capped at `top_n` (default 10).
3. For the returned jobs, Python `set` algebra executes to determine:
    - `"matched_skills"`: Set intersection `(User Skills) & (Job Skills)`.
    - `"missing_skills"`: Set difference `(Job Skills) - (User Skills)`.
    - `"match_pct"`: Ratio of matched skills out of the total required skills.

## 4. RESTful API Server (`app.py`) Detail

The Flask framework provides simple, decoupled REST endpoints:

- **`GET /`**: Renders `templates/index.html`.
- **`POST /api/ingest`**: Primary setup endpoint. Receives raw binary payload (`request.get_data()`), forwards bytes to `engine.ingest_parquet()`.
- **`POST /api/recommend`**: Rejects requests if the engine is not initialized (HTTP 503). Parses JSON body containing `"skills"`, `"experience"`, `"categories"`, and `"top_n"`.
- **`GET /api/autocomplete?q=...`**: Takes a string query `q`. If `len(q) >= 2`, returns up to 14 skills from the global vocabulary that contain the substring.
- **`GET /api/job/<job_id>`**: Returns a single dictionary containing specific job details.
- **`GET /api/stats`**: Serves a system health payload (total elements loaded in memory).

## 5. Development & Deployment

### Dependencies
Defined in `requirements.txt`:
- `flask (>=2.3.0)`: Web server interface.
- `pandas (>=2.0.0)`: In-memory dataset management operations.
- `scikit-learn (>=1.3.0)`: TF-IDF vectorization and cosine mathematics.
- `numpy (>=1.24.0)`: Fast array/matrix implementations utilized by `sklearn` and `pandas`.
- `pyarrow (>=12.0.0)`: Codec allowing `pandas` to read Parquet binaries.

### Quick Start
```bash
# 1. Set up a virtual environment to isolate dependencies
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# 2. Install requirements
pip install -r requirements.txt

# 3. Start the Flask application
python app.py
```
The application runs locally on `http://127.0.0.1:5000`.

## 6. Future Expansion Options
1. **Model Persistence**: Caching the `self.matrix` and `self.vectorizer` to disk using `joblib` so restarting the Flask app does not require a fresh Parquet ingest sequence.
2. **Dense Vector Embeddings**: Migrating from sparse vectorization (`TfidfVectorizer`) to dense neural embeddings (e.g. `sentence-transformers` models). Given skill semantic overlaps (e.g., "Javascript" and "NodeJS"), a dense vector space would understand relationships where exact string matching via TF-IDF fails.
3. **Database Migration**: Currently bounded by hardware RAM. Using Vector Databases (`Pinecone`, `Milvus`, or `pgvector`) to store vectors out-of-core would allow scanning datasets scaling upwards of millions of jobs efficiently without RAM limits.

## 7. Complete Workflow: End-to-End Execution

The step-by-step lifecycle of the application—from startup to a user receiving a matched job—operates as follows:

### Step 1: Initialization stage 
1. The user boots the web server by running `python app.py`.
2. As Flask spins up, the singleton `engine = MLEngine()` is initialized. At this point, the engine is inactive (`ready = False`) and awaits a dataset.
3. A user navigates to `http://127.0.0.1:5000` via a web browser. The Flask server serves the frontend (`templates/index.html`).

### Step 2: Data Ingestion (HuggingFace Integration)
1. Upon loading `index.html`, JavaScript executes a background `fetch` request directed to the **HuggingFace CDN** requesting `batuhanmtl/job-skill-set` in `.parquet` format.
2. The browser successfully downloads this binary file payload.
3. The JavaScript in the browser bounces this raw parquet binary immediately down to the local Flask server (`POST /api/ingest`).
4. `ml_engine.py` ingests the parquet (using `pandas`), cleans out erratic strings/lists, builds the semantic Document vectors, and fully fits the TF-IDF parameters.
5. The ML Engine is marked `ready = True` and tells the frontend the total jobs/skills ingested.

### Step 3: User Interaction
1. **Query Construction**: The candidate enters specific skills (e.g., "Python, AWS, Terraform").
2. **Autocomplete Assist**: As the user types "Pyth", the frontend queries `GET /api/autocomplete?q=pyth`. The ML Engine scans its global vocabulary list and returns matches, helping the user correctly define their skills to maximize dataset overlap.
3. **Filter Selection**: The user selects they are a "Senior" candidate looking for an "Engineering" role.

### Step 4: Machine Learning Inference
1. The user clicks "Find Jobs". A JSON payload with their skills and variables is POSTed to `/api/recommend`.
2. **Vector Building**: The back-end stringifies the skills and bumps the weight of "Engineering" before passing it into the pre-adjusted `TfidfVectorizer`.
3. **Similarity Calculation**: `cosine_similarity` compares the new query vector against all 300+ jobs instantly.
4. **Heuristic Modifiers**: The ML Engine scrapes job titles/descriptions for "Senior" keywords, artificially raising the score of executive/lead roles, while burying junior jobs. "Engineering" jobs get heavily favored.

### Step 5: Rendering Results
1. The algorithm thresholds low-probability matches and sorts the winners descending by score.
2. It parses the results to evaluate exactly what the user is missing versus what they hit.
3. The data array is returned to the browser.
4. The JavaScript loops over the API response to render animated matching cards—displaying the final percentage match, what skills hit (green), and what skills the user lacks (red) to visualize their skill gaps.
