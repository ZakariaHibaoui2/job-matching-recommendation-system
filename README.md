# JobMatch AI — ML-Powered Job Recommendation System

> Recommends the best-fitting job category from a candidate's hard and soft skills, using a Random Forest trained on 10,000 real skill profiles blended with TF-IDF similarity.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-2.3-000000?logo=flask&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-F7931E?logo=scikitlearn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-2.0-150458?logo=pandas&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green)

🏆 **Best Project of the Semester — INTI International University, Malaysia (2026)**

---

## Overview

JobMatch AI takes a user's **hard skills** (technical abilities) and **soft skills** (interpersonal traits) and ranks 11 job categories by how well they match. Each category receives a blended score:

| Signal | Source | Weight |
|---|---|---|
| RF probability | Random Forest (200 trees, balanced class weights) trained on 10,000 profiles | **60%** |
| Cosine similarity | TF-IDF vector of the user vs. each job's skill profile | **40%** |

```python
blended = 0.60 * rf_prob + 0.40 * cos_score
```

The UI also shows which of the user's skills match each job and which skills they still need to develop.

## Features

- **Hybrid ranking**: supervised classifier plus content-based similarity, so categories missing from the training data can still be ranked.
- **Skill-gap analysis**: matched vs. missing skills per job category.
- **Autocomplete** for hard and soft skills, built from the training vocabulary.
- **Background training** on startup, with a `/api/status` readiness endpoint.
- **Responsive UI** for mobile, tablet, and desktop.
- **Model metrics** endpoint with accuracy and a per-class report.

## Tech stack

| Layer | Technology |
|---|---|
| Backend | Python, Flask |
| ML | scikit-learn (TF-IDF with 5,000 features, Random Forest), pandas, NumPy |
| Frontend | HTML, CSS, vanilla JavaScript |
| Data | `User-data-10000.csv` (10,000 profiles), `jobs_data.csv` (11 categories) |

## Project structure

```
├── app.py                  # Flask app and REST API
├── ml_engine.py            # Preprocessing, TF-IDF, Random Forest, ranking
├── templates/index.html    # Single-page responsive UI
├── User-data-10000.csv     # Training data: hard/soft skills -> field
├── jobs_data.csv           # Job catalogue with required skills
├── docs/
│   ├── TECHNICAL_GUIDE.md  # Full technical write-up (pipeline, API, design system)
│   └── job_recommender_documentation.md
└── prototype/              # First TF-IDF-only prototype
```

## Getting started

```bash
git clone https://github.com/ZakariaHibaoui2/job-matching-recommendation-system.git
cd job-matching-recommendation-system
python -m venv venv
venv\Scripts\activate          # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open <http://127.0.0.1:5000>. The model trains in the background in a few seconds; the UI shows when it's ready.

## API

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/status` | Model readiness and dataset statistics |
| `POST` | `/api/recommend` | Body: `{"hard_skills": "python, sql", "soft_skills": "leadership"}` → ranked categories |
| `GET` | `/api/autocomplete/hard?q=py` | Hard-skill suggestions |
| `GET` | `/api/autocomplete/soft?q=lea` | Soft-skill suggestions |
| `GET` | `/api/metrics` | Accuracy and classification report |

## Results

- Random Forest test accuracy: **~77%** on a held-out split (single decision tree: ~65%).
- Hard skills are weighted 2× in the document representation because they discriminate between categories more than soft skills do.

See [docs/TECHNICAL_GUIDE.md](docs/TECHNICAL_GUIDE.md) for the full pipeline, scoring math, and design notes.

## Team

- **Zakaria Hibaoui** — [@ZakariaHibaoui2](https://github.com/ZakariaHibaoui2)
- **Abdul Kuddoos Yahya** — [@yahya-n](https://github.com/yahya-n)

Built for the Machine Learning course at INTI International University, Malaysia (2026).

## License

Released under the [MIT License](LICENSE).
