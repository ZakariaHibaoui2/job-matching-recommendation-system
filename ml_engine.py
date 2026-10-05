"""
ml_engine.py
─────────────────────────────────────────────────────────────
Training data : User-data-10000.csv   (10,000 user profiles)
Job catalogue : jobs_data.csv          (11 job categories)

Pipeline
────────
1. Parse hard_skill + soft_skill lists for every user row.
2. Build a multi-label TF-IDF feature matrix over all skills.
3. Train a Random Forest to predict `candidate_field` (job category).
4. At query time:
     a. Transform user's skills with the same TF-IDF.
     b. Get per-class probabilities from the RF.
     c. For every job category, compute a cosine-similarity bonus
        between user skills and the job's required skills.
     d. Blend both scores → ranked recommendation list.
"""

import os, ast, re, logging
import numpy as np
import pandas as pd
from sklearn.ensemble            import RandomForestClassifier
from sklearn.multiclass          import OneVsRestClassifier
from sklearn.preprocessing       import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise    import cosine_similarity
from sklearn.model_selection     import train_test_split
from sklearn.metrics             import classification_report, accuracy_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

BASE = os.path.dirname(__file__)


# ── helpers ──────────────────────────────────────────────
def _parse(raw) -> list[str]:
    """Safely parse a stringified Python list of skills."""
    if pd.isna(raw):
        return []
    try:
        out = ast.literal_eval(str(raw))
        if isinstance(out, list):
            return [str(s).strip().lower() for s in out if str(s).strip()]
    except Exception:
        pass
    return [s.strip().lower() for s in re.split(r"[,;\n|]", str(raw)) if s.strip()]


def _skills_to_doc(hard: list, soft: list) -> str:
    """Combine hard + soft skills into a single text document."""
    # weight hard skills 2× by repeating them
    return " ".join(hard * 2 + soft)


# ═════════════════════════════════════════════════════════
class RecommenderEngine:
    def __init__(self):
        self.rf         = None
        self.tfidf      = None
        self.le         = None          # LabelEncoder for candidate_field
        self.jobs_df    = None          # job catalogue
        self.job_vecs   = None          # TF-IDF vectors of job profiles
        self.all_hard   = []
        self.all_soft   = []
        self.categories = []
        self.metrics    = {}
        self.ready      = False

    # ── public boot ──────────────────────────────────────
    def train(self,
              users_path: str = None,
              jobs_path:  str = None) -> dict:
        users_path = users_path or os.path.join(BASE, "User-data-10000.csv")
        jobs_path  = jobs_path  or os.path.join(BASE, "jobs_data.csv")

        log.info("Loading CSVs …")
        users_df = pd.read_csv(users_path)
        self.jobs_df = pd.read_csv(jobs_path)

        log.info("Pre-processing user data …")
        self._preprocess_users(users_df)

        log.info("Pre-processing job catalogue …")
        self._preprocess_jobs()

        log.info("Fitting TF-IDF …")
        self._fit_tfidf(users_df)

        log.info("Training Random Forest classifier …")
        self._train_rf(users_df)

        log.info("Vectorising job catalogue …")
        self._vectorise_jobs()

        self.ready = True
        log.info("Engine ready ✓")
        return {
            "users":      len(users_df),
            "jobs":       len(self.jobs_df),
            "categories": self.categories,
            "features":   int(self.tfidf.max_features or 0),
            "accuracy":   self.metrics.get("accuracy", 0),
        }

    # ── pre-processing ───────────────────────────────────
    def _preprocess_users(self, df: pd.DataFrame):
        df["_hard"] = df["hard_skill"].apply(_parse)
        df["_soft"] = df["soft_skill"].apply(_parse)
        df["_doc"]  = df.apply(lambda r: _skills_to_doc(r["_hard"], r["_soft"]), axis=1)

        hard_all, soft_all = set(), set()
        for h in df["_hard"]: hard_all.update(h)
        for s in df["_soft"]: soft_all.update(s)
        self.all_hard = sorted(hard_all)
        self.all_soft = sorted(soft_all)

        self.categories = sorted(df["candidate_field"].dropna().unique().tolist())

    def _preprocess_jobs(self):
        self.jobs_df["_hard"] = self.jobs_df["Hard Skills"].apply(_parse)
        self.jobs_df["_soft"] = self.jobs_df["Soft Skills"].apply(_parse)
        self.jobs_df["_doc"]  = self.jobs_df.apply(
            lambda r: _skills_to_doc(r["_hard"], r["_soft"]), axis=1)
        # normalise Major to lowercase for alignment
        self.jobs_df["_major"] = self.jobs_df["Major"].str.lower().str.strip()

    # ── TF-IDF ───────────────────────────────────────────
    def _fit_tfidf(self, df: pd.DataFrame):
        self.tfidf = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=5000,
            sublinear_tf=True,
            min_df=2,
        )
        self.tfidf.fit(df["_doc"])

    # ── Random Forest ────────────────────────────────────
    def _train_rf(self, df: pd.DataFrame):
        X = self.tfidf.transform(df["_doc"])
        self.le = LabelEncoder()
        y = self.le.fit_transform(df["candidate_field"])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        self.rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=4,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        )
        self.rf.fit(X_train, y_train)

        y_pred = self.rf.predict(X_test)
        acc    = accuracy_score(y_test, y_pred)
        report = classification_report(
            y_test, y_pred,
            target_names=self.le.classes_,
            output_dict=True,
        )
        self.metrics = {
            "accuracy":  round(float(acc) * 100, 2),
            "report":    report,
            "n_train":   int(X_train.shape[0]),
            "n_test":    int(X_test.shape[0]),
        }
        log.info(f"RF accuracy: {acc*100:.1f}%")

    # ── Job vectors ──────────────────────────────────────
    def _vectorise_jobs(self):
        self.job_vecs = self.tfidf.transform(self.jobs_df["_doc"])

    # ── Recommend ────────────────────────────────────────
    def recommend(
        self,
        hard_skills : list[str],
        soft_skills : list[str],
        top_n       : int = 11,
    ) -> list[dict]:
        if not self.ready:
            return []

        # Build user document
        user_doc = _skills_to_doc(
            [s.strip().lower() for s in hard_skills],
            [s.strip().lower() for s in soft_skills],
        )
        user_vec = self.tfidf.transform([user_doc])

        # ── RF class probabilities ────────────────────────
        proba      = self.rf.predict_proba(user_vec)[0]       # shape (n_classes,)
        class_prob = {
            self.le.classes_[i]: float(proba[i])
            for i in range(len(self.le.classes_))
        }

        # ── Cosine similarity against job catalogue ───────
        cos_scores = cosine_similarity(user_vec, self.job_vecs).flatten()

        # ── Blend: 60% RF prob + 40% cosine ──────────────
        user_set  = set(hard_skills + soft_skills)
        results   = []

        for idx, jrow in self.jobs_df.iterrows():
            major   = jrow["_major"]
            rf_prob = class_prob.get(major, 0.0)
            cos_sc  = float(cos_scores[idx])
            blended = 0.60 * rf_prob + 0.40 * cos_sc

            job_skills = jrow["_hard"] + jrow["_soft"]
            job_set    = set(job_skills)
            matched    = sorted(user_set & job_set)
            missing    = sorted(job_set - user_set)
            match_pct  = round(len(matched) / max(len(job_set), 1) * 100)

            results.append({
                "job_id"        : int(jrow["Job ID"]),
                "category"      : jrow["Major"],
                "hard_skills"   : jrow["_hard"],
                "soft_skills"   : jrow["_soft"],
                "matched_skills": matched,
                "missing_skills": missing[:8],
                "match_pct"     : match_pct,
                "rf_score"      : round(rf_prob * 100, 1),
                "cos_score"     : round(cos_sc  * 100, 1),
                "score"         : round(blended  * 100, 1),
            })

        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:top_n]

    # ── Autocomplete ─────────────────────────────────────
    def suggest_hard(self, q: str) -> list[str]:
        q = q.lower()
        return [s for s in self.all_hard if q in s][:14]

    def suggest_soft(self, q: str) -> list[str]:
        q = q.lower()
        return [s for s in self.all_soft if q in s][:14]

    # ── Stats ─────────────────────────────────────────────
    def stats(self) -> dict:
        if not self.ready:
            return {}
        return {
            "total_users"  : self.metrics["n_train"] + self.metrics["n_test"],
            "total_jobs"   : len(self.jobs_df),
            "total_hard"   : len(self.all_hard),
            "total_soft"   : len(self.all_soft),
            "accuracy"     : self.metrics["accuracy"],
            "categories"   : self.categories,
            "rf_report"    : self.metrics["report"],
        }


# Singleton
engine = RecommenderEngine()