import os
import sys
import sqlite3
import numpy as np
import config

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import faiss
except Exception:
    faiss = None


def _as_f32_contig(x: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(x.astype(np.float32))


def _l2_normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-wise L2 normalization (safe for cosine/IP)."""
    x = _as_f32_contig(x)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return x / norms


def _l2_normalize_vec(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = _as_f32_contig(x.reshape(-1))
    n = float(np.linalg.norm(x))
    if n < eps:
        return x
    return x / n


class AnchorMatcher:
    """
    Adds FAISS-based retrieval + per-person aggregation on top of:
      - self._cache_embs (N, D) float32, L2-normalized
      - self._cache_name_to_rows: dict[str, list[int]] or dict[str, np.ndarray]
    """

    def __init__(self):
        self._cache_embs: np.ndarray | None = None
        self._cache_name_to_rows: dict[str, list[int] | np.ndarray] = {}
        self._cache_row_to_name: list[str] | None = None

        self._faiss_index = None
        self._faiss_index_type: str | None = None

        # to detect when cache changed
        self._cache_version: int = 0
        self._faiss_built_version: int = -1

    def invalidate_indices(self):
        self._faiss_index = None
        self._faiss_index_type = None
        self._faiss_built_version = -1
        self._cache_row_to_name = None

    def _build_row_to_name(self):
        if self._cache_embs is None:
            raise RuntimeError("Cache not loaded.")
        n = int(self._cache_embs.shape[0])
        row_to_name = [None] * n

        for name, idxs in self._cache_name_to_rows.items():
            for r in idxs:
                row_to_name[int(r)] = name

        if any(v is None for v in row_to_name):
            raise RuntimeError("row_to_name has None entries. Cache mapping is inconsistent.")
        self._cache_row_to_name = row_to_name

    def build_faiss_index(
        self,
        index_type: str = "hnsw",
        *,
        nlist: int = 256,
        nprobe: int = 32,
        hnsw_m: int = 32,
        ef_search: int = 64,
        ef_construction: int = 200,
        pq_m: int = 16,
        pq_nbits: int = 8,
        use_gpu: bool = False,
    ):
        if faiss is None:
            raise ImportError("faiss is not installed. Run: pip install faiss-cpu")

        if self._cache_embs is None:
            self.load_cache()

        if self._cache_embs is None or self._cache_embs.size == 0:
            self.invalidate_indices()
            return

        if self._cache_row_to_name is None:
            self._build_row_to_name()

        embs = _as_f32_contig(self._cache_embs)
        n, d = embs.shape

        index_type = index_type.lower().strip()

        if index_type == "flatip":
            index = faiss.IndexFlatIP(d)

        elif index_type == "hnsw":
            index = faiss.IndexHNSWFlat(d, int(hnsw_m), faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efSearch = int(ef_search)
            index.hnsw.efConstruction = int(ef_construction)

        elif index_type == "ivf_flat":
            quantizer = faiss.IndexFlatIP(d)
            nlist_eff = int(min(nlist, max(1, int(np.sqrt(n)))))
            index = faiss.IndexIVFFlat(quantizer, d, nlist_eff, faiss.METRIC_INNER_PRODUCT)
            index.train(embs)
            index.nprobe = int(min(nprobe, nlist_eff))

        elif index_type == "ivf_pq":
            quantizer = faiss.IndexFlatIP(d)
            nlist_eff = int(min(nlist, max(1, int(np.sqrt(n)))))
            if d % pq_m != 0:
                raise ValueError(f"ivf_pq requires embedding_dim % pq_m == 0. Got d={d}, pq_m={pq_m}")
            index = faiss.IndexIVFPQ(
                quantizer, d, nlist_eff, int(pq_m), int(pq_nbits), faiss.METRIC_INNER_PRODUCT
            )
            index.train(embs)
            index.nprobe = int(min(nprobe, nlist_eff))

        else:
            raise ValueError("Unknown index_type. Use: flatip | hnsw | ivf_flat | ivf_pq")

        index.add(embs)

        if use_gpu:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)

        self._faiss_index = index
        self._faiss_index_type = index_type
        self._faiss_built_version = self._cache_version

    def find_closest(
        self,
        target_embedding: np.ndarray,
        threshold=config.SIMILARITY_THRESHOLD,
        aggregation="topk",
        top_k=3,
        *,
        backend: str = "faiss",
        index_type: str = "hnsw",
        faiss_k: int = 120,
    ):
        if self._cache_embs is None:
            self.load_cache()

        if self._cache_embs is None or self._cache_embs.size == 0:
            return None, 0.0

        # Normalize query (keeps cosine/IP consistent even if caller forgets)
        target_embedding = _l2_normalize_vec(target_embedding)

        backend = backend.lower().strip()

        # ---- exact numpy fallback ----
        if backend == "numpy" or faiss is None:
            sims = self._cache_embs @ target_embedding  # (N,)

            best_name = None
            best_score = -1.0

            for name, idxs in self._cache_name_to_rows.items():
                person_sims = sims[idxs]
                if aggregation == "max":
                    score = float(np.max(person_sims))
                else:
                    k = min(int(top_k), int(person_sims.shape[0]))
                    top = np.partition(person_sims, -k)[-k:]
                    score = float(np.mean(top))

                if score > best_score:
                    best_score = score
                    best_name = name

            return (best_name, best_score) if best_score >= threshold else (None, best_score)

        # ---- FAISS path ----
        if self._cache_row_to_name is None:
            self._build_row_to_name()

        # rebuild if never built, index type changed, or cache changed
        if (
            self._faiss_index is None
            or self._faiss_index_type != index_type
            or self._faiss_built_version != self._cache_version
        ):
            self.build_faiss_index(index_type=index_type)

        if self._faiss_index is None:
            return None, 0.0

        N = int(self._cache_embs.shape[0])
        k = int(min(max(int(faiss_k), int(top_k)), N))

        D, I = self._faiss_index.search(target_embedding.reshape(1, -1), k)
        sims = D[0]
        rows = I[0]

        per_person: dict[str, list[float]] = {}
        for sim, r in zip(sims, rows):
            if r < 0:
                continue
            name = self._cache_row_to_name[int(r)]
            per_person.setdefault(name, []).append(float(sim))

        if not per_person:
            return None, -1.0

        best_name = None
        best_score = -1.0

        for name, scores in per_person.items():
            if aggregation == "max":
                score = max(scores)
            else:
                kk = min(int(top_k), len(scores))
                score = float(np.mean(np.sort(scores)[-kk:]))

            if score > best_score:
                best_score = score
                best_name = name

        return (best_name, best_score) if best_score >= threshold else (None, best_score)


class FaceDatabase(AnchorMatcher):
    def __init__(self, db_path=config.DB_PATH):
        super().__init__()
        self.db_path = db_path
        self._init_db()

        self._cache_names: np.ndarray | None = None  # optional

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS faces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                embedding BLOB NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_faces_name ON faces(name)")
        conn.commit()
        conn.close()

    def add_person(self, name: str, embedding: np.ndarray, *, normalize: bool = True):
        emb = _l2_normalize_vec(embedding) if normalize else _as_f32_contig(embedding)
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("INSERT INTO faces (name, embedding) VALUES (?, ?)", (name, emb.tobytes()))
        conn.commit()
        conn.close()

        # cache/index are now stale
        self._cache_embs = None
        self._cache_name_to_rows = {}
        self._cache_names = None
        self.invalidate_indices()

    def add_person_many(self, name: str, embeddings, *, normalize: bool = True):
        rows = []
        for emb in embeddings:
            e = _l2_normalize_vec(emb) if normalize else _as_f32_contig(emb)
            rows.append((name, e.tobytes()))

        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.executemany("INSERT INTO faces (name, embedding) VALUES (?, ?)", rows)
        conn.commit()
        conn.close()

        self._cache_embs = None
        self._cache_name_to_rows = {}
        self._cache_names = None
        self.invalidate_indices()

    def get_all_embeddings(self):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("SELECT id, name, embedding FROM faces")
        rows = cur.fetchall()
        conn.close()

        results = []
        for face_id, name, blob in rows:
            emb = np.frombuffer(blob, dtype=np.float32)
            results.append((face_id, name, emb))
        return results

    def load_cache(self):
        faces = self.get_all_embeddings()
        if not faces:
            self._cache_names = np.array([], dtype=object)
            self._cache_embs = np.zeros((0, 0), dtype=np.float32)
            self._cache_name_to_rows = {}
            self.invalidate_indices()
            self._cache_version += 1
            return

        names = []
        embs = []
        for _, name, emb in faces:
            names.append(name)
            embs.append(emb.astype(np.float32))

        embs = np.stack(embs, axis=0)           # (N, D)
        embs = _l2_normalize_rows(embs)         # ensure cosine/IP correctness

        self._cache_names = np.array(names, dtype=object)
        self._cache_embs = _as_f32_contig(embs)

        name_to_rows: dict[str, list[int]] = {}
        for i, nm in enumerate(self._cache_names):
            name_to_rows.setdefault(str(nm), []).append(i)
        self._cache_name_to_rows = name_to_rows

        self.invalidate_indices()
        self._cache_version += 1
    # ── Management methods ─────────────────────────────────────────────────────

    def delete_by_id(self, face_id: int) -> bool:
        """Delete a single embedding row by its DB id. Returns True if found."""
        conn = sqlite3.connect(self.db_path)
        cur  = conn.cursor()
        cur.execute("DELETE FROM faces WHERE id = ?", (face_id,))
        deleted = cur.rowcount > 0
        conn.commit()
        conn.close()
        if deleted:
            self._cache_embs = None
            self._cache_name_to_rows = {}
            self._cache_names = None
            self.invalidate_indices()
        return deleted

    def delete_by_name(self, name: str) -> int:
        """Delete ALL embeddings for a given name. Returns number of rows deleted."""
        conn = sqlite3.connect(self.db_path)
        cur  = conn.cursor()
        cur.execute("DELETE FROM faces WHERE name = ?", (name,))
        count = cur.rowcount
        conn.commit()
        conn.close()
        if count:
            self._cache_embs = None
            self._cache_name_to_rows = {}
            self._cache_names = None
            self.invalidate_indices()
        return count

    def rename_person(self, old_name: str, new_name: str) -> int:
        """Rename all embeddings from old_name to new_name. Returns rows updated."""
        conn = sqlite3.connect(self.db_path)
        cur  = conn.cursor()
        cur.execute("UPDATE faces SET name = ? WHERE name = ?", (new_name, old_name))
        count = cur.rowcount
        conn.commit()
        conn.close()
        if count:
            self._cache_embs = None
            self._cache_name_to_rows = {}
            self._cache_names = None
            self.invalidate_indices()
        return count

    def get_summary(self) -> list[dict]:
        """Return one row per distinct name with embedding count and latest timestamp."""
        conn = sqlite3.connect(self.db_path)
        cur  = conn.cursor()
        cur.execute("""
            SELECT name, COUNT(*) as count, MAX(created_at) as last_seen
            FROM faces
            GROUP BY name
            ORDER BY name
        """)
        rows = cur.fetchall()
        conn.close()
        return [{"name": r[0], "count": r[1], "last_seen": r[2]} for r in rows]
