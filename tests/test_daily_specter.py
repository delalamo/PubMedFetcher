import hashlib
from pathlib import Path

import numpy as np

import run
import specter_rank
from specter_rank import Paper


def _long_abstract(word: str, count: int = 100) -> str:
    return " ".join([word] * count)


def test_specter2_daily_ranking_preserves_dataframe_contract(monkeypatch, tmp_path):
    references = [
        Paper(work_id=f"ref:{index}", title=f"Reference {index}")
        for index in range(5)
    ]
    reference_embeddings = np.repeat(
        np.array([[1.0, 0.0]], dtype=np.float32), len(references), axis=0
    )

    class FakeEmbedder:
        def __init__(self, batch_size):
            assert batch_size == 8

        def encode(self, papers):
            vectors = {
                "Most relevant": [1.0, 0.0],
                "Second relevant": [0.5, np.sqrt(0.75)],
                "Below cutoff": [0.0, 1.0],
            }
            return np.asarray([vectors[paper.title] for paper in papers], dtype=np.float32)

    monkeypatch.setattr(specter_rank, "build_http_session", lambda: object())
    monkeypatch.setattr(
        specter_rank,
        "load_or_build_reference_corpus",
        lambda **kwargs: (references, {}, True),
    )
    monkeypatch.setattr(
        specter_rank,
        "load_or_build_reference_embeddings",
        lambda *args, **kwargs: (reference_embeddings, True),
    )
    monkeypatch.setattr(specter_rank, "Specter2Embedder", FakeEmbedder)

    data = {
        "Title": ["Below cutoff", "Second relevant", "Most relevant"],
        "Abstract": [
            _long_abstract("low"),
            _long_abstract("second"),
            _long_abstract("best"),
        ],
        "Journal": ["Journal C", "Journal B", "Journal A"],
        "Link": ["", "https://doi.org/10.1000/second", ""],
        "Authors": ["Author C", "Author B", "Author A"],
    }

    ranked = run.rank_papers_specter2(
        data,
        cutoff=0.35,
        top_n=2,
        cache_dir=tmp_path,
    )

    assert list(ranked.columns) == [
        "Title",
        "Abstract",
        "Journal",
        "Link",
        "Authors",
        "Relevance",
    ]
    assert ranked["Title"].tolist() == ["Most relevant", "Second relevant"]
    assert ranked["Journal"].tolist() == ["Journal A", "Journal B"]
    assert ranked["Authors"].tolist() == ["Author A", "Author B"]
    assert ranked["Relevance"].tolist() == [1.0, 0.5]


def test_email_pipeline_after_relevance_boundary_is_unchanged():
    source = Path(run.__file__).read_text(encoding="utf-8")
    marker = "    # Build recipient list - include second email if configured\n"
    downstream = source[source.index(marker) :]
    assert hashlib.sha256(downstream.encode()).hexdigest() == (
        "dd6af5dde12012e0880e4f2954e928e11a004b372444057862abe68899f8f1bb"
    )
