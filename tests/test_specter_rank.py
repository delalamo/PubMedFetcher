import csv
import json

import numpy as np
import pytest

from specter_rank import (
    Neighbor,
    RESULT_REQUIRED_FIELDS,
    Paper,
    ScoreSet,
    build_ranked_results,
    calculate_backtest_metrics,
    calculate_scores,
    l2_normalize,
    merge_duplicate_works,
    normalize_arxiv_id,
    normalize_doi,
    normalize_title,
    parse_bibliography,
    validate_report_schema,
    write_reports,
)


def paper(number, *, title=None, year=2020, label=None, abstract="abstract"):
    return Paper(
        work_id=f"paper:{number}",
        title=title or f"Paper {number}",
        abstract=abstract,
        source="fixture",
        publication_date=str(year),
        year=year,
        source_id=str(number),
        label=label,
    )


def test_identifier_and_title_normalization():
    assert normalize_doi("HTTPS://doi.org/10.1000/ABC.12.") == "10.1000/abc.12"
    assert normalize_arxiv_id("https://arxiv.org/pdf/2401.01234v2.pdf") == "2401.01234"
    assert normalize_arxiv_id("10.48550/arXiv.2401.01234") == "2401.01234"
    assert normalize_arxiv_id("https://europepmc.org/article/MED/4236678") == ""
    assert normalize_title("{A  Méthod}: for RNA—Seq") == "a method for rna seq"


def test_bibtex_parsing_normalizes_fields_and_keeps_title_only():
    entries = parse_bibliography(
        r"""
        @article{Example2025,
          title = {{A Useful Method}},
          author = {Doe, Jane and Roe, Richard},
          year = {2025},
          month = july,
          doi = {https://doi.org/10.1000/EXAMPLE},
          url = {https://doi.org/10.1000/EXAMPLE}
        }
        """
    )
    assert len(entries) == 1
    assert entries[0].doi == "10.1000/example"
    assert entries[0].year == 2025
    assert entries[0].authors == ["Doe, Jane", "Roe, Richard"]
    assert not entries[0].has_abstract
    assert entries[0].title == "A Useful Method"


def test_duplicate_merging_unifies_preprint_and_published_title():
    preprint = Paper(
        work_id="arxiv:2401.01234",
        title="A new protein structure model",
        abstract="Long preprint abstract",
        year=2024,
        source="arxiv",
        source_id="2401.01234",
        bibkeys=["preprint"],
    )
    published = Paper(
        work_id="doi:10.1000/model",
        title="A New Protein-Structure Model",
        year=2025,
        source="published",
        doi="10.1000/model",
        aliases=["2401.01234"],
        bibkeys=["journal"],
    )
    merged = merge_duplicate_works([preprint, published])
    assert len(merged) == 1
    assert merged[0].year == 2024
    assert merged[0].doi == "10.1000/model"
    assert merged[0].abstract == "Long preprint abstract"
    assert merged[0].bibkeys == ["journal", "preprint"]


def test_l2_normalization_and_invalid_zero_vector():
    normalized = l2_normalize(np.array([[3.0, 4.0], [1.0, 0.0]], dtype=np.float32))
    assert np.allclose(np.linalg.norm(normalized, axis=1), 1.0)
    with pytest.raises(Exception, match="zero"):
        l2_normalize(np.array([[0.0, 0.0]], dtype=np.float32))


def test_top_five_scoring_and_three_explanations():
    references = [paper(index) for index in range(10)]
    reference_vectors = np.eye(10, dtype=np.float32)
    candidate_vectors = np.array([[5, 4, 3, 2, 1, 0, 0, 0, 0, 0]], dtype=np.float32)
    scores = calculate_scores(candidate_vectors, reference_vectors, references)
    normalized = candidate_vectors[0] / np.linalg.norm(candidate_vectors[0])
    assert scores.top5_mean[0] == pytest.approx(float(np.mean(normalized[:5])))
    assert [neighbor.work_id for neighbor in scores.neighbors[0]] == [
        "paper:0",
        "paper:1",
        "paper:2",
    ]


def test_deterministic_tie_order_and_result_fields():
    candidates = [paper("b"), paper("a")]
    references = [paper(index) for index in range(3)]
    score_set = ScoreSet(
        top5_mean=np.array([0.5, 0.5]),
        centroid=np.array([0.2, 0.2]),
        top1=np.array([0.6, 0.6]),
        top10_mean=np.array([0.4, 0.4]),
        neighbors=[
            [
                Neighbor(
                    work_id=ref.work_id,
                    title=ref.title,
                    identifier=ref.identifier(),
                    similarity=0.5,
                )
                for ref in references
            ]
            for _ in candidates
        ],
    )
    results = build_ranked_results(candidates, score_set)
    assert [result["work_id"] for result in results] == ["paper:a", "paper:b"]
    assert results[0]["score"] == 0.5
    assert RESULT_REQUIRED_FIELDS <= set(results[0])


def test_backtest_metrics_cover_all_variants_without_raising_quality_failures():
    candidates = [paper(index, label=int(index < 2)) for index in range(8)]
    scores = ScoreSet(
        top5_mean=np.arange(8, 0, -1, dtype=np.float32),
        centroid=np.arange(8, 0, -1, dtype=np.float32),
        top1=np.arange(8, 0, -1, dtype=np.float32),
        top10_mean=np.arange(8, 0, -1, dtype=np.float32),
        neighbors=[[] for _ in candidates],
    )
    metrics = calculate_backtest_metrics(scores, candidates)
    assert set(metrics["variants"]) == {
        "mean_top_five",
        "global_centroid",
        "single_nearest",
        "mean_top_ten",
    }
    assert metrics["variants"]["mean_top_five"]["average_precision"] == 1.0
    assert metrics["quality_checks"]["informational_only"] is True


def test_report_schema_and_artifacts(tmp_path):
    references = [paper(index) for index in range(3)]
    candidates = [paper("candidate")]
    embeddings = np.eye(3, dtype=np.float32)
    scores = calculate_scores(
        np.array([[1.0, 0.0, 0.0]], dtype=np.float32), embeddings, references
    )
    results = build_ranked_results(candidates, scores)
    report = {
        "schema_version": 1,
        "run": {"mode": "live", "lookback_days": 7, "max_candidates": 300, "top_n": 25},
        "model": {
            "base": "base",
            "base_revision": "base-sha",
            "adapter": "adapter",
            "adapter_revision": "adapter-sha",
        },
        "bibliography": {
            "commit": "bib-sha",
            "content_sha256": "content-sha",
            "coverage": {},
        },
        "statistics": {
            "candidate_count": 1,
            "reference_count": 3,
            "reference_corpus_count": 3,
            "reference_abstract_count": 3,
            "reference_abstract_fraction": 1.0,
        },
        "cache": {
            "runtime_class": "warm",
            "reference_corpus_hit": True,
            "reference_embeddings_hit": True,
        },
        "timings_seconds": {"total": 1.0},
        "versions": {},
        "candidate_retrieval": {},
        "metrics": None,
        "results": results,
    }
    validate_report_schema(report)
    write_reports(report, tmp_path, top_n=25)
    assert json.loads((tmp_path / "specter2_results.json").read_text())["results"]
    with (tmp_path / "specter2_results.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["bibliography_commit"] == "bib-sha"
    assert "SPECTER2 paper-ranking evaluation" in (
        tmp_path / "specter2_results.md"
    ).read_text()
