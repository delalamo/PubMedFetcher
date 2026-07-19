"""Network-independent scholarly-API smoke test for pinned SPECTER2 inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from specter_rank import OperationalError, Paper, Specter2Embedder


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixture",
        default="tests/fixtures/specter2_smoke.json",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()
    payload = json.loads(Path(args.fixture).read_text(encoding="utf-8"))
    papers = [Paper.from_dict(item) for item in payload["papers"]]
    if len(papers) < 2:
        raise OperationalError("The SPECTER2 smoke fixture needs at least two papers")
    embedder = Specter2Embedder(batch_size=args.batch_size)
    if "proximity" not in str(embedder.model.active_adapters):
        raise OperationalError("The proximity adapter is not active in the smoke test")
    embeddings = embedder.encode(papers)
    if embeddings.shape[0] != len(papers) or embeddings.shape[1] < 100:
        raise OperationalError(f"Unexpected SPECTER2 smoke shape: {embeddings.shape}")
    if not np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-5):
        raise OperationalError("SPECTER2 smoke embeddings are not L2 normalized")
    similarities = embeddings @ embeddings.T
    if not np.isfinite(similarities).all():
        raise OperationalError("SPECTER2 smoke similarities contain non-finite values")
    print(
        json.dumps(
            {
                "fixture": args.fixture,
                "paper_count": len(papers),
                "embedding_shape": list(embeddings.shape),
                "active_adapter": str(embedder.model.active_adapters),
                "minimum_self_similarity": float(np.diag(similarities).min()),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
