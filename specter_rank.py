"""Standalone SPECTER2 paper-ranking evaluation.

This module intentionally has no dependency on ``run.py``.  It builds a weak
positive reference corpus from a pinned SKM bibliography, embeds references and
candidates locally with SPECTER2, and writes review-only reports.  It never
sends email, creates issues, or calls a hosted inference API.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import os
import re
import statistics
import sys
import time
import unicodedata
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field, replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence
from urllib.parse import quote, urlparse

import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# This commit has the 658-entry bibliography reviewed for this evaluation.
# A later SKM head added another entry on 2026-07-17, so do not follow main.
SKM_COMMIT = "cf953b074ee610bc8cb528226e605cc921ecdca4"
SKM_BIB_URL = (
    "https://raw.githubusercontent.com/delalamo/SKM/"
    f"{SKM_COMMIT}/bibliography.bib"
)
SPECTER2_BASE = "allenai/specter2_base"
SPECTER2_BASE_REVISION = "3447645e1def9117997203454fa4495937bfbd83"
SPECTER2_ADAPTER = "allenai/specter2"
SPECTER2_ADAPTER_REVISION = "2081559630a80fc5851d8f798a05ba81e9468089"
SCHEMA_VERSION = 1
CONTROL_SNAPSHOT_VERSION = 1
REPORT_SIMILARITY_DECIMALS = 4
DEFAULT_USER_AGENT = (
    "PubMedFetcher-SPECTER2-evaluation/1.0 "
    "(https://github.com/delalamo/PubMedFetcher)"
)
ARXIV_CATEGORIES = ("q-bio", "cs.LG", "cs.AI", "stat.ML")
LONG_MONTH_MACROS = "\n".join(
    f'@string{{{month} = "{month.title()}"}}'
    for month in (
        "january", "february", "march", "april", "june",
        "july", "august", "september", "october", "november", "december",
    )
)


class OperationalError(RuntimeError):
    """A failure that should make the evaluation workflow exit nonzero."""


@dataclass
class Paper:
    """Normalized metadata for a distinct scholarly work."""

    work_id: str
    title: str
    abstract: str = ""
    authors: list[str] = field(default_factory=list)
    source: str = "unknown"
    publication_date: str = ""
    year: int | None = None
    doi: str = ""
    source_id: str = ""
    url: str = ""
    bibkeys: list[str] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)
    label: int | None = None

    @property
    def has_abstract(self) -> bool:
        return bool(self.abstract.strip())

    def identifier(self) -> str:
        return self.doi or self.source_id or self.work_id

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Paper":
        allowed = {item.name for item in cls.__dataclass_fields__.values()}
        return cls(**{key: value[key] for key in allowed if key in value})


@dataclass(frozen=True)
class Neighbor:
    work_id: str
    title: str
    identifier: str
    similarity: float


@dataclass
class ScoreSet:
    top5_mean: np.ndarray
    centroid: np.ndarray
    top1: np.ndarray
    top10_mean: np.ndarray
    neighbors: list[list[Neighbor]]


def collapse_whitespace(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def strip_markup(value: Any) -> str:
    """Remove common BibTeX/HTML/JATS wrappers without guessing semantics."""

    text = html.unescape(collapse_whitespace(value))
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("{", "").replace("}", "")
    replacements = {
        r"\&": "&",
        r"\_": "_",
        r"\%": "%",
        r"\textendash": "-",
        r"\textemdash": "-",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"\\[A-Za-z]+\s*", "", text)
    return collapse_whitespace(text)


def normalize_doi(value: Any) -> str:
    text = collapse_whitespace(value).lower()
    text = re.sub(r"^(?:https?://)?(?:dx\.)?doi\.org/", "", text)
    text = re.sub(r"^doi\s*:\s*", "", text)
    text = text.strip().rstrip(".,;)")
    match = re.search(r"10\.\d{4,9}/\S+", text)
    return match.group(0).rstrip(".,;)") if match else ""


def normalize_arxiv_id(value: Any) -> str:
    text = collapse_whitespace(value)
    text = re.sub(r"^(?:https?://)?(?:www\.)?arxiv\.org/(?:abs|pdf)/", "", text, flags=re.I)
    text = re.sub(r"^arxiv\s*:\s*", "", text, flags=re.I)
    text = re.sub(r"^10\.48550/arxiv\.", "", text, flags=re.I)
    text = text.removesuffix(".pdf")
    text = re.sub(r"v\d+$", "", text, flags=re.I)
    match = re.fullmatch(r"(?:[a-z][a-z0-9.-]*/\d{7}|\d{4}\.\d{4,5})", text, flags=re.I)
    return match.group(0).lower() if match else ""


def normalize_title(value: Any) -> str:
    text = strip_markup(value).casefold()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return collapse_whitespace(text)


def parse_year(value: Any) -> int | None:
    match = re.search(r"(?:19|20)\d{2}", collapse_whitespace(value))
    return int(match.group(0)) if match else None


def parse_authors(value: Any) -> list[str]:
    text = strip_markup(value)
    if not text:
        return []
    if " and " in text:
        parts = text.split(" and ")
    elif ";" in text:
        parts = text.split(";")
    else:
        parts = [text]
    return [collapse_whitespace(part) for part in parts if collapse_whitespace(part)]


def infer_source(doi: str, arxiv_id: str, entry_type: str = "") -> str:
    if arxiv_id or doi.startswith("10.48550/arxiv."):
        return "arxiv"
    if doi.startswith("10.1101/") or doi.startswith("10.64898/"):
        return "biorxiv/medrxiv"
    if entry_type.lower() in {"article", "inproceedings", "proceedings"}:
        return "published"
    return "bibliography"


def stable_work_id(
    *, doi: str = "", arxiv_id: str = "", source: str = "", source_id: str = "",
    title: str = "", year: int | None = None,
) -> str:
    if doi:
        return f"doi:{doi}"
    if arxiv_id:
        return f"arxiv:{arxiv_id}"
    if source_id:
        return f"{source or 'source'}:{source_id.lower()}"
    normalized = normalize_title(title)
    digest = hashlib.sha256(f"{normalized}|{year or ''}".encode()).hexdigest()[:20]
    return f"title:{digest}"


def _bibtex_value(entry: dict[str, Any], *keys: str) -> str:
    for key in keys:
        if entry.get(key):
            return collapse_whitespace(entry[key])
    return ""


def parse_bibliography(text: str) -> list[Paper]:
    """Parse BibTeX into normalized papers; duplicate merging is separate."""

    try:
        import bibtexparser
        from bibtexparser.bparser import BibTexParser
    except ImportError as exc:  # pragma: no cover - dependency error path
        raise OperationalError("bibtexparser is required to parse bibliography.bib") from exc

    try:
        parser = BibTexParser(common_strings=True)
        database = bibtexparser.loads(f"{LONG_MONTH_MACROS}\n{text}", parser=parser)
    except Exception as exc:
        raise OperationalError(f"Malformed BibTeX input: {exc}") from exc
    papers: list[Paper] = []
    for entry in database.entries:
        title = strip_markup(_bibtex_value(entry, "title"))
        if not title:
            continue
        doi = normalize_doi(_bibtex_value(entry, "doi", "url"))
        arxiv_id = normalize_arxiv_id(
            _bibtex_value(entry, "eprint", "arxiv", "arxivid", "url", "doi")
        )
        year = parse_year(_bibtex_value(entry, "year", "date", "urldate"))
        source = infer_source(doi, arxiv_id, entry.get("ENTRYTYPE", ""))
        source_id = arxiv_id or _bibtex_value(entry, "pmid", "pubmed", "eid")
        url = _bibtex_value(entry, "url")
        if not url:
            if doi:
                url = f"https://doi.org/{doi}"
            elif arxiv_id:
                url = f"https://arxiv.org/abs/{arxiv_id}"
        paper = Paper(
            work_id=stable_work_id(
                doi=doi,
                arxiv_id=arxiv_id,
                source=source,
                source_id=source_id,
                title=title,
                year=year,
            ),
            title=title,
            abstract=strip_markup(_bibtex_value(entry, "abstract")),
            authors=parse_authors(_bibtex_value(entry, "author")),
            source=source,
            publication_date=_bibtex_value(entry, "date") or (str(year) if year else ""),
            year=year,
            doi=doi,
            source_id=source_id,
            url=url,
            bibkeys=[collapse_whitespace(entry.get("ID", ""))],
            aliases=extract_identifier_aliases(entry),
        )
        papers.append(paper)
    if not papers:
        raise OperationalError("The parsed bibliography contains no titled entries")
    return papers


class _DisjointSet:
    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def _identifier_tokens(paper: Paper) -> set[str]:
    values = {
        paper.doi,
        normalize_arxiv_id(paper.doi),
        normalize_arxiv_id(paper.source_id),
        *paper.aliases,
    }
    return {value for value in values if value}


def extract_identifier_aliases(entry: dict[str, Any]) -> list[str]:
    aliases: set[str] = set()
    for key in ("doi", "url", "eprint", "related", "relateddoi", "preprint", "note"):
        value = collapse_whitespace(entry.get(key))
        for match in re.findall(r"10\.\d{4,9}/[^\s,;}]+", value, flags=re.I):
            doi = normalize_doi(match)
            if doi:
                aliases.add(doi)
        arxiv_id = normalize_arxiv_id(value)
        if arxiv_id:
            aliases.add(arxiv_id)
    return sorted(aliases)


def merge_duplicate_works(papers: Sequence[Paper]) -> list[Paper]:
    """Merge exact DOI/arXiv/title matches so a work contributes one vote."""

    dsu = _DisjointSet(len(papers))
    identifier_owner: dict[str, int] = {}
    title_owner: dict[str, int] = {}
    for index, paper in enumerate(papers):
        for token in _identifier_tokens(paper):
            if token in identifier_owner:
                dsu.union(index, identifier_owner[token])
            else:
                identifier_owner[token] = index
        title_key = normalize_title(paper.title)
        if title_key:
            if title_key in title_owner:
                dsu.union(index, title_owner[title_key])
            else:
                title_owner[title_key] = index

    groups: dict[int, list[Paper]] = {}
    for index, paper in enumerate(papers):
        groups.setdefault(dsu.find(index), []).append(paper)

    merged: list[Paper] = []
    for group in groups.values():
        # Prefer rich metadata, but use the earliest date to prevent a preprint's
        # later journal version leaking from a backtest holdout into training.
        richest = max(
            group,
            key=lambda paper: (
                paper.has_abstract,
                len(paper.abstract),
                bool(paper.doi),
                len(paper.authors),
                -len(paper.title),
            ),
        )
        years = [paper.year for paper in group if paper.year]
        doi = next((paper.doi for paper in group if paper.doi), "")
        arxiv_id = next(
            (
                normalize_arxiv_id(value)
                for paper in group
                for value in (paper.source_id, paper.doi, paper.url)
                if normalize_arxiv_id(value)
            ),
            "",
        )
        source_id = arxiv_id or next((paper.source_id for paper in group if paper.source_id), "")
        source = infer_source(doi, arxiv_id) if (doi or arxiv_id) else richest.source
        year = min(years) if years else None
        merged.append(
            replace(
                richest,
                work_id=stable_work_id(
                    doi=doi,
                    arxiv_id=arxiv_id,
                    source=source,
                    source_id=source_id,
                    title=richest.title,
                    year=year,
                ),
                year=year,
                publication_date=str(year) if year else richest.publication_date,
                doi=doi,
                source_id=source_id,
                source=source,
                bibkeys=sorted(
                    {key for paper in group for key in paper.bibkeys if key}
                ),
                aliases=sorted(
                    {alias for paper in group for alias in paper.aliases if alias}
                ),
                url=next((paper.url for paper in group if paper.url), richest.url),
            )
        )
    return sorted(merged, key=lambda paper: (paper.year or 9999, normalize_title(paper.title)))


def deduplicate_candidates(papers: Sequence[Paper]) -> list[Paper]:
    """Normalize candidates and merge repeats without retaining positive labels."""

    merged = merge_duplicate_works(papers)
    seen: set[str] = set()
    for paper in merged:
        if paper.work_id in seen:
            raise OperationalError(f"Duplicate candidate ID after normalization: {paper.work_id}")
        seen.add(paper.work_id)
    return merged


def build_http_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=4,
        connect=4,
        read=4,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers.update({"User-Agent": os.getenv("SPECTER2_USER_AGENT", DEFAULT_USER_AGENT)})
    return session


def batched(values: Sequence[Any], size: int) -> Iterator[Sequence[Any]]:
    for index in range(0, len(values), size):
        yield values[index : index + size]


def _safe_get_json(
    session: requests.Session,
    url: str,
    *,
    params: dict[str, Any] | None = None,
    timeout: int = 45,
) -> dict[str, Any]:
    response = session.get(url, params=params, timeout=timeout)
    response.raise_for_status()
    value = response.json()
    if not isinstance(value, dict):
        raise OperationalError(f"Unexpected JSON response from {url}")
    return value


def _epmc_result_to_paper(result: dict[str, Any], default_source: str = "europe-pmc") -> Paper:
    doi = normalize_doi(result.get("doi"))
    source_id = collapse_whitespace(
        result.get("pmid") or result.get("pmcid") or result.get("id") or result.get("extId")
    )
    title = strip_markup(result.get("title"))
    publication_date = collapse_whitespace(
        result.get("firstPublicationDate")
        or result.get("electronicPublicationDate")
        or result.get("journalInfo", {}).get("printPublicationDate")
        or result.get("pubYear")
    )
    year = parse_year(publication_date or result.get("pubYear"))
    author_text = result.get("authorString") or ""
    url = f"https://doi.org/{doi}" if doi else ""
    if not url and source_id:
        url = f"https://europepmc.org/article/MED/{source_id}"
    return Paper(
        work_id=stable_work_id(
            doi=doi,
            source=default_source,
            source_id=source_id,
            title=title,
            year=year,
        ),
        title=title,
        abstract=strip_markup(result.get("abstractText")),
        authors=parse_authors(author_text.replace(", ", " and ")),
        source=default_source,
        publication_date=publication_date,
        year=year,
        doi=doi,
        source_id=source_id,
        url=url,
    )


def enrich_from_europe_pmc(papers: list[Paper], session: requests.Session) -> int:
    missing_by_doi = {paper.doi: paper for paper in papers if paper.doi and not paper.has_abstract}
    enriched = 0
    for doi_batch in batched(sorted(missing_by_doi), 20):
        query = " OR ".join(f'DOI:\"{doi}\"' for doi in doi_batch)
        payload = _safe_get_json(
            session,
            "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
            params={"query": query, "format": "json", "resultType": "core", "pageSize": 100},
        )
        for result in payload.get("resultList", {}).get("result", []):
            doi = normalize_doi(result.get("doi"))
            target = missing_by_doi.get(doi)
            abstract = strip_markup(result.get("abstractText"))
            if target and abstract and not target.has_abstract:
                target.abstract = abstract
                target.authors = target.authors or parse_authors(
                    collapse_whitespace(result.get("authorString")).replace(", ", " and ")
                )
                enriched += 1
    return enriched


def enrich_from_arxiv(papers: list[Paper], session: requests.Session) -> int:
    by_id: dict[str, Paper] = {}
    for paper in papers:
        if paper.has_abstract:
            continue
        arxiv_id = next(
            (
                normalize_arxiv_id(value)
                for value in (paper.source_id, paper.doi, paper.url)
                if normalize_arxiv_id(value)
            ),
            "",
        )
        if arxiv_id:
            by_id[arxiv_id] = paper
    enriched = 0
    atom = {"atom": "http://www.w3.org/2005/Atom"}
    for id_batch in batched(sorted(by_id), 20):
        response = session.get(
            "https://export.arxiv.org/api/query",
            params={"id_list": ",".join(id_batch), "max_results": len(id_batch)},
            timeout=60,
        )
        response.raise_for_status()
        root = ET.fromstring(response.content)
        for entry in root.findall("atom:entry", atom):
            entry_id = normalize_arxiv_id(entry.findtext("atom:id", "", atom))
            target = by_id.get(entry_id)
            abstract = strip_markup(entry.findtext("atom:summary", "", atom))
            if target and abstract and not target.has_abstract:
                target.abstract = abstract
                target.authors = target.authors or [
                    collapse_whitespace(node.findtext("atom:name", "", atom))
                    for node in entry.findall("atom:author", atom)
                ]
                enriched += 1
    return enriched


def enrich_from_preprint_apis(papers: list[Paper], session: requests.Session) -> int:
    enriched = 0
    preprints = [
        paper
        for paper in papers
        if not paper.has_abstract
        and (paper.doi.startswith("10.1101/") or paper.doi.startswith("10.64898/"))
    ]
    for paper in preprints:
        for server in ("biorxiv", "medrxiv"):
            try:
                payload = _safe_get_json(
                    session,
                    f"https://api.biorxiv.org/details/{server}/{quote(paper.doi, safe='/')}/na/json",
                    timeout=30,
                )
            except (requests.RequestException, OperationalError, ValueError):
                continue
            records = payload.get("collection", [])
            if records:
                abstract = strip_markup(records[-1].get("abstract"))
                if abstract:
                    paper.abstract = abstract
                    paper.source = server
                    enriched += 1
                    break
    return enriched


def enrich_from_crossref(papers: list[Paper], session: requests.Session) -> int:
    enriched = 0
    for paper in papers:
        if paper.has_abstract or not paper.doi:
            continue
        try:
            payload = _safe_get_json(
                session,
                f"https://api.crossref.org/works/{quote(paper.doi, safe='')}",
                params={"mailto": os.getenv("CROSSREF_MAILTO", "")},
                timeout=30,
            )
        except (requests.RequestException, OperationalError, ValueError):
            continue
        abstract = strip_markup(payload.get("message", {}).get("abstract"))
        if abstract:
            paper.abstract = abstract
            enriched += 1
    return enriched


def enrich_reference_abstracts(
    papers: list[Paper], session: requests.Session
) -> dict[str, int]:
    """Enrich in the documented fallback order, retaining title-only works."""

    counts = {"initial": sum(paper.has_abstract for paper in papers)}
    counts["europe_pmc_pubmed"] = enrich_from_europe_pmc(papers, session)
    counts["arxiv"] = enrich_from_arxiv(papers, session)
    counts["biorxiv_medrxiv"] = enrich_from_preprint_apis(papers, session)
    counts["crossref"] = enrich_from_crossref(papers, session)
    counts["final"] = sum(paper.has_abstract for paper in papers)
    counts["title_only"] = len(papers) - counts["final"]
    return counts


def retrieve_europe_pmc(
    session: requests.Session,
    *,
    start_date: date,
    end_date: date,
    max_records: int,
) -> list[Paper]:
    """Retrieve biomedical papers from Europe PMC with cursor pagination."""

    if max_records <= 0:
        return []
    query = (
        f"FIRST_PDATE:[{start_date.isoformat()} TO {end_date.isoformat()}] "
        "AND HAS_ABSTRACT:y"
    )
    papers: list[Paper] = []
    cursor = "*"
    while len(papers) < max_records:
        page_size = min(1000, max_records - len(papers))
        payload = _safe_get_json(
            session,
            "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
            params={
                "query": query,
                "format": "json",
                "resultType": "core",
                "pageSize": page_size,
                "cursorMark": cursor,
                "sort": "FIRST_PDATE_D desc",
            },
            timeout=60,
        )
        results = payload.get("resultList", {}).get("result", [])
        if not results:
            break
        for result in results:
            paper = _epmc_result_to_paper(result)
            if not paper.title:
                continue
            if paper.doi.startswith(("10.1101/", "10.64898/")):
                paper.source = "biorxiv/medrxiv"
            else:
                paper.source = "published"
            paper.work_id = stable_work_id(
                doi=paper.doi,
                source=paper.source,
                source_id=paper.source_id,
                title=paper.title,
                year=paper.year,
            )
            papers.append(paper)
            if len(papers) >= max_records:
                break
        next_cursor = collapse_whitespace(payload.get("nextCursorMark"))
        if not next_cursor or next_cursor == cursor:
            break
        cursor = next_cursor
    return papers


def _arxiv_entry_to_paper(entry: ET.Element, namespace: dict[str, str]) -> Paper:
    arxiv_id = normalize_arxiv_id(entry.findtext("atom:id", "", namespace))
    title = strip_markup(entry.findtext("atom:title", "", namespace))
    publication_date = collapse_whitespace(entry.findtext("atom:published", "", namespace))
    authors = [
        collapse_whitespace(author.findtext("atom:name", "", namespace))
        for author in entry.findall("atom:author", namespace)
    ]
    doi = normalize_doi(entry.findtext("arxiv:doi", "", namespace))
    return Paper(
        work_id=stable_work_id(
            doi=doi,
            arxiv_id=arxiv_id,
            source="arxiv",
            source_id=arxiv_id,
            title=title,
            year=parse_year(publication_date),
        ),
        title=title,
        abstract=strip_markup(entry.findtext("atom:summary", "", namespace)),
        authors=[author for author in authors if author],
        source="arxiv",
        publication_date=publication_date,
        year=parse_year(publication_date),
        doi=doi,
        source_id=arxiv_id,
        url=f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else "",
    )


def retrieve_arxiv(
    session: requests.Session,
    *,
    start_date: date,
    end_date: date,
    max_records: int,
) -> list[Paper]:
    """Retrieve selected arXiv categories from the official Atom API."""

    if max_records <= 0:
        return []
    category_query = " OR ".join(
        "cat:q-bio.*" if category == "q-bio" else f"cat:{category}"
        for category in ARXIV_CATEGORIES
    )
    start_stamp = start_date.strftime("%Y%m%d0000")
    end_stamp = end_date.strftime("%Y%m%d2359")
    query = f"({category_query}) AND submittedDate:[{start_stamp} TO {end_stamp}]"
    namespace = {
        "atom": "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom",
    }
    papers: list[Paper] = []
    offset = 0
    while len(papers) < max_records:
        page_size = min(100, max_records - len(papers))
        response = session.get(
            "https://export.arxiv.org/api/query",
            params={
                "search_query": query,
                "start": offset,
                "max_results": page_size,
                "sortBy": "submittedDate",
                "sortOrder": "descending",
            },
            timeout=60,
        )
        response.raise_for_status()
        try:
            root = ET.fromstring(response.content)
        except ET.ParseError as exc:
            raise OperationalError(f"Malformed arXiv Atom response: {exc}") from exc
        entries = root.findall("atom:entry", namespace)
        if not entries:
            break
        papers.extend(
            paper
            for paper in (_arxiv_entry_to_paper(entry, namespace) for entry in entries)
            if paper.title
        )
        if len(entries) < page_size:
            break
        offset += len(entries)
        if len(papers) < max_records:
            time.sleep(3)
    return papers[:max_records]


def retrieve_live_candidates(
    session: requests.Session,
    *,
    as_of: date,
    lookback_days: int,
    max_candidates: int,
) -> tuple[list[Paper], dict[str, int]]:
    if lookback_days < 1 or max_candidates < 1:
        raise OperationalError("lookback_days and max_candidates must be positive integers")
    start_date = as_of - timedelta(days=lookback_days)
    epmc = retrieve_europe_pmc(
        session,
        start_date=start_date,
        end_date=as_of,
        max_records=max_candidates,
    )
    arxiv = retrieve_arxiv(
        session,
        start_date=start_date,
        end_date=as_of,
        max_records=max_candidates,
    )
    combined = deduplicate_candidates(epmc + arxiv)
    combined.sort(
        key=lambda paper: (paper.publication_date, paper.work_id), reverse=True
    )
    candidates = combined[:max_candidates]
    if not candidates:
        raise OperationalError("Live retrieval returned no candidates")
    validate_unique_candidates(candidates)
    return candidates, {
        "europe_pmc_retrieved": len(epmc),
        "arxiv_retrieved": len(arxiv),
        "after_deduplication": len(combined),
        "selected": len(candidates),
    }


def _paper_source_group(paper: Paper) -> str:
    if paper.source == "arxiv" or normalize_arxiv_id(paper.source_id):
        return "arxiv"
    if paper.doi.startswith(("10.1101/", "10.64898/")):
        return "biorxiv/medrxiv"
    return "published"


def _deterministic_control_order(paper: Paper) -> tuple[str, str]:
    digest = hashlib.sha256(
        f"specter2-control-v{CONTROL_SNAPSHOT_VERSION}|{paper.work_id}".encode()
    ).hexdigest()
    return digest, paper.work_id


def _load_control_snapshot(path: Path) -> list[Paper] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("version") != CONTROL_SNAPSHOT_VERSION:
            return None
        return [Paper.from_dict(item) for item in payload.get("papers", [])]
    except (OSError, ValueError, TypeError):
        return None


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def build_background_controls(
    holdouts: Sequence[Paper],
    session: requests.Session,
    *,
    snapshot_path: Path,
    rebuild: bool,
) -> tuple[list[Paper], bool]:
    """Load or create the deterministic 3:1 backtest control snapshot."""

    target = len(holdouts) * 3
    if target == 0:
        raise OperationalError("Backtest contains no 2025-2026 bibliography holdouts")
    excluded_dois = {paper.doi for paper in holdouts if paper.doi}
    excluded_titles = {normalize_title(paper.title) for paper in holdouts}

    cached = None if rebuild else _load_control_snapshot(snapshot_path)
    if cached is not None:
        usable = [
            replace(paper, label=0)
            for paper in cached
            if paper.doi not in excluded_dois
            and normalize_title(paper.title) not in excluded_titles
        ]
        if len(usable) >= target:
            return usable[:target], True

    # Pull an oversized deterministic pool once; Actions preserves the selected
    # records in the reference cache so subsequent backtests use the same set.
    pool_limit = max(1200, target * 5)
    epmc_pool = retrieve_europe_pmc(
        session,
        start_date=date(2025, 1, 1),
        end_date=date(2026, 12, 31),
        max_records=pool_limit,
    )
    arxiv_pool = retrieve_arxiv(
        session,
        start_date=date(2025, 1, 1),
        end_date=date(2026, 12, 31),
        max_records=min(pool_limit, 1200),
    )
    pool = deduplicate_candidates(epmc_pool + arxiv_pool)
    pool = [
        paper
        for paper in pool
        if paper.doi not in excluded_dois
        and normalize_title(paper.title) not in excluded_titles
        and paper.year in {2025, 2026}
    ]
    by_source: dict[str, list[Paper]] = {
        "published": [],
        "arxiv": [],
        "biorxiv/medrxiv": [],
    }
    for paper in pool:
        by_source[_paper_source_group(paper)].append(paper)
    for values in by_source.values():
        values.sort(key=_deterministic_control_order)

    desired: dict[str, int] = {key: 0 for key in by_source}
    for holdout in holdouts:
        desired[_paper_source_group(holdout)] += 3
    selected: list[Paper] = []
    selected_ids: set[str] = set()
    for group, quota in desired.items():
        for paper in by_source[group][:quota]:
            selected.append(replace(paper, label=0))
            selected_ids.add(paper.work_id)
    if len(selected) < target:
        remainder = sorted(
            (paper for paper in pool if paper.work_id not in selected_ids),
            key=_deterministic_control_order,
        )
        selected.extend(replace(paper, label=0) for paper in remainder[: target - len(selected)])
    if len(selected) < target:
        raise OperationalError(
            f"Could only create {len(selected)} of {target} required background controls"
        )
    selected = selected[:target]
    _write_json_atomic(
        snapshot_path,
        {
            "version": CONTROL_SNAPSHOT_VERSION,
            "bibliography_commit": SKM_COMMIT,
            "meaning": "not present in the bibliography; not guaranteed irrelevant",
            "papers": [paper.to_dict() for paper in selected],
        },
    )
    return selected, False


def prepare_backtest_candidates(
    bibliography: Sequence[Paper],
    session: requests.Session,
    *,
    snapshot_path: Path,
    rebuild: bool,
) -> tuple[list[Paper], list[Paper], dict[str, Any]]:
    references = [paper for paper in bibliography if paper.year and paper.year <= 2024]
    holdouts = [
        replace(paper, label=1)
        for paper in bibliography
        if paper.year in {2025, 2026}
    ]
    if not references:
        raise OperationalError("Backtest reference split is empty")
    controls, control_cache_hit = build_background_controls(
        holdouts,
        session,
        snapshot_path=snapshot_path,
        rebuild=rebuild,
    )
    candidates = deduplicate_candidates(holdouts + controls)
    # merge_duplicate_works chooses rich metadata; restore labels explicitly.
    positive_dois = {paper.doi for paper in holdouts if paper.doi}
    positive_titles = {normalize_title(paper.title) for paper in holdouts}
    for paper in candidates:
        paper.label = int(
            (paper.doi and paper.doi in positive_dois)
            or normalize_title(paper.title) in positive_titles
        )
    validate_unique_candidates(candidates)
    return references, candidates, {
        "reference_count": len(references),
        "positive_holdouts": sum(paper.label == 1 for paper in candidates),
        "background_controls": sum(paper.label == 0 for paper in candidates),
        "control_ratio": 3,
        "control_cache_hit": control_cache_hit,
        "control_interpretation": "not in bibliography; not guaranteed irrelevant",
    }


def validate_unique_candidates(papers: Sequence[Paper]) -> None:
    seen: set[str] = set()
    for paper in papers:
        if not paper.work_id or not paper.title:
            raise OperationalError("Candidate metadata is malformed")
        if paper.work_id in seen:
            raise OperationalError(f"Duplicate candidate ID after normalization: {paper.work_id}")
        seen.add(paper.work_id)


class Specter2Embedder:
    """CPU SPECTER2 proximity encoder with immutable Hub revisions."""

    def __init__(self, batch_size: int = 8):
        self.batch_size = batch_size
        try:
            import torch
            from adapters import AutoAdapterModel
            from transformers import AutoTokenizer
        except ImportError as exc:  # pragma: no cover - dependency error path
            raise OperationalError(
                "SPECTER2 dependencies are missing; install requirements-specter.txt"
            ) from exc
        self.torch = torch
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                SPECTER2_BASE,
                revision=SPECTER2_BASE_REVISION,
            )
            self.model = AutoAdapterModel.from_pretrained(
                SPECTER2_BASE,
                revision=SPECTER2_BASE_REVISION,
            )
            adapter_name = self.model.load_adapter(
                SPECTER2_ADAPTER,
                source="hf",
                revision=SPECTER2_ADAPTER_REVISION,
                load_as="proximity",
                set_active=True,
            )
            # adapters 1.3.0 emits a warning while transitioning activation
            # state; set and verify the returned name explicitly before any
            # forward pass so a base-only embedding can never pass silently.
            self.model.set_active_adapters(adapter_name)
            if adapter_name not in str(self.model.active_adapters):
                raise OperationalError("The SPECTER2 proximity adapter is not active")
            self.active_adapter = adapter_name
        except Exception as exc:
            raise OperationalError(f"Could not load pinned SPECTER2 model: {exc}") from exc
        self.model.to("cpu")
        self.model.eval()

    def encode(self, papers: Sequence[Paper]) -> np.ndarray:
        if not papers:
            raise OperationalError("Cannot embed an empty paper collection")
        chunks: list[np.ndarray] = []
        separator = self.tokenizer.sep_token or "[SEP]"
        for batch in batched(list(papers), self.batch_size):
            texts = [
                f"{paper.title}{separator}{paper.abstract or ''}"
                for paper in batch
            ]
            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
                return_token_type_ids=False,
                max_length=512,
            )
            inputs = {key: value.to("cpu") for key, value in inputs.items()}
            with self.torch.inference_mode():
                output = self.model(**inputs)
            chunks.append(output.last_hidden_state[:, 0, :].detach().cpu().float().numpy())
        return l2_normalize(np.concatenate(chunks, axis=0))


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise OperationalError(f"Embedding matrix has invalid shape {array.shape}")
    if not np.isfinite(array).all():
        raise OperationalError("Embedding matrix contains non-finite values")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms <= 0) or not np.isfinite(norms).all():
        raise OperationalError("Embedding matrix contains zero or invalid norms")
    normalized = array / norms
    if not np.isfinite(normalized).all():
        raise OperationalError("Normalized embeddings contain non-finite values")
    return normalized


def _reference_cache_root(cache_dir: Path) -> Path:
    model_key = f"{SPECTER2_BASE_REVISION[:12]}-{SPECTER2_ADAPTER_REVISION[:12]}"
    return cache_dir / f"skm-{SKM_COMMIT}" / f"specter2-{model_key}"


def load_or_build_reference_corpus(
    *,
    cache_dir: Path,
    session: requests.Session,
    rebuild: bool,
    bibliography_path: Path | None = None,
) -> tuple[list[Paper], dict[str, Any], bool]:
    root = _reference_cache_root(cache_dir)
    corpus_path = root / "reference_corpus.json"
    if corpus_path.exists() and not rebuild:
        try:
            payload = json.loads(corpus_path.read_text(encoding="utf-8"))
            if (
                payload.get("schema_version") == SCHEMA_VERSION
                and payload.get("bibliography_commit") == SKM_COMMIT
            ):
                papers = [Paper.from_dict(item) for item in payload.get("papers", [])]
                if not papers:
                    raise OperationalError("Cached reference corpus is empty")
                return papers, payload.get("coverage", {}), True
        except (OSError, ValueError, TypeError) as exc:
            raise OperationalError(f"Malformed reference corpus cache: {exc}") from exc

    try:
        if bibliography_path:
            bibliography_text = bibliography_path.read_text(encoding="utf-8")
        else:
            response = session.get(SKM_BIB_URL, timeout=60)
            response.raise_for_status()
            bibliography_text = response.text
    except (OSError, requests.RequestException) as exc:
        raise OperationalError(f"Could not obtain pinned SKM bibliography: {exc}") from exc
    parsed = parse_bibliography(bibliography_text)
    papers = merge_duplicate_works(parsed)
    if not papers:
        raise OperationalError("Reference corpus is empty after duplicate merging")
    enrichment = enrich_reference_abstracts(papers, session)
    coverage = {
        "raw_bibliography_entries": len(parsed),
        "distinct_works": len(papers),
        "duplicates_merged": len(parsed) - len(papers),
        "abstract_count": enrichment["final"],
        "title_only_count": enrichment["title_only"],
        "abstract_fraction": round(enrichment["final"] / len(papers), 6),
        "enrichment": enrichment,
        "bibliography_content_sha256": hashlib.sha256(
            bibliography_text.encode("utf-8")
        ).hexdigest(),
    }
    _write_json_atomic(
        corpus_path,
        {
            "schema_version": SCHEMA_VERSION,
            "bibliography_commit": SKM_COMMIT,
            "bibliography_url": SKM_BIB_URL,
            "coverage": coverage,
            "papers": [paper.to_dict() for paper in papers],
        },
    )
    return papers, coverage, False


def load_or_build_reference_embeddings(
    papers: Sequence[Paper],
    *,
    cache_dir: Path,
    embedder: Specter2Embedder,
    rebuild: bool,
) -> tuple[np.ndarray, bool]:
    root = _reference_cache_root(cache_dir)
    cache_path = root / "reference_embeddings.npz"
    expected_ids = [paper.work_id for paper in papers]
    if cache_path.exists() and not rebuild:
        try:
            with np.load(cache_path, allow_pickle=False) as payload:
                ids = [str(value) for value in payload["work_ids"].tolist()]
                embeddings = np.asarray(payload["embeddings"], dtype=np.float32)
            if ids == expected_ids and embeddings.shape[0] == len(papers):
                normalized = l2_normalize(embeddings)
                return normalized, True
        except (OSError, ValueError, KeyError) as exc:
            raise OperationalError(f"Malformed reference embedding cache: {exc}") from exc
    embeddings = embedder.encode(papers)
    root.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        work_ids=np.asarray(expected_ids, dtype=str),
        embeddings=embeddings,
    )
    temporary.replace(cache_path)
    return embeddings, False


def _stable_similarity_order(values: np.ndarray, references: Sequence[Paper]) -> list[int]:
    return sorted(
        range(len(references)),
        key=lambda index: (-float(values[index]), references[index].work_id),
    )


def calculate_scores(
    candidate_embeddings: np.ndarray,
    reference_embeddings: np.ndarray,
    references: Sequence[Paper],
) -> ScoreSet:
    candidates = l2_normalize(candidate_embeddings)
    refs = l2_normalize(reference_embeddings)
    if refs.shape[0] != len(references):
        raise OperationalError("Reference metadata and embedding counts differ")
    if candidates.shape[1] != refs.shape[1]:
        raise OperationalError("Candidate and reference embedding dimensions differ")
    similarities = candidates @ refs.T
    if not np.isfinite(similarities).all():
        raise OperationalError("Cosine similarity matrix contains non-finite values")

    top5: list[float] = []
    top1: list[float] = []
    top10: list[float] = []
    neighbors: list[list[Neighbor]] = []
    for row in similarities:
        order = _stable_similarity_order(row, references)
        top5.append(float(np.mean([row[index] for index in order[: min(5, len(order))]])))
        top1.append(float(row[order[0]]))
        top10.append(float(np.mean([row[index] for index in order[: min(10, len(order))]])))
        neighbors.append(
            [
                Neighbor(
                    work_id=references[index].work_id,
                    title=references[index].title,
                    identifier=references[index].identifier(),
                    similarity=float(row[index]),
                )
                for index in order[:3]
            ]
        )

    centroid_vector = np.mean(refs, axis=0, keepdims=True)
    centroid_vector = l2_normalize(centroid_vector)[0]
    centroid = candidates @ centroid_vector
    return ScoreSet(
        top5_mean=np.asarray(top5, dtype=np.float32),
        centroid=np.asarray(centroid, dtype=np.float32),
        top1=np.asarray(top1, dtype=np.float32),
        top10_mean=np.asarray(top10, dtype=np.float32),
        neighbors=neighbors,
    )


def _rank_indices(scores: Sequence[float], candidates: Sequence[Paper]) -> list[int]:
    if len(scores) != len(candidates):
        raise OperationalError("Candidate and score counts differ")
    if not all(math.isfinite(float(score)) for score in scores):
        raise OperationalError("Ranking contains non-finite scores")
    return sorted(
        range(len(candidates)),
        key=lambda index: (-float(scores[index]), candidates[index].work_id),
    )


def report_similarity(value: float) -> float:
    """Canonicalize insignificant CPU-kernel drift in report-facing scores."""

    return round(float(value), REPORT_SIMILARITY_DECIMALS)


def build_ranked_results(
    candidates: Sequence[Paper], score_set: ScoreSet
) -> list[dict[str, Any]]:
    order = _rank_indices(score_set.top5_mean, candidates)
    results: list[dict[str, Any]] = []
    for rank, index in enumerate(order, start=1):
        paper = candidates[index]
        results.append(
            {
                "rank": rank,
                "score": report_similarity(score_set.top5_mean[index]),
                "title": paper.title,
                "abstract_available": paper.has_abstract,
                "authors": paper.authors,
                "source": paper.source,
                "publication_date": paper.publication_date,
                "doi": paper.doi,
                "source_id": paper.source_id,
                "work_id": paper.work_id,
                "url": paper.url,
                "label": paper.label,
                "diagnostic_scores": {
                    "centroid": report_similarity(score_set.centroid[index]),
                    "single_nearest": report_similarity(score_set.top1[index]),
                    "mean_top_ten": report_similarity(score_set.top10_mean[index]),
                },
                "nearest_references": [
                    {
                        **asdict(neighbor),
                        "similarity": report_similarity(neighbor.similarity),
                    }
                    for neighbor in score_set.neighbors[index]
                ],
            }
        )
    validate_sorted_results(results)
    return results


def validate_sorted_results(results: Sequence[dict[str, Any]]) -> None:
    expected_ranks = list(range(1, len(results) + 1))
    if [result.get("rank") for result in results] != expected_ranks:
        raise OperationalError("Output ranks are malformed")
    scores = [float(result["score"]) for result in results]
    if any(left < right for left, right in zip(scores, scores[1:])):
        raise OperationalError("Output is not deterministically sorted")


def _metrics_for_scores(
    scores: Sequence[float], candidates: Sequence[Paper]
) -> dict[str, float | int | None]:
    order = _rank_indices(scores, candidates)
    labels = [int(candidates[index].label or 0) for index in order]
    positives = sum(labels)
    controls = len(labels) - positives
    if positives == 0:
        return {
            "positive_count": 0,
            "control_count": controls,
            "recall_at_10": None,
            "recall_at_25": None,
            "recall_at_50": None,
            "mean_reciprocal_rank": None,
            "average_precision": None,
            "median_positive_rank": None,
            "median_control_rank": statistics.median(range(1, len(labels) + 1)) if labels else None,
        }
    positive_ranks = [rank for rank, label in enumerate(labels, 1) if label]
    control_ranks = [rank for rank, label in enumerate(labels, 1) if not label]
    precisions = []
    positives_seen = 0
    for rank, label in enumerate(labels, 1):
        if label:
            positives_seen += 1
            precisions.append(positives_seen / rank)
    return {
        "positive_count": positives,
        "control_count": controls,
        "recall_at_10": sum(labels[:10]) / positives,
        "recall_at_25": sum(labels[:25]) / positives,
        "recall_at_50": sum(labels[:50]) / positives,
        "mean_reciprocal_rank": 1.0 / positive_ranks[0],
        "average_precision": sum(precisions) / positives,
        "median_positive_rank": float(statistics.median(positive_ranks)),
        "median_control_rank": float(statistics.median(control_ranks)) if control_ranks else None,
    }


def calculate_backtest_metrics(
    score_set: ScoreSet, candidates: Sequence[Paper]
) -> dict[str, Any]:
    if any(paper.label not in {0, 1} for paper in candidates):
        raise OperationalError("Backtest candidates require binary labels")
    variants = {
        "mean_top_five": score_set.top5_mean,
        "global_centroid": score_set.centroid,
        "single_nearest": score_set.top1,
        "mean_top_ten": score_set.top10_mean,
    }
    metrics = {
        name: _metrics_for_scores(scores, candidates)
        for name, scores in variants.items()
    }
    primary = metrics["mean_top_five"]
    background_rate = sum(int(paper.label == 1) for paper in candidates) / len(candidates)
    average_precision = primary["average_precision"]
    median_positive = primary["median_positive_rank"]
    median_control = primary["median_control_rank"]
    return {
        "background_positive_rate": background_rate,
        "variants": metrics,
        "quality_checks": {
            "average_precision_exceeds_background_rate": bool(
                average_precision is not None and average_precision > background_rate
            ),
            "median_positive_rank_beats_control_rank": bool(
                median_positive is not None
                and median_control is not None
                and median_positive < median_control
            ),
            "informational_only": True,
        },
    }


RESULT_REQUIRED_FIELDS = {
    "rank",
    "score",
    "title",
    "abstract_available",
    "authors",
    "source",
    "publication_date",
    "doi",
    "source_id",
    "work_id",
    "url",
    "nearest_references",
}


def validate_report_schema(report: dict[str, Any]) -> None:
    for key in ("schema_version", "run", "model", "bibliography", "statistics", "results"):
        if key not in report:
            raise OperationalError(f"Report is missing required field: {key}")
    if report["schema_version"] != SCHEMA_VERSION:
        raise OperationalError("Unexpected report schema version")
    for result in report["results"]:
        missing = RESULT_REQUIRED_FIELDS - set(result)
        if missing:
            raise OperationalError(f"Result is missing fields: {sorted(missing)}")
        if len(result["nearest_references"]) != 3:
            raise OperationalError("Every result must contain three reference explanations")
    validate_sorted_results(report["results"])


def _package_versions() -> dict[str, str]:
    from importlib import metadata

    versions: dict[str, str] = {}
    for package in ("python", "torch", "adapters", "transformers", "numpy", "bibtexparser", "requests"):
        if package == "python":
            versions[package] = sys.version.split()[0]
            continue
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def _markdown_cell(value: Any) -> str:
    return collapse_whitespace(value).replace("|", "\\|")


def _format_duration(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    seconds = float(value)
    if seconds < 60:
        return f"{seconds:.1f}s"
    return f"{seconds / 60:.1f}m"


def _result_table(results: Sequence[dict[str, Any]]) -> list[str]:
    lines = [
        "| Rank | Score | Paper | Source/date | Nearest references |",
        "| ---: | ---: | --- | --- | --- |",
    ]
    for result in results:
        title = _markdown_cell(result["title"])
        if result.get("url"):
            title = f"[{title}]({result['url']})"
        nearest = "<br>".join(
            f"{_markdown_cell(item['title'])} ({float(item['similarity']):.3f})"
            for item in result["nearest_references"]
        )
        source_date = _markdown_cell(
            f"{result['source']} {result['publication_date']}"
        )
        lines.append(
            f"| {result['rank']} | {float(result['score']):.4f} | {title} | "
            f"{source_date} | {nearest} |"
        )
    return lines


def render_markdown_report(report: dict[str, Any], top_n: int) -> str:
    run = report["run"]
    bibliography = report["bibliography"]
    model = report["model"]
    stats = report["statistics"]
    cache = report["cache"]
    timings = report["timings_seconds"]
    results = report["results"]
    lines = [
        "# SPECTER2 paper-ranking evaluation",
        "",
        f"- Mode: `{run['mode']}`",
        f"- Bibliography: `delalamo/SKM@{bibliography['commit']}`",
        f"- Bibliography content SHA-256: `{bibliography['content_sha256']}`",
        f"- Base model: `{model['base']}@{model['base_revision']}`",
        f"- Proximity adapter: `{model['adapter']}@{model['adapter_revision']}`",
        f"- Primary score: mean cosine similarity to five nearest distinct references",
        f"- Candidates ranked: {stats['candidate_count']}",
        f"- References used: {stats['reference_count']}",
        f"- Reference abstract coverage: {stats['reference_abstract_count']}/"
        f"{stats['reference_corpus_count']} ({stats['reference_abstract_fraction']:.1%})",
        f"- Cache class: `{cache['runtime_class']}` "
        f"(corpus={cache['reference_corpus_hit']}, embeddings={cache['reference_embeddings_hit']})",
        f"- Total runtime: {_format_duration(timings.get('total'))}",
        "",
        "Scores are relative ranking values, not probabilities. No relevance cutoff is applied.",
        "",
    ]
    if report.get("metrics"):
        metrics = report["metrics"]
        lines.extend(
            [
                "## Backtest metrics (informational only)",
                "",
                f"Background positive rate: {metrics['background_positive_rate']:.4f}",
                "",
                "| Variant | AP | MRR | R@10 | R@25 | R@50 | Median positive | Median control |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for name, values in metrics["variants"].items():
            def metric(key: str) -> str:
                value = values[key]
                return "n/a" if value is None else f"{float(value):.4f}"

            lines.append(
                f"| {name} | {metric('average_precision')} | "
                f"{metric('mean_reciprocal_rank')} | {metric('recall_at_10')} | "
                f"{metric('recall_at_25')} | {metric('recall_at_50')} | "
                f"{metric('median_positive_rank')} | {metric('median_control_rank')} |"
            )
        checks = metrics["quality_checks"]
        lines.extend(
            [
                "",
                "Quality checks do not fail this evaluation workflow:",
                "",
                f"- AP exceeds the background rate: `{checks['average_precision_exceeds_background_rate']}`",
                f"- Median positive rank beats median control rank: "
                f"`{checks['median_positive_rank_beats_control_rank']}`",
                "",
            ]
        )
    lines.extend([f"## Top {min(top_n, len(results))}", ""])
    lines.extend(_result_table(results[:top_n]))
    if run["mode"] == "backtest" and results:
        lines.extend(["", "## Bottom 25 (manual-review aid)", ""])
        lines.extend(_result_table(results[-25:]))
    lines.extend(
        [
            "",
            "## Run details",
            "",
            "```json",
            json.dumps(
                {
                    "configuration": run,
                    "candidate_retrieval": report.get("candidate_retrieval", {}),
                    "reference_coverage": bibliography["coverage"],
                    "cache": cache,
                    "timings_seconds": timings,
                    "versions": report["versions"],
                },
                indent=2,
                sort_keys=True,
            ),
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def _csv_row(result: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "rank": result["rank"],
        "score": result["score"],
        "title": result["title"],
        "abstract_available": result["abstract_available"],
        "authors": "; ".join(result["authors"]),
        "source": result["source"],
        "publication_date": result["publication_date"],
        "doi": result["doi"],
        "source_id": result["source_id"],
        "work_id": result["work_id"],
        "url": result["url"],
        "label": result["label"],
        "centroid_score": result["diagnostic_scores"]["centroid"],
        "single_nearest_score": result["diagnostic_scores"]["single_nearest"],
        "mean_top_ten_score": result["diagnostic_scores"]["mean_top_ten"],
        "mode": report["run"]["mode"],
        "bibliography_commit": report["bibliography"]["commit"],
        "bibliography_content_sha256": report["bibliography"]["content_sha256"],
        "model_base_revision": report["model"]["base_revision"],
        "model_adapter_revision": report["model"]["adapter_revision"],
        "reference_abstract_fraction": report["statistics"]["reference_abstract_fraction"],
        "reference_abstract_count": report["statistics"]["reference_abstract_count"],
        "reference_count": report["statistics"]["reference_count"],
        "candidate_count": report["statistics"]["candidate_count"],
        "total_runtime_seconds": report["timings_seconds"].get("total"),
        "cache_runtime_class": report["cache"]["runtime_class"],
        "reference_corpus_cache_hit": report["cache"]["reference_corpus_hit"],
        "reference_embeddings_cache_hit": report["cache"]["reference_embeddings_hit"],
        "actions_cache_hit": report["cache"].get("actions_cache_hit", "unknown"),
    }
    for number, neighbor in enumerate(result["nearest_references"], start=1):
        row[f"nearest_{number}_title"] = neighbor["title"]
        row[f"nearest_{number}_identifier"] = neighbor["identifier"]
        row[f"nearest_{number}_similarity"] = neighbor["similarity"]
    return row


def write_reports(report: dict[str, Any], output_dir: Path, top_n: int) -> None:
    validate_report_schema(report)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(output_dir / "specter2_results.json", report)
    markdown = render_markdown_report(report, top_n)
    (output_dir / "specter2_results.md").write_text(markdown, encoding="utf-8")
    (output_dir / "summary.md").write_text(markdown, encoding="utf-8")
    rows = [_csv_row(result, report) for result in report["results"]]
    if not rows:
        raise OperationalError("Cannot write an empty result report")
    with (output_dir / "specter2_results.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _subset_embeddings(
    all_papers: Sequence[Paper],
    all_embeddings: np.ndarray,
    selected_papers: Sequence[Paper],
) -> np.ndarray:
    mapping = {paper.work_id: index for index, paper in enumerate(all_papers)}
    try:
        return np.asarray(
            [all_embeddings[mapping[paper.work_id]] for paper in selected_papers],
            dtype=np.float32,
        )
    except KeyError as exc:
        raise OperationalError(f"Reference embedding cache is missing {exc.args[0]}") from exc


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    if args.lookback_days < 1 or args.max_candidates < 1 or args.top_n < 1:
        raise OperationalError("lookback_days, max_candidates, and top_n must be positive")
    started = time.perf_counter()
    timings: dict[str, float] = {}
    cache_dir = Path(args.cache_dir)
    output_dir = Path(args.output_dir)
    session = build_http_session()

    phase = time.perf_counter()
    bibliography, coverage, corpus_cache_hit = load_or_build_reference_corpus(
        cache_dir=cache_dir,
        session=session,
        rebuild=args.rebuild_reference_cache,
        bibliography_path=Path(args.bibliography_path) if args.bibliography_path else None,
    )
    timings["reference_metadata"] = time.perf_counter() - phase

    as_of = date.fromisoformat(args.as_of_date) if args.as_of_date else datetime.now(timezone.utc).date()
    phase = time.perf_counter()
    if args.mode == "backtest":
        references, candidates, candidate_stats = prepare_backtest_candidates(
            bibliography,
            session,
            snapshot_path=_reference_cache_root(cache_dir) / "background_controls.json",
            rebuild=args.rebuild_reference_cache,
        )
    else:
        references = list(bibliography)
        candidates, candidate_stats = retrieve_live_candidates(
            session,
            as_of=as_of,
            lookback_days=args.lookback_days,
            max_candidates=args.max_candidates,
        )
    timings["candidate_retrieval"] = time.perf_counter() - phase
    if len(references) < 10:
        raise OperationalError("Reference corpus has fewer than ten distinct works")

    phase = time.perf_counter()
    embedder = Specter2Embedder(batch_size=args.batch_size)
    timings["model_loading"] = time.perf_counter() - phase

    phase = time.perf_counter()
    all_reference_embeddings, embedding_cache_hit = load_or_build_reference_embeddings(
        bibliography,
        cache_dir=cache_dir,
        embedder=embedder,
        rebuild=args.rebuild_reference_cache,
    )
    reference_embeddings = _subset_embeddings(
        bibliography, all_reference_embeddings, references
    )
    timings["reference_embedding"] = time.perf_counter() - phase

    phase = time.perf_counter()
    candidate_embeddings = embedder.encode(candidates)
    timings["candidate_embedding"] = time.perf_counter() - phase

    phase = time.perf_counter()
    score_set = calculate_scores(candidate_embeddings, reference_embeddings, references)
    results = build_ranked_results(candidates, score_set)
    metrics = (
        calculate_backtest_metrics(score_set, candidates)
        if args.mode == "backtest"
        else None
    )
    timings["ranking"] = time.perf_counter() - phase
    timings["total"] = time.perf_counter() - started

    reference_abstract_count = sum(paper.has_abstract for paper in references)
    runtime_class = "warm" if corpus_cache_hit and embedding_cache_hit else "cold-or-partial"
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run": {
            "mode": args.mode,
            "lookback_days": args.lookback_days,
            "max_candidates": args.max_candidates,
            "top_n": args.top_n,
            "as_of_date": as_of.isoformat(),
            "rebuild_reference_cache": args.rebuild_reference_cache,
            "automatic_relevance_cutoff": None,
        },
        "model": {
            "base": SPECTER2_BASE,
            "base_revision": SPECTER2_BASE_REVISION,
            "adapter": SPECTER2_ADAPTER,
            "adapter_revision": SPECTER2_ADAPTER_REVISION,
            "representation": "title + [SEP] + abstract; first token; max 512 tokens; L2 normalized",
            "report_similarity_decimals": REPORT_SIMILARITY_DECIMALS,
            "device": "cpu",
        },
        "bibliography": {
            "repository": "delalamo/SKM",
            "commit": SKM_COMMIT,
            "url": SKM_BIB_URL,
            "content_sha256": coverage.get("bibliography_content_sha256", "unknown"),
            "coverage": coverage,
        },
        "statistics": {
            "reference_corpus_count": len(bibliography),
            "reference_count": len(references),
            "reference_abstract_count": reference_abstract_count,
            "reference_abstract_fraction": reference_abstract_count / len(references),
            "candidate_count": len(candidates),
            "candidate_abstract_count": sum(paper.has_abstract for paper in candidates),
        },
        "candidate_retrieval": candidate_stats,
        "cache": {
            "actions_cache_hit": args.actions_cache_hit,
            "reference_corpus_hit": corpus_cache_hit,
            "reference_embeddings_hit": embedding_cache_hit,
            "runtime_class": runtime_class,
            "cache_key_components": {
                "bibliography_commit": SKM_COMMIT,
                "base_revision": SPECTER2_BASE_REVISION,
                "adapter_revision": SPECTER2_ADAPTER_REVISION,
            },
        },
        "timings_seconds": {key: round(value, 6) for key, value in timings.items()},
        "versions": _package_versions(),
        "metrics": metrics,
        "results": results,
    }
    write_reports(report, output_dir, args.top_n)
    return report


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("backtest", "live"), default="backtest")
    parser.add_argument("--lookback-days", type=int, default=7)
    parser.add_argument("--max-candidates", type=int, default=300)
    parser.add_argument("--top-n", type=int, default=25)
    parser.add_argument("--cache-dir", default=".cache/specter2")
    parser.add_argument("--output-dir", default="specter2-reports")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--rebuild-reference-cache", action="store_true")
    parser.add_argument("--actions-cache-hit", default="unknown")
    parser.add_argument(
        "--as-of-date",
        default="",
        help="UTC YYYY-MM-DD override, primarily for reproducible live diagnostics",
    )
    parser.add_argument(
        "--bibliography-path",
        default="",
        help="Local test override; production workflow always uses the pinned SKM URL",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    try:
        run_evaluation(args)
    except (OperationalError, requests.RequestException, ValueError, ET.ParseError) as exc:
        print(f"SPECTER2 evaluation failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
