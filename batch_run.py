"""Batch API variant of the PubMedFetcher pipeline.

Instead of classifying papers via real-time API calls, this script uses the
OpenAI Batch API to submit all classification requests in a single .jsonl file.
The Batch API processes requests asynchronously (typically within 24 hours) at
50% lower cost and with separate, higher rate limits.

Usage:
    # Phase 1 – scrape papers and submit the batch
    python batch_run.py submit [--n-days 1] [--test-mode]

    # Phase 2 – poll for completion, process results, send digest
    python batch_run.py collect --batch-id <BATCH_ID> [--test-mode]

The GitHub Action workflow calls these two phases with a polling loop in between.
"""

import argparse
import asyncio
import contextlib
import json
import os
import re
import smtplib
import sys
import tempfile
import time
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import markdown
import requests
from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI

from run import (
    deduplicate_papers,
    ensure_github_label,
    create_github_issue,
    load_keywords,
    openai_summary_prompt,
    scrape_all_sources,
    summarize_papers,
    _build_keyword_patterns,
    keyword_prefilter,
)

load_dotenv()

CANDIDATES_FILE = "batch_candidates.json"


def _classification_messages(title: str, abstract: str, keywords: list[str]) -> list[dict]:
    """Build the chat messages for a single classification request."""
    keywords_str = "\n".join(f"- {kw}" for kw in keywords)
    return [
        {
            "role": "system",
            "content": (
                "You are a scientific paper classifier. Given a paper's title and abstract, "
                "and a list of research topics/keywords, determine if the paper is relevant to "
                "any of the listed topics. A paper is relevant if its primary focus or a major "
                "contribution relates to one or more of the topics. Respond with ONLY the word "
                "'relevant' or 'not relevant'. Do not include any other text."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Research topics/keywords:\n{keywords_str}\n\n"
                f"Paper title: {title}\n\n"
                f"Paper abstract: {abstract}"
            ),
        },
    ]


def submit(n_days: int, test_mode: bool = False) -> str:
    """Scrape papers, build a Batch API .jsonl file, and submit it.

    Args:
        n_days: Number of days to look back when scraping.
        test_mode: If True, limit candidates to the first 50.

    Returns:
        The batch ID string from the OpenAI API.
    """
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    keywords = load_keywords()
    print(f"Loaded {len(keywords)} keywords")

    # --- Scrape ---
    data_pubmed, data_biorxiv, data_arxiv, biorxiv_errors = scrape_all_sources(n_days)

    data: dict[str, list] = {
        "Title": [], "Abstract": [], "Journal": [], "Link": [], "Authors": []
    }
    for d in [data_pubmed, data_biorxiv, data_arxiv]:
        for field in data:
            data[field].extend(d[field])

    before_dedup = len(data["Title"])
    data = deduplicate_papers(data)
    after_dedup = len(data["Title"])
    print(f"Deduplicated: {before_dedup} -> {after_dedup} papers")

    # --- Pre-filter ---
    keyword_patterns = _build_keyword_patterns(keywords)
    candidates: list[dict] = []
    links = data.get("Link", [""] * len(data["Title"]))
    authors_list = data.get("Authors", [""] * len(data["Title"]))
    for title, abstract, journal, link, authors in zip(
        data["Title"], data["Abstract"], data["Journal"], links, authors_list
    ):
        if len(abstract.split()) < 100:
            continue
        if not keyword_prefilter(title, abstract, keyword_patterns):
            continue
        candidates.append(
            {
                "Title": title,
                "Abstract": abstract,
                "Journal": journal,
                "Link": link,
                "Authors": authors,
            }
        )
    print(f"Pre-filter: {len(candidates)} candidates for classification")

    if test_mode:
        candidates = candidates[:50]
        print(f"Test mode: trimmed to {len(candidates)} candidates")

    if not candidates:
        print("No candidates to classify. Exiting.")
        sys.exit(0)

    # Save candidates so the collect phase can look them up by custom_id
    with open(CANDIDATES_FILE, "w") as f:
        json.dump(candidates, f)

    # Save scraping metadata for the email digest
    metadata = {
        "n_arxiv": len(data_arxiv["Title"]),
        "n_pubmed": len(data_pubmed["Title"]),
        "n_biorxiv": len(data_biorxiv["Title"]),
        "biorxiv_errors": biorxiv_errors,
    }
    with open("batch_metadata.json", "w") as f:
        json.dump(metadata, f)

    # --- Build JSONL ---
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, dir="."
    ) as tmp:
        for idx, paper in enumerate(candidates):
            request_obj = {
                "custom_id": f"paper-{idx}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-5-nano",
                    "messages": _classification_messages(
                        paper["Title"], paper["Abstract"], keywords
                    ),
                    "max_tokens": 10,
                },
            }
            tmp.write(json.dumps(request_obj) + "\n")
        jsonl_path = tmp.name

    # --- Upload & Submit ---
    print(f"Uploading {jsonl_path} ({len(candidates)} requests)...")
    with open(jsonl_path, "rb") as f:
        uploaded = client.files.create(file=f, purpose="batch")
    os.remove(jsonl_path)

    print(f"Submitting batch (file_id={uploaded.id})...")
    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    print(f"Batch submitted: id={batch.id}, status={batch.status}")
    return batch.id


def collect(batch_id: str, test_mode: bool = False) -> None:
    """Poll for batch completion, process results, and send the digest.

    Args:
        batch_id: The batch ID returned by the submit phase.
        test_mode: If True, limit to first 5 relevant papers.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)
    async_client = AsyncOpenAI(api_key=api_key)

    env_vars_set = {
        x: bool(os.environ.get(x)) for x in ["MY_EMAIL", "MY_PW", "OPENAI_API_KEY"]
    }
    if not all(env_vars_set.values()):
        msg = "Missing env vars: " + ", ".join(k for k, v in env_vars_set.items() if not v)
        raise ValueError(msg)

    # --- Load candidates ---
    with open(CANDIDATES_FILE) as f:
        candidates = json.load(f)

    with open("batch_metadata.json") as f:
        metadata = json.load(f)

    # --- Check batch status ---
    batch = client.batches.retrieve(batch_id)
    if batch.status not in ("completed", "expired", "cancelled", "failed"):
        print(f"Batch {batch_id} status: {batch.status} — not ready yet.")
        sys.exit(1)

    if batch.status != "completed":
        print(f"Batch {batch_id} finished with status: {batch.status}")
        sys.exit(1)

    # --- Download results ---
    print(f"Batch completed. Downloading results (file_id={batch.output_file_id})...")
    result_content = client.files.content(batch.output_file_id).text

    # Parse results: map custom_id -> classification
    classifications: dict[str, bool] = {}
    for line in result_content.strip().split("\n"):
        obj = json.loads(line)
        custom_id = obj["custom_id"]
        response_body = obj.get("response", {}).get("body", {})
        choices = response_body.get("choices", [])
        if choices:
            text = choices[0].get("message", {}).get("content", "").strip().lower()
            is_relevant = "not relevant" not in text and "relevant" in text
        else:
            is_relevant = False
        classifications[custom_id] = is_relevant

    # --- Build relevant papers list ---
    relevant_papers: list[dict] = []
    for idx, paper in enumerate(candidates):
        cid = f"paper-{idx}"
        if classifications.get(cid, False):
            relevant_papers.append(paper)
            if test_mode and len(relevant_papers) >= 5:
                break

    print(f"Found {len(relevant_papers)} relevant papers out of {len(candidates)} candidates")

    # --- Create GitHub issues ---
    ensure_github_label()
    for paper in relevant_papers:
        create_github_issue(
            paper["Title"],
            paper["Abstract"],
            paper["Authors"],
            paper["Journal"],
            paper["Link"],
        )

    # --- Email digest ---
    recipients = [os.environ.get("MY_EMAIL")]
    if os.environ.get("MY_EMAIL_2"):
        recipients.append(os.environ.get("MY_EMAIL_2"))

    message = MIMEMultipart()
    message["From"] = os.environ.get("MY_EMAIL")
    message["To"] = ", ".join(recipients)
    message["Subject"] = f"Papers {datetime.now()}"

    n_arxiv = metadata["n_arxiv"]
    n_pubmed = metadata["n_pubmed"]
    n_biorxiv = metadata["n_biorxiv"]
    biorxiv_errors = metadata["biorxiv_errors"]
    n_total = n_arxiv + n_pubmed + n_biorxiv

    body = f"*Fetched {n_total} papers ({n_arxiv} from Arxiv, {n_pubmed} from PubMed, and {n_biorxiv} from Biorxiv/Chemrxiv/Medrxiv)*\n\n"
    body += f"*Found {len(relevant_papers)} relevant papers.*\n\n"
    body += "*Classification performed via OpenAI Batch API.*\n\n"
    if biorxiv_errors:
        body += "*The following errors were encountered while scraping biorxiv/medrxiv/chemrxiv:*\n"
        for err in biorxiv_errors:
            body += f"  - {err}\n"
        body += "\n"
    body += "---\n\n"

    # Summarize in async batches
    print(f"Summarizing {len(relevant_papers)} papers...")
    summaries = asyncio.run(summarize_papers(async_client, relevant_papers))

    for paper, summary in zip(relevant_papers, summaries):
        title = paper["Title"]
        abstract = paper["Abstract"]
        journal = paper["Journal"]
        link = paper["Link"]
        authors = paper["Authors"]
        if link:
            body += f"### [{title}]({link})\n\n"
        else:
            body += f"### {title}\n\n"
        body += f"**Authors**: {authors}\n\n"
        body += f"**Venue**: {journal}\n\n"
        body += f"**Summary**: {summary}\n\n"
        body += f"**Abstract**: {abstract}\n\n---\n\n"

    html_body = markdown.markdown(body)
    message.attach(MIMEText(html_body, "html"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(os.environ.get("MY_EMAIL"), os.environ.get("MY_PW"))
        server.sendmail(os.environ.get("MY_EMAIL"), recipients, message.as_string())

    print("Email digest sent.")

    # Clean up intermediate files
    for path in [CANDIDATES_FILE, "batch_metadata.json"]:
        if os.path.exists(path):
            os.remove(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PubMedFetcher — OpenAI Batch API variant"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    submit_p = sub.add_parser("submit", help="Scrape papers and submit a batch job")
    submit_p.add_argument("--n-days", type=int, default=1)
    submit_p.add_argument("--test-mode", action="store_true")

    collect_p = sub.add_parser("collect", help="Collect batch results and send digest")
    collect_p.add_argument("--batch-id", required=True)
    collect_p.add_argument("--test-mode", action="store_true")

    args = parser.parse_args()

    if args.command == "submit":
        batch_id = submit(args.n_days, test_mode=args.test_mode)
        # Write batch ID so the workflow can read it
        print(f"BATCH_ID={batch_id}")
    elif args.command == "collect":
        collect(args.batch_id, test_mode=args.test_mode)


if __name__ == "__main__":
    main()
