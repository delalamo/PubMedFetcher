import asyncio
import contextlib
import json
import os
import re
import smtplib
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import arxivscraper
import markdown
import requests
from dotenv import load_dotenv
from openai import AsyncOpenAI
from paperscraper.get_dumps import biorxiv, chemrxiv, medrxiv
from pymed import PubMed

# Load environment variables
load_dotenv()


def load_keywords(filepath: str = "keywords.txt") -> list[str]:
    """Load keywords from a newline-delimited file.

    Args:
        filepath: Path to the keywords file. One keyword/phrase per line.

    Returns:
        A list of keyword strings.
    """
    with open(filepath) as f:
        return [line.strip() for line in f if line.strip()]


def format_date(date, sep: str = "/") -> str:
    """Formats a datetime object into a string.

    Args:
        date: The datetime object to format.
        sep: The separator to use between year, month, and day.

    Returns:
        A string representation of the date in YYYY/MM/DD format.
    """
    assert len(sep) == 1
    return sep.join([date.strftime(f"%{x}") for x in "Ymd"])


def deduplicate_papers(data: dict[str, list]) -> dict[str, list]:
    """Remove duplicate papers based on normalized title.

    Args:
        data: Dictionary with keys Title, Abstract, Journal, Link, Authors.

    Returns:
        A new dictionary with duplicates removed (first occurrence kept).
    """
    seen: set[str] = set()
    out: dict[str, list] = {k: [] for k in data}
    for i in range(len(data["Title"])):
        # Normalize: lowercase, strip whitespace/punctuation for comparison
        key = re.sub(r"[^a-z0-9 ]", "", data["Title"][i].lower()).strip()
        if key in seen:
            continue
        seen.add(key)
        for field in data:
            out[field].append(data[field][i])
    return out


async def classify_paper(
    client: AsyncOpenAI, title: str, abstract: str, keywords: list[str]
) -> bool:
    """Classify a paper as relevant or not using gpt-5-nano.

    Args:
        client: An instance of the async OpenAI client.
        title: The paper title.
        abstract: The paper abstract.
        keywords: List of keyword phrases defining research interests.

    Returns:
        True if the paper is classified as relevant, False otherwise.
    """
    keywords_str = "\n".join(f"- {kw}" for kw in keywords)
    response = await client.chat.completions.create(
        model="gpt-5-nano",
        messages=[
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
        ],
        max_tokens=10,
    )
    if not response.choices:
        return False
    result = response.choices[0].message.content.strip().lower()
    return "not relevant" not in result and "relevant" in result


def classify_papers(
    async_client: AsyncOpenAI,
    data: dict[str, list],
    keywords: list[str],
    test_mode: bool = False,
    batch_size: int = 50,
) -> list[dict]:
    """Classify papers using gpt-5-nano and return relevant ones.

    Sends requests in concurrent batches for speed.

    Args:
        async_client: An instance of the async OpenAI client.
        data: A dictionary containing paper data (Title, Abstract, Journal, Link, Authors).
        keywords: List of keyword phrases defining research interests.
        test_mode: If True, stop after finding 5 relevant papers.
        batch_size: Number of concurrent classification requests per batch.

    Returns:
        A list of dicts, each representing a relevant paper.
    """

    async def _classify_all() -> list[dict]:
        # Collect papers that pass the word-count filter
        candidates = []
        links = data.get("Link", [""] * len(data["Title"]))
        authors_list = data.get("Authors", [""] * len(data["Title"]))
        for title, abstract, journal, link, authors in zip(
            data["Title"], data["Abstract"], data["Journal"], links, authors_list
        ):
            if len(abstract.split()) < 100:
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

        relevant = []
        for i in range(0, len(candidates), batch_size):
            batch = candidates[i : i + batch_size]
            results = await asyncio.gather(
                *[
                    classify_paper(async_client, p["Title"], p["Abstract"], keywords)
                    for p in batch
                ]
            )
            for paper, is_relevant in zip(batch, results):
                if is_relevant:
                    relevant.append(paper)
            if test_mode and len(relevant) >= 5:
                return relevant[:5]
            # Brief pause between batches to stay within TPM limits
            if i + batch_size < len(candidates):
                await asyncio.sleep(2)
        return relevant

    return asyncio.run(_classify_all())


def ensure_github_label() -> None:
    """Ensure the 'paper' label exists in the GitHub repository."""
    token = os.environ.get("GITHUB_TOKEN")
    repo = os.environ.get("GITHUB_REPOSITORY")
    if not token or not repo:
        return
    url = f"https://api.github.com/repos/{repo}/labels"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
    }
    # Attempt to create; ignore errors if label already exists
    with contextlib.suppress(Exception):
        requests.post(
            url,
            headers=headers,
            json={
                "name": "paper",
                "color": "0075ca",
                "description": "Relevant paper identified by PubMedFetcher",
            },
            timeout=30,
        )


def create_github_issue(
    title: str,
    abstract: str,
    authors: str,
    journal: str,
    link: str,
) -> bool:
    """Create a GitHub issue for a relevant paper.

    Args:
        title: The paper title.
        abstract: The paper abstract.
        authors: Comma-separated list of authors.
        journal: The venue/journal name.
        link: URL to the paper (DOI or preprint link).

    Returns:
        True if the issue was created successfully, False otherwise.
    """
    token = os.environ.get("GITHUB_TOKEN")
    repo = os.environ.get("GITHUB_REPOSITORY")
    if not token or not repo:
        print("GITHUB_TOKEN or GITHUB_REPOSITORY not set, skipping issue creation")
        return False

    body = f"**Authors**: {authors}\n\n"
    body += f"**Venue**: {journal}\n\n"
    if link:
        body += f"**Link**: {link}\n\n"
    body += f"### Abstract\n\n{abstract}\n"

    url = f"https://api.github.com/repos/{repo}/issues"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
    }
    payload = {"title": title, "body": body, "labels": ["paper"]}

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        return True
    except requests.exceptions.HTTPError:
        # Retry without labels in case label doesn't exist
        payload.pop("labels", None)
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=30)
            resp.raise_for_status()
            return True
        except Exception as e:
            print(f"Failed to create issue for '{title}': {e}")
            return False
    except Exception as e:
        print(f"Failed to create issue for '{title}': {e}")
        return False


def scrape_arxiv(
    n_days: int, categories: list[str] | None = None
) -> tuple[dict[str, list], str]:
    """Scrapes arXiv for papers published in the last n_days.

    Args:
        n_days: The number of days to look back.
        categories: List of arXiv categories to scrape. Defaults to q-bio, cond-mat, stat.

    Returns:
        A tuple of (dictionary containing paper data, error message).
    """
    if categories is None:
        categories = ["q-bio", "cond-mat", "stat"]
    print("Scraping arxiv")
    data = {"Title": [], "Abstract": [], "Journal": [], "Link": [], "Authors": []}
    start = (datetime.now() - timedelta(days=n_days + 1)).strftime("%Y-%m-%d")
    end = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    for cat in categories:
        scraper = arxivscraper.Scraper(category=cat, date_from=start, date_until=end)

        arxiv_papers = scraper.scrape()

        # Indicates failure
        if arxiv_papers is None or arxiv_papers == 1 or len(arxiv_papers) == 0:
            return data, ""
        for paper in arxiv_papers:
            data["Title"].append(paper["title"])
            data["Abstract"].append(paper["abstract"].replace("\n", " "))
            data["Journal"].append("arXiv")
            # Construct arXiv link from paper ID
            arxiv_id = paper.get("id", "")
            if arxiv_id:
                data["Link"].append(f"https://arxiv.org/abs/{arxiv_id}")
            else:
                data["Link"].append("")
            # Extract authors
            authors = paper.get("authors", "")
            if isinstance(authors, list):
                authors = ", ".join(authors)
            data["Authors"].append(authors if authors else "Authors not available")
    return data, ""


def scrape_biorxiv(n_days: int) -> tuple[dict[str, list], list[str]]:
    """Scrapes biorxiv, medrxiv, and chemrxiv for papers published in the last n_days.

    Args:
        n_days: The number of days to look back.

    Returns:
        A tuple of (dictionary containing paper data, list of error messages).
    """
    print("Scraping biorxiv")
    start_rxivs = datetime.now() - timedelta(days=n_days + 1)
    end_rxivs = datetime.now() - timedelta(days=1)

    error_msgs = []
    try:
        chemrxiv(
            begin_date=format_date(start_rxivs, "-"),
            end_date=format_date(end_rxivs, "-"),
            save_path="chemrxiv.jsonl",
        )
    except Exception as e:
        error_msgs.append(f"Chemrxiv scrape failed with error {e}. Continuing...")
    try:
        medrxiv(
            begin_date=format_date(start_rxivs, "-"),
            end_date=format_date(end_rxivs, "-"),
            save_path="medrxiv.jsonl",
        )
    except Exception as e:
        error_msgs.append(f"Medrxiv scrape failed with error {e}. Continuing...")
    try:
        biorxiv(
            begin_date=format_date(start_rxivs, "-"),
            end_date=format_date(end_rxivs, "-"),
            save_path="biorxiv.jsonl",
        )
    except Exception as e:
        error_msgs.append(f"Biorxiv scrape failed with error {e}. Continuing...")
    data = {"Title": [], "Abstract": [], "Journal": [], "Link": [], "Authors": []}
    for jsonfile in ["medrxiv.jsonl", "biorxiv.jsonl", "chemrxiv.jsonl"]:
        if not os.path.exists(jsonfile):
            continue
        with open(jsonfile) as infile:
            for line in infile:
                entry = json.loads(line)
                data["Title"].append(entry["title"])
                data["Abstract"].append(entry["abstract"].replace("\n", " "))
                data["Journal"].append(jsonfile.split(".")[0])
                # Construct DOI link if available
                doi = entry.get("doi", "")
                if doi:
                    data["Link"].append(f"https://doi.org/{doi}")
                else:
                    data["Link"].append("")
                # Extract authors
                authors = entry.get("authors", "")
                if isinstance(authors, list):
                    authors = ", ".join(authors)
                data["Authors"].append(authors if authors else "Authors not available")
    for file in ["chemrxiv", "biorxiv", "medrxiv"]:
        filepath = f"{file}.jsonl"
        if os.path.exists(filepath):
            os.remove(filepath)
    return data, error_msgs


def scrape_pubmed(n_days: int) -> tuple[dict[str, list], str]:
    """Scrapes PubMed for papers published in the last n_days.

    Args:
        n_days: The number of days to look back.

    Returns:
        A tuple of (dictionary containing paper data, error message).
    """
    print("Scraping pubmed")
    end = datetime.now() - timedelta(days=1)
    days = [format_date(end - timedelta(days=i + 1), "/") for i in range(n_days)]

    data = {"Title": [], "Abstract": [], "Journal": [], "Link": [], "Authors": []}
    for date in days:
        pubmed = PubMed(tool="MyTool", email="your@email.address")
        search_query = f"{date}[PDAT]"
        try:
            results = pubmed.query(search_query, max_results=50000)
        except Exception:
            continue
        for article in results:
            if article.abstract is None:
                continue
            abstract = article.abstract.replace("\n", " ")
            if len(abstract.split()) < 100:
                continue
            data["Title"].append(article.title)
            data["Abstract"].append(abstract)
            try:
                data["Journal"].append(article.journal.strip().replace("\n", " "))
            except (AttributeError, TypeError):
                data["Journal"].append("Journal not found")
            # Prefer DOI link, fallback to PubMed link
            doi = getattr(article, "doi", None)
            pubmed_id = getattr(article, "pubmed_id", None)
            if doi:
                # DOI may contain newlines, clean it
                doi = str(doi).strip().split("\n")[0]
                data["Link"].append(f"https://doi.org/{doi}")
            elif pubmed_id:
                # Use first PMID if multiple
                pmid = str(pubmed_id).strip().split("\n")[0]
                data["Link"].append(f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/")
            else:
                data["Link"].append("")
            # Extract authors
            authors = getattr(article, "authors", None)
            if authors and isinstance(authors, list):
                author_names = []
                for a in authors:
                    if isinstance(a, dict):
                        firstname = a.get("firstname", "") or ""
                        lastname = a.get("lastname", "") or ""
                        name = f"{firstname} {lastname}".strip()
                        if name:
                            author_names.append(name)
                    elif isinstance(a, str):
                        author_names.append(a)
                data["Authors"].append(
                    ", ".join(author_names) if author_names else "Authors not available"
                )
            else:
                data["Authors"].append("Authors not available")
    return data, ""


def scrape_all_sources(n_days: int) -> tuple[dict[str, list], dict[str, list], dict[str, list], list[str]]:
    """Scrape PubMed, bioRxiv, and arXiv in parallel using threads.

    Args:
        n_days: The number of days to look back.

    Returns:
        A tuple of (pubmed_data, biorxiv_data, arxiv_data, biorxiv_errors).
    """
    with ThreadPoolExecutor(max_workers=3) as pool:
        fut_pubmed = pool.submit(scrape_pubmed, n_days)
        fut_biorxiv = pool.submit(scrape_biorxiv, n_days)
        fut_arxiv = pool.submit(scrape_arxiv, n_days)

    data_pubmed, _ = fut_pubmed.result()
    data_biorxiv, biorxiv_errors = fut_biorxiv.result()
    data_arxiv, _ = fut_arxiv.result()
    return data_pubmed, data_biorxiv, data_arxiv, biorxiv_errors


def main(n_days: int, test_mode: bool = False) -> None:
    """Scrapes papers, classifies them with gpt-5-nano, creates GitHub issues,
    and sends an email digest.

    Args:
        n_days: The number of days to look back.
        test_mode: If True, stop after finding 5 relevant papers.
    """

    api_key = os.environ.get("OPENAI_API_KEY")
    async_client = AsyncOpenAI(api_key=api_key)

    env_vars_set = {
        x: bool(os.environ.get(x)) for x in ["MY_EMAIL", "MY_PW", "OPENAI_API_KEY"]
    }

    if not all(env_vars_set.values()):
        msg1 = "One or more environment variables are not set: "
        msg2 = ", ".join([k for k, v in env_vars_set.items() if not v])
        raise ValueError(msg1 + msg2)

    # Load keywords from file
    keywords = load_keywords()
    print(f"Loaded {len(keywords)} keywords")

    # Scrape all sources in parallel
    data_pubmed, data_biorxiv, data_arxiv, biorxiv_errors = scrape_all_sources(n_days)

    data = {"Title": [], "Abstract": [], "Journal": [], "Link": [], "Authors": []}
    for d in [data_pubmed, data_biorxiv, data_arxiv]:
        for field in data:
            data[field].extend(d[field])

    # Deduplicate papers by normalized title
    before_dedup = len(data["Title"])
    data = deduplicate_papers(data)
    after_dedup = len(data["Title"])
    print(f"Deduplicated: {before_dedup} -> {after_dedup} papers ({before_dedup - after_dedup} duplicates removed)")

    print(f"Classifying {len(data['Title'])} papers with gpt-5-nano...")
    relevant_papers = classify_papers(async_client, data, keywords, test_mode=test_mode)
    print(f"Found {len(relevant_papers)} relevant papers")

    # Create GitHub issues for relevant papers
    ensure_github_label()
    for paper in relevant_papers:
        create_github_issue(
            paper["Title"],
            paper["Abstract"],
            paper["Authors"],
            paper["Journal"],
            paper["Link"],
        )

    # Build recipient list - include second email if configured
    recipients = [os.environ.get("MY_EMAIL")]
    if os.environ.get("MY_EMAIL_2"):
        recipients.append(os.environ.get("MY_EMAIL_2"))

    message = MIMEMultipart()
    message["From"] = os.environ.get("MY_EMAIL")
    message["To"] = ", ".join(recipients)
    message["Subject"] = f"Papers {datetime.now()}"

    n_arxiv = len(data_arxiv["Title"])
    n_pubmed = len(data_pubmed["Title"])
    n_biorxiv = len(data_biorxiv["Title"])
    n_total = n_arxiv + n_pubmed + n_biorxiv
    body = f"*Fetched {n_total} papers ({n_arxiv} from Arxiv, {n_pubmed} from PubMed, and {n_biorxiv} from Biorxiv/Chemrxiv/Medrxiv)*\n\n"
    body += f"*Found {len(relevant_papers)} relevant papers.*\n\n"
    if len(biorxiv_errors) > 0:
        body += "*The following errors were encountered while scraping biorxiv/medrxiv/chemrxiv:*\n"
        for err in biorxiv_errors:
            body += f"  - {err}\n"
        body += "\n"
    body += "---\n\n"

    for paper in relevant_papers:
        title = paper["Title"]
        abstract = paper["Abstract"]
        journal = paper["Journal"]
        link = paper["Link"]
        authors = paper["Authors"]
        # Add link to title if available
        if link:
            body += f"### [{title}]({link})\n\n"
        else:
            body += f"### {title}\n\n"
        body += f"**Authors**: {authors}\n\n"
        body += f"**Venue**: {journal}\n\n"
        body += f"**Abstract**: {abstract}\n\n---\n\n"

    html_body = markdown.markdown(body)
    message.attach(MIMEText(html_body, "html"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(os.environ.get("MY_EMAIL"), os.environ.get("MY_PW"))
        server.sendmail(os.environ.get("MY_EMAIL"), recipients, message.as_string())
