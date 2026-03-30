import json
import os
import pickle
import smtplib
import time
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import arxiv
import markdown
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from openai import OpenAI
from paperscraper.get_dumps import biorxiv, chemrxiv, medrxiv
from pymed import PubMed

# Load environment variables
load_dotenv()


def openai_summary_prompt() -> str:
    """Returns the prompt used for summarizing abstracts with OpenAI."""
    return (
        "You are a helpful assistant that summarizes scientific abstracts into plain english in a single sentence of no more than thirty words. "
        "Do not use semicolons, parentheses, or any other syntax intended to extend a sentence. Keep it simple and concise. "
        "If the study is a literature review, mention that. "
        "If the study introduces a new method, and the name of the method is not in the title, mention the name of the method explicitly. "
        "Prioritize mentioning the main findings of the study. Do not offer your own interpretation. "
        "There is no need to be exhaustive about the contents of the study. Just broad highlights."
    )


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


def embed_papers(
    client: OpenAI, data: dict[str, list], cutoff=0.35, test_mode: bool = False
) -> pd.DataFrame:
    """Embeds papers using OpenAI's API and filters them by relevance.

    Args:
        client: An instance of the OpenAI client.
        data: A dictionary containing paper titles, abstracts, journals, links, and authors.
        cutoff: The minimum relevance score for a paper to be included.
        test_mode: If True, stop after finding 5 relevant papers.

    Returns:
        A Pandas DataFrame with relevant papers, sorted by relevance.
    """
    with open("model_openai.pkl", "rb") as f:
        clf = pickle.load(f)

    analyzed_data = {"Title": [], "Abstract": [], "Journal": [], "Link": [], "Authors": [], "Relevance": []}
    prompts = []
    links = data.get("Link", [""] * len(data["Title"]))
    authors_list = data.get("Authors", [""] * len(data["Title"]))
    for i, (title, abstract, journal, link, authors) in enumerate(
        zip(data["Title"], data["Abstract"], data["Journal"], links, authors_list)
    ):
        if len(abstract.split()) < 100:
            continue
        prompts.append((abstract, title, journal, link, authors))

        if len(prompts) >= 100 or i == len(data["Title"]) - 1:
            abstracts = [p[0] for p in prompts]
            response = client.embeddings.create(
                input=abstracts, model="text-embedding-3-small"
            )
            for p, d in zip(prompts, response.data):
                prob = clf.predict_proba(np.asarray(d.embedding[:512]).reshape(1, -1))
                prob = prob[:, 1].item()
                if prob < cutoff:
                    continue
                analyzed_data["Title"].append(p[1])
                analyzed_data["Abstract"].append(p[0])
                analyzed_data["Journal"].append(p[2])
                analyzed_data["Link"].append(p[3])
                analyzed_data["Authors"].append(p[4])
                analyzed_data["Relevance"].append(prob)
            prompts = []
        if test_mode and len(analyzed_data["Title"]) >= 5:
            break
    df = pd.DataFrame.from_dict(analyzed_data)
    df = df.sort_values("Relevance", ascending=False)
    df = df[df["Relevance"] >= cutoff]
    return df


def summarize_abstract(client: OpenAI, title: str, abstract: str, model: str = "gpt-5-nano") -> str:
    """Summarize an abstract into a single sentence using the OpenAI API.

    Args:
        client: An instance of the OpenAI client.
        title: The paper title.
        abstract: The abstract text to summarize.
        model: The OpenAI model to use for summarization.

    Returns:
        A concise summary of the abstract.
    """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": openai_summary_prompt(),
            },
            {
                "role": "user",
                "content": f"Title of the study: {title}; Abstract: {abstract}",
            },
        ],
    )
    if not response.choices:
        return "Summary not available."
    return response.choices[0].message.content.strip()


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
    start = (datetime.now() - timedelta(days=n_days + 1)).strftime("%Y%m%d")
    end = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")

    client = arxiv.Client(delay_seconds=5.0, num_retries=5)
    for cat in categories:
        query = f"cat:{cat} AND submittedDate:[{start}0000 TO {end}2359]"
        search = arxiv.Search(
            query=query,
            max_results=1000,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending,
        )
        retries = 0
        max_retries = 5
        while retries <= max_retries:
            try:
                for result in client.results(search):
                    data["Title"].append(result.title)
                    data["Abstract"].append(result.summary.replace("\n", " "))
                    data["Journal"].append("arXiv")
                    data["Link"].append(result.entry_id)
                    authors = ", ".join(author.name for author in result.authors)
                    data["Authors"].append(authors if authors else "Authors not available")
                break
            except arxiv.HTTPError as e:
                if e.status == 429 and retries < max_retries:
                    wait = 60 * (2 ** retries)
                    print(f"arXiv rate limit hit for category {cat}, waiting {wait}s before retry {retries + 1}/{max_retries}...")
                    time.sleep(wait)
                    retries += 1
                else:
                    raise
        if cat != categories[-1]:
            time.sleep(10)
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
            start_date=format_date(start_rxivs, "-"),
            end_date=format_date(end_rxivs, "-"),
            save_path="chemrxiv.jsonl",
        )
    except Exception as e:
        error_msgs.append(f"Chemrxiv scrape failed with error {e}. Continuing...")
    try:
        medrxiv(
            start_date=format_date(start_rxivs, "-"),
            end_date=format_date(end_rxivs, "-"),
            save_path="medrxiv.jsonl",
        )
    except Exception as e:
        error_msgs.append(f"Medrxiv scrape failed with error {e}. Continuing...")
    try:
        biorxiv(
            start_date=format_date(start_rxivs, "-"),
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
                doi = entry.get("doi", "")
                if doi:
                    data["Link"].append(f"https://doi.org/{doi}")
                else:
                    data["Link"].append("")
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
        for attempt in range(3):
            try:
                results = pubmed.query(search_query, max_results=50000)
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
                    doi = getattr(article, "doi", None)
                    pubmed_id = getattr(article, "pubmed_id", None)
                    if doi:
                        doi = str(doi).strip().split("\n")[0]
                        data["Link"].append(f"https://doi.org/{doi}")
                    elif pubmed_id:
                        pmid = str(pubmed_id).strip().split("\n")[0]
                        data["Link"].append(f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/")
                    else:
                        data["Link"].append("")
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
                break  # Successful query
            except requests.exceptions.ConnectionError:
                if attempt < 2:
                    time.sleep(2 ** (attempt + 1))  # 2s, then 4s
            except Exception:
                break  # Non-connection error, skip this date
    return data, ""


def main(n_days: int, test_mode: bool = False, cutoff: float = 3.5, model: str = "gpt-5-nano") -> None:
    """Scrapes papers from PubMed, biorxiv, and arXiv, embeds them, and sends an email.

    Args:
        n_days: The number of days to look back.
        test_mode: If True, stop after finding 5 relevant papers.
        cutoff: The minimum relevance score (out of 10) for a paper to be included.
        model: The OpenAI model to use for summarization.
    """

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    env_vars_set = {
        x: bool(os.environ.get(x)) for x in ["MY_EMAIL", "MY_PW", "OPENAI_API_KEY"]
    }

    if not all(env_vars_set.values()):
        msg1 = "One or more environment variables are not set: "
        msg2 = ", ".join([k for k, v in env_vars_set.items() if not v])
        raise ValueError(msg1 + msg2)

    data_pubmed, _ = scrape_pubmed(n_days)
    data_biorxiv, biorxiv_errors = scrape_biorxiv(n_days)
    data_arxiv, _ = scrape_arxiv(n_days)

    data = {"Title": [], "Abstract": [], "Journal": [], "Link": [], "Authors": []}
    for d in [data_pubmed, data_biorxiv, data_arxiv]:
        for field in data:
            data[field].extend(d[field])

    df = embed_papers(client, data, test_mode=test_mode, cutoff=cutoff / 10)

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
    body += f"*Found {len(df)} relevant papers with cutoff {cutoff / 10:.2f}.*\n\n"
    if len(biorxiv_errors) > 0:
        body += "*The following errors were encountered while scraping biorxiv/medrxiv/chemrxiv:*\n"
        for err in biorxiv_errors:
            body += f"  - {err}\n"
        body += "\n"
    body += "---\n\n"

    for _, row in df.iterrows():
        title = row["Title"]
        abstract = row["Abstract"]
        journal = row["Journal"]
        link = row["Link"]
        authors = row["Authors"]
        prob = row["Relevance"]
        summary = summarize_abstract(client, title, abstract, model=model)
        if link:
            body += f"### [{title}]({link})\n\n"
        else:
            body += f"### {title}\n\n"
        body += f"**Authors**: {authors}\n\n"
        body += f"**Venue**: {journal}\n\n"
        body += f"**Relevance**: {(10*prob):.2f}/10\n\n"
        body += f"**Summary**: {summary}\n\n"
        body += f"**Abstract**: {abstract}\n\n---\n\n"

    html_body = markdown.markdown(body)
    message.attach(MIMEText(html_body, "html"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(os.environ.get("MY_EMAIL"), os.environ.get("MY_PW"))
        server.sendmail(
            os.environ.get("MY_EMAIL"), recipients, message.as_string()
        )
