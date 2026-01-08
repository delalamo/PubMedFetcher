import json
import os
import pickle
import smtplib
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import arxivscraper
import markdown
import numpy as np
import pandas as pd
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
        data: A dictionary containing paper titles, abstracts, and journals.
        cutoff: The minimum relevance score for a paper to be included.

    Returns:
        A Pandas DataFrame with relevant papers, sorted by relevance.
    """
    with open("model_openai.pkl", "rb") as f:
        clf = pickle.load(f)

    analyzed_data = {"Title": [], "Abstract": [], "Journal": [], "Link": [], "Relevance": []}
    prompts = []
    links = data.get("Link", [""] * len(data["Title"]))
    for i, (title, abstract, journal, link) in enumerate(
        zip(data["Title"], data["Abstract"], data["Journal"], links)
    ):
        if len(abstract.split()) < 100:
            continue
        prompts.append((abstract, title, journal, link))

        if len(prompts) >= 100 or i == len(data) - 1:
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
                analyzed_data["Relevance"].append(prob)
            prompts = []
        if test_mode and len(analyzed_data["Title"]) >= 5:
            break
    df = pd.DataFrame.from_dict(analyzed_data)
    df = df.sort_values("Relevance", ascending=False)
    df = df[df["Relevance"] >= cutoff]
    return df


def summarize_abstract(client: OpenAI, title: str, abstract: str) -> str:
    """Summarize an abstract into two sentences using the OpenAI API.

    Args:
        client: An instance of the OpenAI client.
        abstract: The abstract text to summarize.

    Returns:
        A concise summary of the abstract.
    """
    response = client.chat.completions.create(
        model="gpt-5-nano",
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
    """Scrapes arXiv for papers in the q-bio category published in the last n_days.

    Args:
        n_days: The number of days to look back.
        categories: List of arXiv categories to scrape. Defaults to q-bio, cond-mat, stat.

    Returns:
        A tuple of (dictionary containing paper titles, abstracts, and journals, error message).
    """
    if categories is None:
        categories = ["q-bio", "cond-mat", "stat"]
    print("Scraping arxiv")
    data = {"Title": [], "Abstract": [], "Journal": [], "Link": []}
    start = (datetime.now() - timedelta(days=n_days + 1)).strftime("%Y-%m-%d")
    end = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    for cat in categories:
        scraper = arxivscraper.Scraper(category=cat, date_from=start, date_until=end)

        arxiv_papers = scraper.scrape()

        # Indicates failuer
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
    return data, ""


def scrape_biorxiv(n_days: int) -> tuple[dict[str, list], list[str]]:
    """Scrapes biorxiv, medrxiv, and chemrxiv for papers published in the last n_days.

    Args:
        n_days: The number of days to look back.

    Returns:
        A tuple of (dictionary containing paper titles, abstracts, and journals, list of error messages).
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
    data = {"Title": [], "Abstract": [], "Journal": [], "Link": []}
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
        A tuple of (dictionary containing paper titles, abstracts, and journals, error message).
    """
    print("Scraping pubmed")
    end = datetime.now() - timedelta(days=1)
    days = [format_date(end - timedelta(days=i + 1), "/") for i in range(n_days)]

    data = {"Title": [], "Abstract": [], "Journal": [], "Link": []}
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
    return data, ""


def main(n_days: int, test_mode: bool = False, cutoff: float = 3.5) -> None:
    """Scrapes papers from PubMed, biorxiv, and arXiv, embeds them, and sends an email.

    Args:
        n_days: The number of days to look back.
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

    data = {"Title": [], "Abstract": [], "Journal": [], "Link": []}
    for d in [data_pubmed, data_biorxiv, data_arxiv]:
        for field in data:
            data[field].extend(d[field])

    df = embed_papers(client, data, test_mode=test_mode, cutoff=cutoff / 10)

    message = MIMEMultipart()
    message["From"] = os.environ.get("MY_EMAIL")
    message["To"] = os.environ.get("MY_EMAIL")
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
        prob = row["Relevance"]
        summary = summarize_abstract(client, title, abstract)
        # Add link to title if available
        if link:
            body += f"### [{title}]({link})\n\n"
        else:
            body += f"### {title}\n\n"
        body += f"**Journal**: {journal}\n\n**Relevance**: {(10*prob):.2f}/10\n\n**Summary**: {summary}\n\n**Abstract**: {abstract}\n\n---\n\n"

    html_body = markdown.markdown(body)
    # message.attach(MIMEText(body, "plain"))
    message.attach(MIMEText(html_body, "html"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(os.environ.get("MY_EMAIL"), os.environ.get("MY_PW"))
        server.sendmail(
            os.environ.get("MY_EMAIL"), os.environ.get("MY_EMAIL"), message.as_string()
        )
