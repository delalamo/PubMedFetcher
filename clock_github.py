import markdown
import json
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import pickle
from typing import Dict, List

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import pandas as pd
import numpy as np
from pymed import PubMed
import arxivscraper
from paperscraper.get_dumps import biorxiv, medrxiv, chemrxiv
from openai import OpenAI

# Load environment variables
load_dotenv()


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
    client: OpenAI, data: Dict[str, List], cutoff=0.05, test_mode: bool = False
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

    analyzed_data = {"Title": [], "Abstract": [], "Journal": [], "Relevance": []}
    prompts = []
    for i, (title, abstract, journal) in enumerate(
        zip(data["Title"], data["Abstract"], data["Journal"])
    ):
        if len(abstract.split()) < 100:
            continue
        prompts.append((abstract, title, journal))

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
                analyzed_data["Relevance"].append(prob)
            prompts = []
        if test_mode and len(analyzed_data["Title"]) >= 2:
            break
    df = pd.DataFrame.from_dict(analyzed_data)
    df = df.sort_values("Relevance", ascending=False)
    df = df[df["Relevance"] >= cutoff]
    return df


def summarize_abstract(client: OpenAI, abstract: str) -> str:
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
                "content": "You are a helpful assistant that summarizes scientific abstracts into plain english in a single sentence of no more than thirty words. Do not use semicolons, parentheses, or any other syntax intended to extend a sentence. Keep it simple and concise.",
            },
            {"role": "user", "content": abstract},
        ],
    )
    return response.choices[0].message.content.strip()


def scrape_arxiv(
    n_days: int, categories: List[str] = ["q-bio", "cond-mat", "stat"]
) -> Dict[str, List]:
    """Scrapes arXiv for papers in the q-bio category published in the last n_days.

    Args:
        n_days: The number of days to look back.

    Returns:
        A dictionary containing paper titles, abstracts, and journals.
    """
    print("Scraping arxiv")
    data = {"Title": [], "Abstract": [], "Journal": []}
    start = (
        str(datetime.now() - timedelta(days=n_days + 1)).split()[0].replace("/", "-")
    )
    end = str(datetime.now() - timedelta(days=1)).split()[0].replace("/", "-")

    for cat in categories:
        scraper = arxivscraper.Scraper(category=cat, date_from=start, date_until=end)

        arxiv_papers = scraper.scrape()

        # Indicates failuer
        if arxiv_papers is None or arxiv_papers == 1 or len(arxiv_papers) == 0:
            return data
        for paper in arxiv_papers:
            data["Title"].append(paper["title"])
            data["Abstract"].append(paper["abstract"].replace("\n", " "))
            data["Journal"].append("arXiv")
    return data


def scrape_biorxiv(n_days: int) -> Dict[str, List]:
    """Scrapes biorxiv, medrxiv, and chemrxiv for papers published in the last n_days.

    Args:
        n_days: The number of days to look back.

    Returns:
        A dictionary containing paper titles, abstracts, and journals.
    """
    print("Scraping biorxiv")
    start_rxivs = datetime.now() - timedelta(days=n_days + 1)
    end_rxivs = datetime.now() - timedelta(days=1)

    chemrxiv(
        begin_date=format_date(start_rxivs, "-"),
        end_date=format_date(end_rxivs, "-"),
        save_path="chemrxiv.jsonl",
    )
    medrxiv(
        begin_date=format_date(start_rxivs, "-"),
        end_date=format_date(end_rxivs, "-"),
        save_path="medrxiv.jsonl",
    )
    biorxiv(
        begin_date=format_date(start_rxivs, "-"),
        end_date=format_date(end_rxivs, "-"),
        save_path="biorxiv.jsonl",
    )

    data = {"Title": [], "Abstract": [], "Journal": []}
    for jsonfile in ["medrxiv.jsonl", "biorxiv.jsonl", "chemrxiv.jsonl"]:
        with open(jsonfile) as infile:
            for i, line in enumerate(infile):
                l = json.loads(line)
                data["Title"].append(l["title"])
                data["Abstract"].append(l["abstract"].replace("\n", " "))
                data["Journal"].append(jsonfile.split(".")[0])
    for file in ["chemrxiv", "biorxiv", "medrxiv"]:
        os.system(f"rm {file}.jsonl")
    return data


def scrape_pubmed(n_days: int) -> Dict[str, List]:
    """Scrapes PubMed for papers published in the last n_days.

    Args:
        n_days: The number of days to look back.

    Returns:
        A dictionary containing paper titles, abstracts, and journals.
    """
    print("Scraping pubmed")
    end = datetime.now() - timedelta(days=1)
    days = [format_date(end - timedelta(days=i + 1), "/") for i in range(n_days)]

    data = {"Title": [], "Abstract": [], "Journal": []}
    for date in days:
        pubmed = PubMed(tool="MyTool", email="your@email.address")
        search_query = f"{date}[PDAT]"
        try:
            results = pubmed.query(search_query, max_results=50000)
        except:
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
            except:
                data["Journal"].append("Journal not found")
    return data


def main(n_days: int, test_mode: bool = False) -> None:
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

    data_pubmed = scrape_pubmed(n_days)
    data_biorxiv = scrape_biorxiv(n_days)
    data_arxiv = scrape_arxiv(n_days)

    data = {"Title": [], "Abstract": [], "Journal": []}
    for d in [data_pubmed, data_biorxiv, data_arxiv]:
        n_titles = len(d["Title"])
        for field in data.keys():
            data[field].extend(d[field])

    df = embed_papers(client, data, test_mode=test_mode)

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    message = MIMEText(f"The current time is: {now}")

    message = MIMEMultipart()
    message["From"] = os.environ.get("MY_EMAIL")
    message["To"] = os.environ.get("MY_EMAIL")
    message["Subject"] = f"Papers {datetime.now()}"

    body = "Summary of fetched papers\n\n"
    body += "{} total papers from Arxiv. \n\n".format(len(data_arxiv["Title"]))
    body += "{} total papers from PubMed. \n\n".format(len(data_pubmed["Title"]))
    body += "{} total papers from Biorxiv, Chemrxiv, and Medrxiv. \n\n".format(
        len(data_biorxiv["Title"])
    )

    for _, row in df.iterrows():
        title = row["Title"]
        abstract = row["Abstract"]
        journal = row["Journal"]
        prob = row["Relevance"]
        summary = summarize_abstract(client, abstract)
        body += f"### {title}\n\n**Journal**: {journal}\n\n**Relevance**: {(100*prob):.1f}%\n\n**Summary**: {summary}\n\n**Abstract**: {abstract}\n\n---\n\n"

    html_body = markdown.markdown(body)
    # message.attach(MIMEText(body, "plain"))
    message.attach(MIMEText(html_body, "html"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(os.environ.get("MY_EMAIL"), os.environ.get("MY_PW"))
        server.sendmail(
            os.environ.get("MY_EMAIL"), os.environ.get("MY_EMAIL"), message.as_string()
        )
