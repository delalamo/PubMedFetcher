# PubMedFetcher: Automated research paper discovery

This repo scrapes new papers every day from five preprint and publication sources, uses gpt-5-nano to decide which ones match your research interests, and delivers the results to you — no manual searching required.

## What it does

Every day a GitHub Actions workflow:

1. **Scrapes** new papers from **PubMed**, **arXiv** (q-bio, cond-mat, and stats sections), **bioRxiv**, **medRxiv**, and **chemRxiv**.
2. **Classifies** each paper by sending its title and abstract, along with your keywords, to **gpt-5-nano**. Papers whose topics match your keywords are marked as relevant.
3. **Creates a GitHub Issue** for every relevant paper containing the title, authors, venue, abstract, and a link or DOI.
4. **Emails you an HTML digest** with a one-sentence AI-generated summary of each relevant paper.

All you need to provide is an OpenAI API key, a Gmail address for email delivery, and a list of keywords that describe the topics you care about.

## How to use this repo

Follow these steps to set up your own personal paper feed:

### 1. Fork the repository

Click **Fork** at the top of this page to create your own copy.

### 2. Edit `keywords.txt`

Open `keywords.txt` in the root of your fork and replace the contents with your own research interests, one keyword or phrase per line. For example:

```
CRISPR gene editing
single-cell RNA sequencing
spatial transcriptomics
tumor microenvironment
```

These keywords are sent to gpt-5-nano alongside each paper's title and abstract so the model can judge relevance. Be as specific or as broad as you like — the model understands natural-language descriptions of research topics. This costs about ~$0.10 per day in OpenAI credits.

### 3. Add repository secrets

Go to your fork's **Settings → Secrets and variables → Actions** and add the following secrets:

| Secret | Required | Description |
|--------|----------|-------------|
| `OPENAI_API_KEY` | Yes | An [OpenAI API key](https://platform.openai.com/api-keys) used for gpt-5-nano classification and summarization |
| `MY_EMAIL` | Yes | Your Gmail address (used as both sender and recipient) |
| `MY_PW` | Yes | A Gmail [App Password](https://support.google.com/accounts/answer/185833) (not your regular Gmail password) |
| `MY_EMAIL_2` | No | An optional second email address to receive the daily digest |

### 4. Enable GitHub Actions

Go to **Settings → Actions → General** and select **Allow all actions and reusable workflows**.

### 5. Test it

Go to the **Actions** tab, select the **Test fetching script on reduced input set** workflow, and click **Run workflow**. This runs the full pipeline but stops after finding 5 relevant papers so you can verify everything works without waiting for the full run.

### 6. You're done

The main workflow runs automatically every day at **midnight UTC**. You can also trigger it manually from the **Actions** tab at any time. Relevant papers will appear as GitHub Issues in your fork and land in your inbox as an email digest.

## How it works

The core pipeline lives in `run.py` and is orchestrated by a GitHub Actions workflow (`.github/workflows/submit_jobs.yml`):

1. **Scraping** — Papers from the last day are fetched from PubMed (via `pymed`), arXiv (via `arxivscraper`), and bioRxiv/medRxiv/chemRxiv (via `paperscraper`). Abstracts shorter than 100 words are filtered out as likely incomplete.
2. **Classification** — Each paper's title and abstract are sent to gpt-5-nano in concurrent batches of 20 along with your keywords from `keywords.txt`. The model returns a binary relevant / not-relevant decision.
3. **Issue creation** — For each relevant paper, the script calls the GitHub API to open an issue labeled `paper` containing the full metadata and abstract.
4. **Email digest** — Relevant papers are summarized into single sentences by gpt-5-nano and compiled into an HTML email sent via Gmail SMTP.

## Local development

```bash
pip install -r requirements.txt
pytest
```

You can also run the pipeline locally by setting the required environment variables (`OPENAI_API_KEY`, `MY_EMAIL`, `MY_PW`, `GITHUB_TOKEN`, `GITHUB_REPOSITORY`) and calling:

```python
from run import main
main(1, test_mode=True)
```
