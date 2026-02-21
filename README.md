# PubMedFetcher

This repo fetches new papers every day and uses GPT-4o-mini to classify them as relevant or not based on a configurable list of keywords. Relevant papers are:

1. Filed as **GitHub Issues** with the title, authors, venue, abstract, and link/DOI
2. Emailed as an **HTML digest** with AI-generated summaries

The workflow runs daily via GitHub Actions. It scrapes abstracts from biorxiv, chemrxiv, medrxiv, arXiv (q-bio, cond-mat, and stats sections), and PubMed. Each paper's title and abstract are sent to GPT-4o-mini along with your keywords to determine relevance. Relevant papers are then summarized and delivered to you.

## How to use this repo

1. **Fork this repo**
2. **Edit `keywords.txt`** in the repo root to list your research interests, one per line. For example:
   ```
   protein design
   protein language models
   antibody bioinformatics
   ```
3. **Add the following secrets** under Settings > Secrets and variables > Actions:
   | Secret | Required | Description |
   |--------|----------|-------------|
   | `OPENAI_API_KEY` | Yes | OpenAI API key for GPT-4o-mini classification and summarization |
   | `MY_EMAIL` | Yes | Your Gmail address (used as sender and recipient) |
   | `MY_PW` | Yes | A Gmail [app password](https://support.google.com/accounts/answer/185833) (not your actual email password) |
   | `MY_EMAIL_2` | No | A second email address to also receive the daily digest |

4. **Ensure Actions are enabled** in your fork (Settings > Actions > General > Allow all actions)

That's it. The workflow runs daily at midnight UTC. You can also trigger it manually from the Actions tab.

## How it works

1. **Scrape** papers from PubMed, arXiv, biorxiv, medrxiv, and chemrxiv
2. **Classify** each paper by sending its title and abstract along with your keywords to GPT-4o-mini
3. **Create GitHub Issues** for each relevant paper (with title, authors, venue, link, and abstract)
4. **Email a digest** with AI-generated summaries of each relevant paper

## Customization

- **Keywords**: Edit `keywords.txt` — one keyword or phrase per line
- **arXiv categories**: Modify the `categories` default in `scrape_arxiv()` in `run.py`
- **Summarization style**: Modify `openai_summary_prompt()` in `run.py`
- **Schedule**: Edit the cron expression in `.github/workflows/submit_jobs.yml`

## Testing

Trigger the test workflow manually from the Actions tab. It runs the same pipeline but stops after finding 5 relevant papers.

```bash
# Run tests locally
pip install -r requirements.txt
pytest
```
