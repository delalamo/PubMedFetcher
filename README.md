# PubMedFetcher
This repo contains the code I use for fetching new papers every morning that might interest me. The good stuff is in `clock.py`. Right now (as of 9 September 2025) the main script is called daily via Github Actions and fetches abstracts from biorxiv, chemrxiv, medrxiv, arxiv (the bio-ml and stats sections only), and PubMed. These are then embedded using the OpenAI API and assigned a score using a custom-trained logistic regression model. Then it emails me all abstracts above a certain threshold, ranked by score, at 5AM GMT. The whole setup is free to run on the cloud with Github Actions, though there is the cost of calculating OpenAI embeddings ($1.00; this is much cheaper than renting a GPU from Azure/AWS/wherever).

#### TODO
- [ ] Restructure README to show instructions for forking, retraining, and retrieval
- [ ] Include pre-computed embeddings of negative examples
- [ ] Move cutoff option to main()

#### Retraining the logistic regression model
The logistic regression model used here was trained on approximately 900 positive examples of papers that interest me, and 9000 negative examples that are a randomly assembled mix of abstracts from pubmed fetch from [the github repo associated with this manuscript](https://doi.org/10.1016/j.patter.2024.100968). The difficult part of retraining the model used to identify papers of interest is finding the positive examples; once those are obtained, however, the model can be trained as follows

```python
import pandas as pd
from sklearn.linear_model import LogisticRegression
from openai import OpenAI

client = OpenAI(api_key = userdata.get('OPENAI_API_KEY')) # you key goes here

# dda_scripts is the repo I keep with the training data (positives/negatives)
df = pd.read_csv("dda_scripts/extras/annotated_abstracts.tsv", sep="\t").dropna(subset="label")
labels, reps, prompt = [], [], []
for i, row in df.iterrows():
  prompt.append(row["abstract"])
  labels.append(row["label"])
  if i % 100 == 0 and i > 0: # to avoid sending too many tokens
    response = client.embeddings.create(
        input=prompt,
        model="text-embedding-3-small"
    )
    for d in response.data:
      reps.append(np.asarray(d.embedding[:512]))
    prompt = []
if len(prompt) > 0:
  response = client.embeddings.create(
        input=prompt,
        model="text-embedding-3-small"
    )
  for d in response.data:
    reps.append(np.asarray(d.embedding[:512]))
clf = LogisticRegression(random_state=0, max_iter=10000).fit(reps, labels)
import pickle
with open('model_openai_512.pkl','wb') as f:
  pickle.dump(clf,f)
```
