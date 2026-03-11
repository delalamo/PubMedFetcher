import os
import pickle

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.linear_model import LogisticRegression

load_dotenv()

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# dda_scripts is the repo I keep with the training data (positives/negatives)
df = pd.read_csv("dda_scripts/extras/annotated_abstracts.tsv", sep="\t").dropna(
    subset="label"
)
labels, reps, prompt = [], [], []
for i, row in df.iterrows():
    prompt.append(row["abstract"])
    labels.append(row["label"])
    if i % 100 == 0 and i > 0:
        response = client.embeddings.create(
            input=prompt, model="text-embedding-3-small"
        )
        for d in response.data:
            reps.append(np.asarray(d.embedding[:512]))
        prompt = []
if len(prompt) > 0:
    response = client.embeddings.create(input=prompt, model="text-embedding-3-small")
    for d in response.data:
        reps.append(np.asarray(d.embedding[:512]))
clf = LogisticRegression(random_state=0, max_iter=10000).fit(reps, labels)

with open("model_openai_512.pkl", "wb") as f:
    pickle.dump(clf, f)
