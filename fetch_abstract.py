import os, re

with open("new_abstract.tsv", "w") as outfile:
    for i, file in enumerate(
        os.listdir(
            "/Users/diegoda/Library/Mobile Documents/iCloud~md~obsidian/Documents/Unfiltered/Sorted_notes/Raw_data/Paper_notes/"
        )
    ):
        if "__" not in file:
            continue
        doi = file.replace("__", "/")[:-3]
        cmd = (
            f"curl -s https://api.crossref.org/works/{doi} | jq -r '.message.abstract'"
        )
        # cmd = f"curl -s https://api.crossref.org/works/{doi}"
        # os.system(f"echo {file} && echo {doi}")
        for line in os.popen(cmd):
            line = re.sub(r"</?jats:[^>]+>", "", line)
            line = re.sub("^Abstract", "", line)
            if len(line) < 100:
                continue
            outfile.write("\t".join([doi, "NULL", line.replace("\n", ""), "1\n"]))
