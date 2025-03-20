import random

import pandas as pd
from rich.progress import track

from tos.tos_dataset import ToSDataset

random.seed(42)


dataset_paths = [
    "data/mage/machine-dev.csv",
    "data/mage/machine-test.csv",
]

argu_dev = pd.read_csv(dataset_paths[0], header=0)
argu_test = pd.read_csv(dataset_paths[1], header=0)

print(argu_dev.head(2))
print(argu_test.head(2))

tos_dataset = ToSDataset(
    dmrst_parser_dir="/content/DMRST_Parser",
    batch_size=1024,
    gpu_id=0,
    motif_dir="data/motifs",
)


def process_df(df):
    dataset = []
    for i, row in track(df.iterrows(), total=len(df), description="Processing..."):
        text = row["text"]
        label = 0
        source = row["model"]
        document = tos_dataset.parse_document_discourse(
            document=text,
            source=source,
            label=label,
            filter_none=True,
            add_graph=True,
            add_motif_dists=True,
        )
        dataset.append(document)
    return dataset


dataset_argu_dev = process_df(argu_dev)
tos_dataset.save_dataset_as_jsonl(
    dataset_argu_dev,
    "data/mage/argu_dev.discourse_parsed.graph_added.motif_dists.jsonl",
)

dataset_argu_test = process_df(argu_test)
tos_dataset.save_dataset_as_jsonl(
    dataset_argu_test,
    "data/mage/argu_test.discourse_parsed.graph_added.motif_dists.jsonl",
)
