import os
from typing import List, Dict, Union, Optional
import argparse
import evaluate
import torch
import numpy as np
from dataclasses import dataclass
from accelerate import Accelerator
from rich.progress import track
from safetensors.torch import load_file
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, LongformerForSequenceClassification, PreTrainedTokenizerBase
from transformers.tokenization_utils_base import PaddingStrategy

from tos.tos_dataset import LongformerDataCollator, LongformerDataset
from tos.tos_models import LongformerWithMotifsForSequenceClassification


parser = argparse.ArgumentParser()
parser.add_argument("--no-m3", action='store_true')
parser.add_argument("--no-m6", action='store_true')
parser.add_argument("--no-m9", action='store_true')


@dataclass
class LongformerDataCollatorNew:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    pad_to_multiple_of: Optional[int] = None
    add_motif: bool = False
    no_m3: bool = False
    no_m6: bool = False
    no_m9: bool = False

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        text_data = []
        labels = []
        motif_dists = []

        for data in features:
            text_data.append(data["text"])
            labels.append(data["label"])
            if self.add_motif:
                motif_dists.append(data["motif_dists"])

        batch = self.tokenizer(
            text_data,
            padding=self.padding,
            return_tensors="pt",
            truncation=True,
        )
        batch["labels"] = torch.tensor(labels)
        if self.add_motif:
            batch["motif_dists"] = torch.tensor(
                np.stack(motif_dists), dtype=torch.float
            )
        return batch


def evaluate_longformer(testset_name: str, model_path: str, add_motif: bool, args):
    accelerator = Accelerator()
    accelerator.print(f"\n\n------{testset_name}-------")

    tokenizer = AutoTokenizer.from_pretrained(
        "allenai/longformer-base-4096", use_fast=True
    )

    metric = evaluate.combine(
        ["accuracy", "f1", "precision", "recall", "BucketHeadP65/confusion_matrix"]
    )

    if add_motif:
        model = LongformerWithMotifsForSequenceClassification()
        state_dict = load_file(os.path.join(model_path, "model.safetensors"))
        model.load_state_dict(state_dict)
    else:
        model = LongformerForSequenceClassification.from_pretrained(
            model_path, num_labels=2
        )

    test_dataset = LongformerDataset(
        split=testset_name, shuffle=False, saved_dir="data"
    )
    data_loader = DataLoader(
        test_dataset,
        batch_size=8,
        collate_fn=LongformerDataCollatorNew(
            tokenizer=tokenizer, padding="longest", add_motif=add_motif, no_m3=args.no_m3, no_m6=args.no_m6, no_m9=args.no_m9
        ),
    )

    model, data_loader = accelerator.prepare(model, data_loader)

    for data in track(data_loader, total=len(data_loader), description="Evaluating..."):
        targets = data["labels"]
        predictions = model(**data).logits

        predictions = torch.argmax(predictions, dim=1)
        all_predictions, all_targets = accelerator.gather_for_metrics(
            (predictions, targets)
        )
        metric.add_batch(predictions=all_predictions, references=all_targets)

    accelerator.print(metric.evaluation_modules[0].__len__())
    accelerator.print(metric.compute())
    accelerator.print("-----------------------")


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    # testset_names = ["hc3_test", "mage_test", "mage_ood_test", "mage_ood_para_test"]
    testset_names = ["mage_ood_test", "mage_ood_para_test"]
    for testset_name in testset_names:
        evaluate_longformer(
            testset_name,
            model_path="results/longformer_base_motif/checkpoint-1300",
            add_motif=True,
            args=args
        )
