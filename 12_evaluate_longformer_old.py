import os
import argparse
import evaluate
import random
import numpy as np
import pickle
import torch
from typing import List
from accelerate import Accelerator
from rich.progress import track
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, LongformerForSequenceClassification

from tos.tos_dataset import LongformerDataCollator, LongformerDataset, ToSDataset, Document, SceneDiscourseTree
from tos.tos_models import LongformerWithMotifsForSequenceClassification


parser = argparse.ArgumentParser()
parser.add_argument("--no-m3", action='store_true')
parser.add_argument("--no-m6", action='store_true')
parser.add_argument("--no-m9", action='store_true')


class LongformerDatasetNew(Dataset):
    def __init__(self, split: str, shuffle: bool, saved_dir: str, args):
        self.args = args
        if not os.path.isdir(saved_dir):
            os.makedirs(saved_dir)
        dataset_path = os.path.join(saved_dir, f"longformer_dataset.{split}.pkl")
        if os.path.exists(dataset_path):
            with open(dataset_path, "rb") as f:
                self.dataset = pickle.load(f)
                print(f"Dataset loaded from {dataset_path}")
        else:
            self.create_dataset(split, dataset_path)

        if shuffle:
            random.shuffle(self.dataset)

    def create_dataset(self, split: str, save_path: str):
        if split == "train":
            dataset_paths = [
                "data/hc3/hc3_train.discourse_parsed.graph_added.motif_dists.jsonl",
                "data/mage/mage_train_human.discourse_parsed.graph_added.motif_dists.jsonl",
                "data/mage/mage_train_machine.discourse_parsed.graph_added.motif_dists.jsonl",
            ]
        elif split == "valid":
            dataset_paths = [
                "data/hc3/hc3_validation.discourse_parsed.graph_added.motif_dists.jsonl",
                "data/mage/mage_validation_human.discourse_parsed.graph_added.motif_dists.jsonl",
                "data/mage/mage_validation_machine.discourse_parsed.graph_added.motif_dists.jsonl",
            ]
        elif split == "test":
            dataset_paths = [
                "data/hc3/hc3_test.discourse_parsed.graph_added.motif_dists.jsonl",
                "data/mage/mage_test_human.discourse_parsed.graph_added.motif_dists.jsonl",
                "data/mage/mage_test_machine.discourse_parsed.graph_added.motif_dists.jsonl",
            ]
        elif split == "hc3_test":
            dataset_paths = [
                "data/hc3/hc3_test.discourse_parsed.graph_added.motif_dists.jsonl",
            ]
        elif split == "mage_test":
            dataset_paths = [
                "data/mage/mage_test_human.discourse_parsed.graph_added.motif_dists.jsonl",
                "data/mage/mage_test_machine.discourse_parsed.graph_added.motif_dists.jsonl",
            ]
        elif split == "mage_ood_test":
            dataset_paths = [
                "data/mage/test_ood_set_gpt.discourse_parsed.graph_added.motif_dists.jsonl",
            ]
        elif split == "mage_ood_para_test":
            dataset_paths = [
                "data/mage/test_ood_set_gpt_para.discourse_parsed.graph_added.motif_dists.jsonl",
            ]
        else:
            raise ValueError(f"Invalid split: {split}")

        self.dataset = self.prepare_at_scene_level(
            ToSDataset.load_datasets(dataset_paths)
        )
        with open(save_path, "wb") as f:
            pickle.dump(self.dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Dataset saved at {save_path}")

    def prepare_at_scene_level(
        self, _dataset: List[Document]
    ) -> List[SceneDiscourseTree]:
        dataset = []
        for document in _dataset:
            label = document.label
            for scene in document.scene_discourse_trees.values():
                if scene is None:
                    continue
                m3_mf = np.asarray(scene.motif_dists["m3"].mf)
                m3_wad = np.asarray(scene.motif_dists["m3"].wad)
                m6_mf = np.asarray(scene.motif_dists["m6"].mf)
                m6_wad = np.asarray(scene.motif_dists["m6"].wad)
                m9_mf = np.asarray(scene.motif_dists["m9"].mf)
                m9_wad = np.asarray(scene.motif_dists["m9"].wad)
                assert m3_mf.shape == m3_wad.shape
                assert m6_mf.shape == m6_wad.shape
                assert m9_mf.shape == m9_wad.shape
                m3_feats = np.zeros(m3_mf.shape[0] * 2, dtype=np.float32)
                if not self.args.no_m3:
                    m3_feats[::2] += m3_mf
                    m3_feats[1::2] += m3_wad
                m6_feats = np.zeros(m6_mf.shape[0] * 2, dtype=np.float32)
                if not self.args.no_m6:
                    m6_feats[::2] += m6_mf
                    m6_feats[1::2] += m6_wad
                m9_feats = np.zeros(m9_mf.shape[0] * 2, dtype=np.float32)
                if not self.args.no_m9:
                    m9_feats[::2] += m9_mf
                    m9_feats[1::2] += m9_wad
                motif_dists = np.concatenate([m3_feats, m6_feats, m9_feats], axis=0)
                sample = {
                    "text": scene.text,
                    "label": label,
                    "motif_dists": motif_dists,
                }
                dataset.append(sample)
        return dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx) -> Document:
        return self.dataset[idx]


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

    test_dataset = LongformerDatasetNew(
        split=testset_name, shuffle=False, saved_dir="data", args=args
    )
    data_loader = DataLoader(
        test_dataset,
        batch_size=8,
        collate_fn=LongformerDataCollator(
            tokenizer=tokenizer, padding="longest", add_motif=add_motif
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
