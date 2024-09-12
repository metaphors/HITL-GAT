import os

input_file_dir_1 = "Adv.TSAttacker.Tibetan-BERT+TNCC-title"
input_file_dir_2 = "Adv.TSCheater_s.Tibetan-BERT+TNCC-title"
input_file_dir_3 = "Adv.TSCheater_w.Tibetan-BERT+TNCC-title"
input_file_dir_4 = "Adv.TSCheaterPlus.Tibetan-BERT+TNCC-title"
input_file_dir_5 = "Adv.TSTricker_s.Tibetan-BERT+TNCC-title"
input_file_dir_6 = "Adv.TSTricker_w.Tibetan-BERT+TNCC-title"
input_file_dir_7 = "Adv.TSAttacker.Tibetan-BERT+TU_SA"
input_file_dir_8 = "Adv.TSCheater_s.Tibetan-BERT+TU_SA"
input_file_dir_9 = "Adv.TSCheater_w.Tibetan-BERT+TU_SA"
input_file_dir_10 = "Adv.TSCheaterPlus.Tibetan-BERT+TU_SA"
input_file_dir_11 = "Adv.TSTricker_s.Tibetan-BERT+TU_SA"
input_file_dir_12 = "Adv.TSTricker_w.Tibetan-BERT+TU_SA"

input_file_dir_list_1 = [input_file_dir_1, input_file_dir_2, input_file_dir_3, input_file_dir_4, input_file_dir_5,
                         input_file_dir_6]
input_file_dir_list_2 = [input_file_dir_7, input_file_dir_8, input_file_dir_9, input_file_dir_10, input_file_dir_11,
                         input_file_dir_12]

output_file_1 = "Adv.TNCC-title.txt"
output_file_2 = "Adv.TU_SA.txt"

for input_file_dir in input_file_dir_list_1:
    with open(os.path.join("data", input_file_dir, "log_annotated.txt"), "r") as input_file:
        for line in input_file:
            if line[-2] == "4" or line[-2] == "5":
                label = line.strip().split("\t")[0]
                orig = line.strip().split("\t")[1]
                adv = line.strip().split("\t")[2]
                score = line.strip().split("\t")[3]
                attack = input_file_dir.split(".")[1]
                with open(os.path.join("data", "Dataset.AdvTS", output_file_1), "a") as output_file:
                    output_file.write(label + "\t" + orig + "\t" + adv + "\t" + score + "\t" + attack + "\n")

for input_file_dir in input_file_dir_list_2:
    with open(os.path.join("data", input_file_dir, "log_annotated.txt"), "r") as input_file:
        for line in input_file:
            if line[-2] == "4" or line[-2] == "5":
                label = line.strip().split("\t")[0]
                orig = line.strip().split("\t")[1]
                adv = line.strip().split("\t")[2]
                score = line.strip().split("\t")[3]
                attack = input_file_dir.split(".")[1]
                with open(os.path.join("data", "Dataset.AdvTS", output_file_2), "a") as output_file:
                    output_file.write(label + "\t" + orig + "\t" + adv + "\t" + score + "\t" + attack + "\n")

from transformers import XLMRobertaForSequenceClassification
from transformers import XLMRobertaTokenizer
from datasets import load_dataset
import torch


def test_model_robustness(model_path, num_labels, dataset_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = XLMRobertaForSequenceClassification.from_pretrained(model_path, num_labels=num_labels).to(device)
    tokenizer = XLMRobertaTokenizer.from_pretrained(model_path)

    dataset = load_dataset(dataset_path, split="test")
    labels = dataset["label"]

    predictions = []
    for text in dataset["text"]:
        inputs = tokenizer(text, return_tensors="pt", max_length=512, padding="max_length", truncation=True).to(device)
        with torch.no_grad():
            prediction = model(**inputs).logits.argmax().item()
            predictions.append(prediction)

    right_num = 0
    for (idx, prediction) in enumerate(predictions):
        if prediction == labels[idx]:
            right_num += 1

    with open(os.path.join("data", "Dataset.AdvTS", "results.txt"), "a") as file:
        file.write(model_path.split(".")[-1] + "\t" + str(right_num) + "\t" + str(len(predictions)) + "\n")


model_1 = "Victim.XLM-RoBERTa.CINO-small-v2+TNCC-title"
model_2 = "Victim.XLM-RoBERTa.CINO-base-v2+TNCC-title"
model_3 = "Victim.XLM-RoBERTa.CINO-large-v2+TNCC-title"
model_4 = "Victim.XLM-RoBERTa.CINO-small-v2+TU_SA"
model_5 = "Victim.XLM-RoBERTa.CINO-base-v2+TU_SA"
model_6 = "Victim.XLM-RoBERTa.CINO-large-v2+TU_SA"

dataset_loader_1 = "Adv-TNCC-title.py"
dataset_loader_2 = "Adv-TU_SA.py"

test_model_robustness(os.path.join("data", model_1), 12, os.path.join("data", "Dataset.Loader", dataset_loader_1))
test_model_robustness(os.path.join("data", model_2), 12, os.path.join("data", "Dataset.Loader", dataset_loader_1))
test_model_robustness(os.path.join("data", model_3), 12, os.path.join("data", "Dataset.Loader", dataset_loader_1))
test_model_robustness(os.path.join("data", model_4), 2, os.path.join("data", "Dataset.Loader", dataset_loader_2))
test_model_robustness(os.path.join("data", model_5), 2, os.path.join("data", "Dataset.Loader", dataset_loader_2))
test_model_robustness(os.path.join("data", model_6), 2, os.path.join("data", "Dataset.Loader", dataset_loader_2))
