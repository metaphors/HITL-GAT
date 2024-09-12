import os
import Levenshtein

log_path = os.path.join("data", "Adv.TSCheater_s.Tibetan-BERT+TU_SA", "log.txt")
data_path = os.path.join("data", "Adv.TSCheater_s.Tibetan-BERT+TU_SA", "data_filtered.txt")
id2label = {0: "Negative", 1: "Positive"}


def filter_log(log_path, data_path, id2label):
    attack_results = []
    attack_result = {}

    with open(log_path, "r") as file:
        for line in file:
            if line != "---\n":
                elements = line.strip().split("\t")
                if elements[0] != "Orig" and elements[0] != "Adv":
                    attack_result[elements[0]] = elements[1]
                else:
                    attack_result[elements[0]] = {"text": elements[1], "id": elements[2], "id_score": elements[3]}
            else:
                if attack_result["Succeed"] == "False":
                    attack_result["Adv"] = attack_result["Orig"]
                attack_results.append(attack_result)
                attack_result = {}

    with open(data_path, "a") as file:
        for attack_result in attack_results:
            if attack_result["Succeed"] == "True" and \
                    Levenshtein.distance(attack_result["Orig"]["text"], attack_result["Adv"]["text"]) / len(
                attack_result["Orig"]["text"]) <= 0.1:
                file.write(id2label[int(attack_result["Orig"]["id"])] + "\t" + attack_result["Orig"]["text"] + "\t" +
                           attack_result["Adv"]["text"] + "\n")


filter_log(log_path, data_path, id2label)
