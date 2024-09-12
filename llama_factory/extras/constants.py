# Source Attribution:
# The majority of the code is derived from the following source:
# - LLaMA-Factory GitHub Repository: https://github.com/hiyouga/LLaMA-Factory
# - Tag: v0.9.0
# - File: /src/llamafactory/extras/constants.py
# - Reference Paper: LlamaFactory: Unified Efficient Fine-Tuning of 100+ Language Models (Zheng et al., ACL 2024)

DATA_CONFIG = "Dataset.Info/all.json"

PRETRAINED_LANGUAGE_MODEL_LIST = ["CINO-small-v2", "CINO-base-v2", "CINO-large-v2", "Tibetan-BERT"]
ATTACKED_MODEL_LIST = ["Tibetan-BERT"]
EVALUATION_MODEL_LIST = ["CINO-small-v2, CINO-base-v2, CINO-large-v2"]
TRAIN_DATASET_LIST = ["TNCC-title.train", "TNCC-document.train", "TU_SA.train"]
TEST_DATASET_LIST = ["TNCC-title.test", "TNCC-document.test", "TU_SA.test"]
EVALUATION_DATASET_LIST = ["Adv.TNCC-title, Adv.TU_SA"]
ATTACK_METHOD_LIST = ["TSAttacker", "TSTricker_s", "TSTricker_w", "TSCheater_s", "TSCheater_w", "TSCheaterPlus"]
VICTIM_OUTPUT_DIR_LIST = ["Victim.Output"]
ADV_OUTPUT_DIR_LIST = ["Adv.Output"]
ADV_INPUT_DIR_LIST = ["Adv.TSAttacker.Tibetan-BERT+TNCC-title", "Adv.TSTricker_s.Tibetan-BERT+TNCC-title",
                      "Adv.TSTricker_w.Tibetan-BERT+TNCC-title", "Adv.TSCheater_s.Tibetan-BERT+TNCC-title",
                      "Adv.TSCheater_w.Tibetan-BERT+TNCC-title", "Adv.TSCheaterPlus.Tibetan-BERT+TNCC-title",
                      "Adv.TSAttacker.Tibetan-BERT+TU_SA", "Adv.TSTricker_s.Tibetan-BERT+TU_SA",
                      "Adv.TSTricker_w.Tibetan-BERT+TU_SA", "Adv.TSCheater_s.Tibetan-BERT+TU_SA",
                      "Adv.TSCheater_w.Tibetan-BERT+TU_SA", "Adv.TSCheaterPlus.Tibetan-BERT+TU_SA"]
METRIC_FOR_BEST_MODEL_LIST = ["F1/Macro-F1"]
