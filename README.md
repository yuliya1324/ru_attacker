# Robustness Evaluation of Pre-trained Language Models in the Russian Language

This is a repo with experiments for *Robustness Evaluation of Pre-trained Language Models in the Russian Language* and a tool `ru_attacker` for attacking Russian NLP models

### Installation
```
pip install ru_attacker
```
### Usage example

Set model
```
>>> from ru_attacker.models import RobertaModel
>>> roberta_checkpoints = "Roberta_checkpoints"
>>> ruRoberta = RobertaModel(roberta_checkpoints)
```

Set dataset
```
>>> from ru_attacker.models.set_dataset import get_data
>>> data_dir = "TERRa/val.jsonl" 
>>> data = get_data(data_dir)
```

Set attack
```
>>> from ru_attacker.ru_attacker.attacks import WordOrder
>>> word_order_attack = WordOrder()
```

Attack model and view results
```
>>> results = word_order_attack.attack(ruT5, data)
                  [Succeeded / Failed / Skipped / Total] 0 / 1 / 0 / 1:
                  entailment --> not_entailment
                  original premise: """������� ����� ������������� ��������, ��� ��� �������� ����������� ����������� ����������"", - ���������� � ���������."
                  original hypothesis: �������� ����������� �� ���������.
                  
                  transformed: �� ����������� �������� ��������� .
```
Convert results to DataFrame
```
>>> dataframe = pd.DataFrame(results)
```

### Experiments

All the data used in experiments and the results are in 
[`data`](https://github.com/yuliya1324/ru_attacker/tree/main/data) 
folder ([`TERRa`](https://github.com/yuliya1324/ru_attacker/tree/main/data/TERRa) and 
[`results`](https://github.com/yuliya1324/ru_attacker/tree/main/data/results) correspondingly).

All experiments can be reproduced in [`Experiments.ipynb`](https://github.com/yuliya1324/ru_attacker/blob/main/Experiments.ipynb).

Models checkpoints are available via:
- [ruT5-large](https://drive.google.com/file/d/1NmyPu_VCgR4IO3PIcyV4BHMShYGhqtSJ/view?usp=sharing)
- [ruRoberta-large](https://drive.google.com/file/d/1llS2LbnW9KREFAHGCS1EteIUOKE6WO7U/view?usp=sharing)

