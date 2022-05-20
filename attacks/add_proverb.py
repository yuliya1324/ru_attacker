from .attacks.basic_attack import BasicAttack

__all__ = ["AddProverb"]


class AddProverb(BasicAttack):
    def attack(self, model, dataset):
        results = {
            "original label": [],
            "attacked label": [],
            "original premise": [],
            "original hypothesis": [],
            "transformed": [],
            "attack": []
        }
        total = 0
        correct = 0
        correct_attack = 0
        for i, row in dataset.iterrows():
            total += 1
            premise = row["premise"]
            hypothesis = row["hypothesis"]
            label = row["label"]
            prediction = model.predict(model.prepare_data(premise, hypothesis))
            if label == prediction:
                correct += 1
                transformed = premise + " Без труда не выловишь и рыбку из пруда."
                prediction = model.predict(model.prepare_data(transformed, hypothesis))
                results["original label"].append(label)
                results["attacked label"].append(prediction)
                results["original premise"].append(premise)
                results["original hypothesis"].append(hypothesis)
                results["transformed"].append(transformed)
                if label == prediction:
                    correct_attack += 1
                    results["attack"].append("failed")
                    self.print_results(results)
                else:
                    results["attack"].append("succeeded")
                    self.print_results(results)
        print(f"Accuracy before attack {round(correct/total, 2)} --> Accuracy after attack {round(correct_attack/total, 2)}")
        return results
