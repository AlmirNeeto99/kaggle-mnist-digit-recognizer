import pandas as pd


class MNISTSubmission:
    def __init__(self):
        self.data: pd.DataFrame = pd.DataFrame({"ImageId": [], "Label": []})

    def add(self, id, label):
        self.data.loc[len(self.data)] = [id, label]

    def save(self):
        if len(self.data) != 28000:
            print("Submission file should have 28000 lines")
            return
        self.data.to_csv("submission.csv", index=False)
