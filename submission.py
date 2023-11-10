from MNISTClassifier import MNISTClassifier
import torch


device = "cuda" if torch.cuda.is_available() else "cpu"
classifier = torch.load("./models/mnist-1699530821.564436-99.7.pth").to(device)

print(classifier)

classifier.eval()


from MNISTDataset import MNISTDatasetReader, MNISTDataset


dataset = MNISTDatasetReader("./data/test.csv", 1).getDatasets()
dataset = MNISTDataset(dataset, False)

from torch.utils.data import DataLoader

loader = DataLoader(dataset, 10)

from MNISTSubmission import MNISTSubmission

sub = MNISTSubmission()

for i, image in enumerate(loader):
    im = image.to(device)
    pred = classifier(im)

    pred = pred.argmax(1)
    for idx, p in enumerate(pred):
        p = p.item()
        sub.add((10 * i + idx) + 1, p)

sub.save()
