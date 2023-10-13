from MNISTDataset import MNISTDataset


m = MNISTDataset('./data/train.csv')

print(len(m))


print(m[0])
