from MNISTDataset import MNISTDataset


m = MNISTDataset('./data/train.csv')

t = MNISTDataset('./data/test.csv', False)

print('Size of train:',len(m))
print('Size of test:', len(t))

print(m[10])
print(t[0])
