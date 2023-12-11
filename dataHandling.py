from torch.utils.data import Dataset, DataLoader, random_split
import h5py
import torch

class CustomDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


def getDataAsTensors(filepath, numpy=False):
    with h5py.File(filepath, "r") as f:
        # Print all root level object names (aka keys) 
        # these can be group or dataset names 
        print(f"Keys: {f.keys()}")
        print(len(f["image"]))
        imgs = f["image"][:]
        signals = f["signal"][:]
    if numpy:
        return imgs, signals
    t = torch.from_numpy(imgs)
    s = torch.from_numpy(signals)

    # Print overall max and min pixel values
    print(f"Overall Max Pixel Value: {t.max()}")
    print(f"Overall Min Pixel Value: {t.min()}")

    print("imgs:", t.size(), ", signals: ",s.size())
    return t, s, t.max()

### Datasets ###
#jets - 8 726 datapoints (formerly jet-images_micro)
#jet-images_train - 628 320 datapoints
#jet-images_val - 157 080 datapoints
#jet-images_test - 87 266 datapoints
micro_path = "jet-data/jets.hdf5"
test_path = "jet-data/jet-images_test.hdf5"
val_path = "jet-data/jet-images_val.hdf5"
train_path = "jet-data/jet-images_train.hdf5"