
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torch



class ImageOnlyDataset(Dataset):
    def __init__(self, df, transform=None, label_map=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.label_map = label_map

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = row["skincap_file_path"]
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label_str = row["disease"]
        label = self.label_map[label_str] if self.label_map else label_str

        return {"image": image, "label": torch.tensor(label), "path": image_path}
