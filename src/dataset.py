import os, glob
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class PairedFolderDataset(Dataset):
    def __init__(self, root, size=256):
        self.root = root
        self.A = sorted(glob.glob(os.path.join(root, "A", "*.png")))
        self.B = sorted(glob.glob(os.path.join(root, "B", "*.png")))
        assert len(self.A) == len(self.B) and len(self.A) > 0, "A/B mismatch or empty."
        bmap = {os.path.basename(p): p for p in self.B}
        self.files = [(a, bmap[os.path.basename(a)]) for a in self.A if os.path.basename(a) in bmap]
        self.tf = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        a_path, b_path = self.files[idx]
        a = Image.open(a_path).convert("RGB")
        b = Image.open(b_path).convert("RGB")
        return self.tf(a), self.tf(b), os.path.basename(a_path)
