from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from CustomDataset import CustomDataset, Cap2seq


def get_loader(root_folder, annotation_file, transform, batch_size=32, num_workers=8, shuffle=True, pin_memory=True):
    # init dataset
    dataset = CustomDataset(root_folder, annotation_file, transform=transform)

    pad_idx = dataset.vocab.stoi["<PAD>"]
    # creating a DataLoader from torch lib
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        collate_fn=Cap2seq(pad_idx=pad_idx),
    )

    return loader, dataset


