from torch.utils.data import DataLoader

from backend.data.tcir_dataset import TCIRDataset


H5_PATH = "backend/data/raw/tcir/Cyclone_Images.h5"

TRAIN_MANIFEST = "backend/data/labels/tcir_train.csv"


def test_channels(channels):

    dataset = TCIRDataset(
        manifest_path=TRAIN_MANIFEST,
        h5_path=H5_PATH,
        channels=channels,
    )

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
    )

    batch = next(iter(loader))

    print("\nChannels:", channels)

    print(
        "Image shape:",
        batch["image"].shape
    )

    print(
        "Image dtype:",
        batch["image"].dtype
    )

    print(
        "Image min:",
        batch["image"].min().item()
    )

    print(
        "Image max:",
        batch["image"].max().item()
    )

    print(
        "Wind:",
        batch["wind_kt"]
    )

    print(
        "Pressure:",
        batch["pressure_hpa"]
    )

    print(
        "Size:",
        batch["size_nmi"]
    )


if __name__ == "__main__":

    test_channels(
        ("IR", "WV", "VIS", "PMW")
    )

    test_channels(
        ("IR", "PMW")
    )