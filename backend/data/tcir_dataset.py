import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class TCIRDataset(Dataset):
    """
    Lazy-loading dataset for TCIR satellite imagery.

    HDF5:
        Images -> (N, 128, 128, 4)

    Channels:
        0 -> IR
        1 -> WV
        2 -> VIS
        3 -> PMW

    Returns:
        image
        wind_kt
        pressure_hpa
        size_nmi
        cyclone_id
        timestamp
    """

    CHANNELS = {
        "IR": 0,
        "WV": 1,
        "VIS": 2,
        "PMW": 3,
    }

    def __init__(
        self,
        manifest_path,
        h5_path,
        channels=("IR", "WV", "VIS", "PMW"),
    ):
        self.df = pd.read_csv(manifest_path)

        self.h5_path = h5_path
        self.h5_file = None

        # Validate channels
        for channel in channels:
            if channel not in self.CHANNELS:
                raise ValueError(
                    f"Unknown channel: {channel}. "
                    f"Available channels: {list(self.CHANNELS.keys())}"
                )

        self.channels = [
            self.CHANNELS[channel]
            for channel in channels
        ]

    def _open_h5(self):
        """
        Open HDF5 lazily.

        This prevents the 5+ GB file from being loaded
        into RAM at dataset creation time.
        """
        if self.h5_file is None:
            self.h5_file = h5py.File(
                self.h5_path,
                "r"
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        self._open_h5()

        row = self.df.iloc[idx]

        # Original index inside the HDF5 file
        h5_index = int(row["h5_index"])

        # Read ONLY one image
        image = self.h5_file["Images"][h5_index]

        # Select requested channels
        image = image[:, :, self.channels]

        # Convert to float32
        image = image.astype(np.float32)

        # TCIR image values are scaled approximately [0, 255]
        # Normalize to [0, 1]
        image /= 255.0

        # Handle invalid values safely
        image = np.nan_to_num(
            image,
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        )

        # HWC -> CHW
        image = np.transpose(
            image,
            (2, 0, 1)
        )

        # Make contiguous before converting to Tensor
        image = torch.from_numpy(
            image.copy()
        )

        # Labels
        wind = torch.tensor(
            float(row["wind_kt"]),
            dtype=torch.float32
        )

        pressure = torch.tensor(
            float(row["pressure_hpa"]),
            dtype=torch.float32
        )

        size = torch.tensor(
            float(row["size_nmi"]),
            dtype=torch.float32
        )

        return {
            "image": image,
            "wind_kt": wind,
            "pressure_hpa": pressure,
            "size_nmi": size,
            "cyclone_id": str(row["cyclone_id"]),
            "timestamp": str(row["timestamp"]),
        }

    def __del__(self):
        try:
            if self.h5_file is not None:
                self.h5_file.close()
        except Exception:
            pass