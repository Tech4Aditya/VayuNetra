import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class TCIRTemporalDataset(Dataset):

    CHANNELS = {
        "IR": 0,
        "WV": 1,
        "VIS": 2,
        "PMW": 3
    }

    def __init__(
        self,
        manifest_path,
        h5_path,
        channels=("IR", "PMW")
    ):
        self.df = pd.read_csv(manifest_path)

        self.h5_path = h5_path
        self.h5_file = None

        # Validate requested channels
        for channel in channels:
            if channel not in self.CHANNELS:
                raise ValueError(
                    f"Unknown channel: {channel}. "
                    f"Available channels: {list(self.CHANNELS.keys())}"
                )

        # Convert channel names to HDF5 indices
        self.channels = [
            self.CHANNELS[channel]
            for channel in channels
        ]

        # Four temporal frames
        self.frame_columns = [
            f"frame_{i}_h5_index"
            for i in range(4)
        ]

        # Make sure manifest contains required columns
        required_columns = (
            self.frame_columns
            + [
                "target_wind_kt",
                "target_pressure_hpa",
                "target_size_nmi",
                "cyclone_id",
                "target_timestamp"
            ]
        )

        missing = [
            col
            for col in required_columns
            if col not in self.df.columns
        ]

        if missing:
            raise ValueError(
                f"Missing columns in manifest: {missing}"
            )

    def _open_h5(self):
        """
        Open HDF5 lazily.

        This is important because DataLoader workers
        should not inherit an already-open HDF5 handle.
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

        frames = []

        # -------------------------------------------------
        # Load the four consecutive satellite frames
        # -------------------------------------------------
        for column in self.frame_columns:

            h5_index = int(row[column])

            image = self.h5_file["Images"][h5_index]

            # HDF5 shape:
            # [128, 128, 4]
            #
            # Select requested channels:
            # [128, 128, 2] for IR + PMW
            image = image[:, :, self.channels]

            image = image.astype(np.float32)

            # Original TCIR values are 0-255
            # Normalize to 0-1
            image /= 255.0

            # Protect against NaN / Inf
            image = np.nan_to_num(
                image,
                nan=0.0,
                posinf=1.0,
                neginf=0.0
            )

            # HWC → CHW
            #
            # [128, 128, 2]
            #       ↓
            # [2, 128, 128]
            image = np.transpose(
                image,
                (2, 0, 1)
            )

            frames.append(image)

        # -------------------------------------------------
        # Stack temporal frames
        # -------------------------------------------------
        #
        # [frame0, frame1, frame2, frame3]
        #
        # Result:
        # [4, channels, 128, 128]
        #
        frames = np.stack(frames)

        frames = torch.from_numpy(
            frames.copy()
        )

        # -------------------------------------------------
        # Targets
        # -------------------------------------------------

        wind = torch.tensor(
            float(row["target_wind_kt"]),
            dtype=torch.float32
        )

        pressure = torch.tensor(
            float(row["target_pressure_hpa"]),
            dtype=torch.float32
        )

        size = torch.tensor(
            float(row["target_size_nmi"]),
            dtype=torch.float32
        )

        return {
            "frames": frames,

            "wind_kt": wind,

            "pressure_hpa": pressure,

            "size_nmi": size,

            "cyclone_id": str(
                row["cyclone_id"]
            ),

            "timestamp": str(
                row["target_timestamp"]
            )
        }

    def __del__(self):

        try:
            if self.h5_file is not None:
                self.h5_file.close()
        except Exception:
            pass