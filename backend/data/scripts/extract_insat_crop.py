import h5py
import numpy as np
from pyproj import CRS, Transformer
from PIL import Image


FILE = (
    r"backend\data\raw\insat\3DIMG_L1C_ASIA_MER\2019\03MAY"
    r"\3DIMG_03MAY2019_1600_L1C_ASIA_MER_V01R00.h5"
)

LAT = 21.6
LON = 86.9

CROP_SIZE = 128
OUTPUT = r"backend\data\raw\insat\test_fani_crop.png"


crs_geo = CRS.from_epsg(4326)

crs_insat = CRS.from_proj4(
    "+proj=merc "
    "+lon_0=75 "
    "+lat_ts=17.75 "
    "+datum=WGS84"
)

transformer = Transformer.from_crs(
    crs_geo,
    crs_insat,
    always_xy=True
)


with h5py.File(FILE, "r") as f:

    x = f["X"][:]
    y = f["Y"][:]

    tir = f["IMG_TIR1"][0]

    # Convert FANI lat/lon → projected coordinates
    px, py = transformer.transform(LON, LAT)

    col = int(np.abs(x - px).argmin())
    row = int(np.abs(y - py).argmin())

    print("FANI pixel:")
    print("row:", row)
    print("col:", col)

    half = CROP_SIZE // 2

    r1 = row - half
    r2 = row + half
    c1 = col - half
    c2 = col + half

    crop = tir[r1:r2, c1:c2]

    print("Crop shape:", crop.shape)

    # Remove fill value
    fill = int(f["IMG_TIR1"].attrs["_FillValue"][0])

    crop = crop.astype(np.float32)

    crop[crop == fill] = np.nan

    # Normalize only for visualization
    lo = np.nanpercentile(crop, 2)
    hi = np.nanpercentile(crop, 98)

    crop = np.clip((crop - lo) / (hi - lo), 0, 1)

    crop = np.nan_to_num(crop)

    image = (crop * 255).astype(np.uint8)

    Image.fromarray(image).save(OUTPUT)

    print("Saved:", OUTPUT)