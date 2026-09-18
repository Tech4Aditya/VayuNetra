import h5py
import numpy as np
from pyproj import CRS, Transformer


FILE = (
    r"backend\data\raw\insat\3DIMG_L1C_ASIA_MER\2019\03MAY"
    r"\3DIMG_03MAY2019_1600_L1C_ASIA_MER_V01R00.h5"
)

# INSAT-3D ASIA_MER projection taken directly from the HDF5 metadata
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


def latlon_to_pixel(lat, lon, x, y):
    """
    Convert geographic latitude/longitude to nearest INSAT pixel.
    """

    px, py = transformer.transform(lon, lat)

    col = int(np.abs(x - px).argmin())
    row = int(np.abs(y - py).argmin())

    return row, col, px, py


with h5py.File(FILE, "r") as f:

    x = f["X"][:]
    y = f["Y"][:]

    print("INSAT GRID")
    print("X:", x.shape, x.min(), x.max())
    print("Y:", y.shape, y.min(), y.max())

    # FANI ground-truth positions from your label file
    test_points = [
        (16.8, 84.8, "FANI 2019-05-02 06:00"),
        (17.2, 84.8, "FANI 2019-05-02 09:00"),
        (17.6, 84.8, "FANI 2019-05-02 12:00"),
        (18.3, 85.0, "FANI 2019-05-02 18:00"),
        (19.1, 85.5, "FANI 2019-05-03 00:00"),
        (20.2, 85.9, "FANI 2019-05-03 06:00"),
    ]

    print("\nFANI PIXEL LOCATIONS")
    print("-" * 70)

    for lat, lon, label in test_points:

        row, col, px, py = latlon_to_pixel(
            lat, lon, x, y
        )

        print(
            f"{label:28s} "
            f"lat={lat:5.1f} lon={lon:5.1f} "
            f"-> row={row:4d} col={col:4d} "
            f"X={px:10.1f} Y={py:10.1f}"
        )

        if not (0 <= row < len(y) and 0 <= col < len(x)):
            print("  WARNING: outside image!")