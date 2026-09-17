from pathlib import Path
import sys
import h5py


def print_attrs(obj, indent=""):
    if not obj.attrs:
        return

    print(f"{indent}Attributes:")

    for key, value in obj.attrs.items():
        try:
            print(f"{indent}  {key}: {value}")
        except Exception:
            print(f"{indent}  {key}: <unprintable>")


def inspect_group(group, indent=""):
    for name, obj in group.items():

        if isinstance(obj, h5py.Group):

            print(f"{indent}[GROUP] {name}")
            print_attrs(obj, indent + "  ")

            inspect_group(
                obj,
                indent + "  "
            )

        elif isinstance(obj, h5py.Dataset):

            print(f"{indent}[DATASET] {name}")

            print(
                f"{indent}  shape: {obj.shape}"
            )

            print(
                f"{indent}  dtype: {obj.dtype}"
            )

            print_attrs(
                obj,
                indent + "  "
            )


def main():

    if len(sys.argv) != 2:
        print(
            "Usage:\n"
            "python inspect_hdf5.py <file.h5>"
        )
        sys.exit(1)

    file_path = Path(sys.argv[1])

    if not file_path.exists():
        print(f"File not found:\n{file_path}")
        sys.exit(1)

    print("=" * 70)
    print("INSAT HDF5 INSPECTOR")
    print("=" * 70)

    print(f"\nFile:")
    print(file_path)

    print(f"\nSize:")
    print(f"{file_path.stat().st_size / (1024**2):.2f} MB")

    print("\n" + "=" * 70)
    print("HDF5 STRUCTURE")
    print("=" * 70)

    with h5py.File(file_path, "r") as h5:

        print("\nRoot attributes:")
        print_attrs(h5, "  ")

        print("\nDatasets / groups:")
        inspect_group(h5)

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()