import os
import shutil
from pathlib import Path

def gather_sdfs(root_dir):
    """
    Gathers all .sdf files under any `delete` subdirectories into a single `delete/` directory
    at the root, renaming files to consecutive integers (0.sdf, 1.sdf, ...).
    """
    root_path = Path(root_dir)
    target_dir = root_path / "delete"
    target_dir.mkdir(exist_ok=True)

    counter = 0
    for subdir in root_path.iterdir():
        sub_delete = subdir / "delete"
        if subdir.is_dir() and sub_delete.exists() and sub_delete.is_dir():
            for sdf_file in sorted(sub_delete.glob("*.sdf")):
                dest_file = target_dir / f"{counter}.sdf"
                shutil.copy(sdf_file, dest_file)
                counter += 1

    print(f"Gathered {counter} .sdf files into {target_dir}")

if __name__ == "__main__":
    root = input("Enter path to the root output directory: ").strip()
    gather_sdfs(root)

# (3.11.11) (base) ziseok@a100-6-002:/mnt/nas5/AIBL-Research/ziseok/Delete$ python gather_sdfs.py 
# Enter path to the root output directory: outputs/adrb1_rotated_20250719_131844
# Gathered 108 .sdf files into outputs/adrb1_rotated_20250719_131844/delete