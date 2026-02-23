import os
import time
import shutil
from pathlib import Path


def main():
    runs_dir = Path("runs")
    if not runs_dir.exists():
        print("runs/ does not exist; nothing to archive.")
        return

    subdirs = [p for p in runs_dir.iterdir() if p.is_dir()]
    if not subdirs:
        print("No run directories to archive.")
        return

    ts = time.strftime("%Y%m%dT%H%M%S")
    archive_root = Path("runs_archive") / ts
    archive_root.mkdir(parents=True, exist_ok=True)

    moved = 0
    for d in sorted(subdirs):
        dest = archive_root / d.name
        shutil.move(str(d), str(dest))
        moved += 1

    print(f"Archived {moved} run directories to {archive_root}")


if __name__ == "__main__":
    main()

