
from pathlib import Path
from zipfile import ZipFile, BadZipFile

ZIP_FILES = [
    Path(r"C:\Users\Jigesha\Downloads\Fold3_part1.zip"),
    Path(r"C:\Users\Jigesha\Downloads\Fold3_part2.zip"),
    Path(r"C:\Users\Jigesha\Downloads\Fold4_part1.zip"),
    Path(r"C:\Users\Jigesha\Downloads\Fold4_part2.zip"),
]

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}
LABELS = {"0", "5", "10"}

total_videos = 0
total_labeled = 0
total_unlabeled = 0

for zip_path in ZIP_FILES:
    print(f"\n{'=' * 55}")
    print(f"ZIP FILE: {zip_path.name}")
    print("=" * 55)

    if not zip_path.exists():
        print("ERROR: ZIP NOT FOUND")
        continue

    try:
        with ZipFile(zip_path, "r") as archive:
            names = [
                name for name in archive.namelist()
                if Path(name).suffix.lower() in VIDEO_EXTENSIONS
                and not name.endswith("/")
            ]

            labeled = [
                name for name in names
                if Path(name).stem in LABELS
            ]

            unlabeled = [
                name for name in names
                if Path(name).stem not in LABELS
            ]

            print(f"Total video files: {len(names)}")
            print(f"Correctly named label files: {len(labeled)}")
            print(f"Files needing inspection: {len(unlabeled)}")

            print("\nLabel counts:")
            for label, class_name in [
                ("0", "Alert"),
                ("5", "Low Vigilant"),
                ("10", "Drowsy"),
            ]:
                count = sum(
                    Path(name).stem == label for name in names
                )
                print(f"  {class_name} ({label}): {count}")

            if unlabeled:
                print("\nFILES NEEDING LABEL INSPECTION:")
                for name in unlabeled:
                    print(f"  {name}")
            else:
                print("\nAll video filenames match expected labels.")

            # Check whether the ZIP archive itself passes its CRC test.
            bad_file = archive.testzip()
            if bad_file:
                print(f"\nWARNING: ZIP integrity issue: {bad_file}")
            else:
                print("\nZIP integrity check: PASSED")

            total_videos += len(names)
            total_labeled += len(labeled)
            total_unlabeled += len(unlabeled)

    except BadZipFile:
        print("ERROR: This is not a valid ZIP archive or it is corrupted.")
    except Exception as error:
        print(f"ERROR: {type(error).__name__}: {error}")

print(f"\n{'=' * 55}")
print("OVERALL SUMMARY")
print("=" * 55)
print(f"Total videos found: {total_videos}")
print(f"Videos with expected labels: {total_labeled}")
print(f"Files needing inspection: {total_unlabeled}")
print("=" * 55)

