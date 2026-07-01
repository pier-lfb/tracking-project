from __future__ import annotations

from pathlib import Path

from trackers import Dataset, DatasetAsset, DatasetSplit, download_dataset


def main() -> None:
    download_dataset(
        dataset=Dataset.MOT17,
        split=DatasetSplit.VAL,
        asset=[DatasetAsset.ANNOTATIONS, DatasetAsset.DETECTIONS],
        output=str(Path("data")),
    )


if __name__ == "__main__":
    main()
