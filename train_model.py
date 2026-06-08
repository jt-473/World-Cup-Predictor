import pathlib

from predictor import train_and_save_model

ROOT = pathlib.Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "results.csv"


def main() -> None:
    print("Training World Cup match outcome predictor...")
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing data file. Download the Kaggle dataset and save it to {DATA_PATH}"
        )

    train_and_save_model(DATA_PATH)


if __name__ == "__main__":
    main()
