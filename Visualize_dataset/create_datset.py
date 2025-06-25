from pathlib import Path
import argparse
import shutil
import random

def gen_dataset(img_folder_path, name_dataset, cls, train_pct=0.7, valid_pct=0.2):
    base_path = Path(name_dataset)
    for split in ["train", "valid", "test"]:
        (base_path / split / "images").mkdir(parents=True, exist_ok=True)
        (base_path / split / "labels").mkdir(parents=True, exist_ok=True)

    img_paths = list(Path(img_folder_path).glob("*.jpg")) + list(Path(img_folder_path).glob("*.png"))
    random.shuffle(img_paths)

    total = len(img_paths)
    n_train = int(total * train_pct)
    n_valid = int(total * valid_pct)

    splits = {
        "train": img_paths[:n_train],
        "valid": img_paths[n_train:n_train + n_valid],
        "test": img_paths[n_train + n_valid:]
    }

    for split, images in splits.items():
        for img_path in images:
            shutil.copy(img_path, base_path / split / "images" / img_path.name)
            label_path = img_path.with_suffix('.txt')
            if label_path.exists():
                shutil.copy(label_path, base_path / split / "labels" / label_path.name)

    yaml_path = base_path / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write(f"train: {base_path / 'train' / 'images'}\n")
        f.write(f"val: {base_path / 'valid' / 'images'}\n")
        f.write(f"test: {base_path / 'test' / 'images'}\n\n")
        f.write(f"nc: {len(cls)}\n")
        f.write(f"names: {cls}\n")

    print(f"Dataset generado en: {base_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_folder_img", type=str, required=True, help="Path to the folder images")
    parser.add_argument("--output", type=str, default="output", help="Path to the output folder")
    parser.add_argument("--data_cls", nargs="*", default=[], help="class lists")
    parser.add_argument("--train_pct", type=int, default=0.7, help="train porcentage of image")
    parser.add_argument("--valid_pct", type=int, default=0.2, help="valid porcentage of image")

    args = parser.parse_args()
    gen_dataset(args.path_folder_img, args.output, args.data_cls, args.train_pct, args.valid_pct)

