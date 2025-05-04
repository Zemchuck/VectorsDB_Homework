def load_valid_image_paths(csv_path: str, image_root: str, min_size: int = 100):
    import os
    import pandas as pd
    from PIL import Image

    df = pd.read_csv(csv_path)
    valid_paths = []
    skipped = 0
    missing = 0
    too_small = 0

    for rel_path in df["path"]:
        full_path = os.path.join(image_root, "small", rel_path)

        if not os.path.exists(full_path):
            missing += 1
            continue

        try:
            with Image.open(full_path) as img:
                if img.width >= min_size and img.height >= min_size:
                    valid_paths.append(full_path)
                else:
                    too_small += 1
        except Exception:
            skipped += 1

    print(f"Załadowano: {len(valid_paths)} obrazów")
    print(f"Za małe: {too_small} | Brakujące: {missing} |  Błędy: {skipped}")
    return valid_paths
