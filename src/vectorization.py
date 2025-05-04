import os
import sys
import joblib
import torch
from PIL import Image
from tqdm import tqdm
from itertools import batched
from sentence_transformers import SentenceTransformer
from sqlalchemy.orm import Session

# umożliwia import src.db jeśli uruchamiasz ten plik bezpośrednio
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from src.db import engine, Img, Base  # <- Base potrzebne do create_all

# Ustawienia modelu
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("clip-ViT-B-32").to(device)

# Ustawienia przetwarzania
MAX_IMAGES = 500  # dostosuj do zasobów
BATCH_SIZE = joblib.cpu_count(only_physical_cores=True)


def insert_images(engine, images):
    """Dodaje listę obiektów Img do bazy danych."""
    with Session(engine) as session:
        session.add_all(images)
        session.commit()


def vectorize_images(engine, model, image_paths):
    """Wektoryzuje obrazy i zapisuje embeddingi w bazie."""
    print(f" Start wektoryzacji: {len(image_paths)} obrazów")
    Base.metadata.create_all(engine)

    with tqdm(total=min(MAX_IMAGES, len(image_paths))) as pbar:
        for batch in batched(image_paths[:MAX_IMAGES], BATCH_SIZE):
            try:
                # Wczytanie i przygotowanie obrazów
                images = [Image.open(p).convert("RGB") for p in batch]

                # Wektoryzacja
                embeddings = model.encode(
                    images,
                    batch_size=BATCH_SIZE,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )

                # Utworzenie obiektów Img
                objs = [
                    Img(image_path=path, embedding=vec.tolist())
                    for path, vec in zip(batch, embeddings)
                ]

                insert_images(engine, objs)
                print(f" Dodano {len(objs)} obrazów do bazy")
                pbar.update(len(objs))

            except Exception as e:
                print(f" Błąd w batchu: {e}")
                continue


if __name__ == "__main__":
    # Przykładowe uruchomienie
    from src.utils import load_valid_image_paths

    csv_path = os.path.abspath("data/metadata/images.csv.gz")
    image_root = os.path.abspath("data/images")

    image_paths = load_valid_image_paths(csv_path, image_root, min_size=50)
    print(f" Znaleziono {len(image_paths)} poprawnych obrazów")

    vectorize_images(engine, model, image_paths)
