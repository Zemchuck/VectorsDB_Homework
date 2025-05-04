import matplotlib.pyplot as plt
from sqlalchemy import select
from sqlalchemy.orm import Session
from PIL import Image
from src.db import Img


class ImageSearch:
    def __init__(self, engine, model):
        self.engine = engine
        self.model = model

    def __call__(self, image_description: str, k: int):
        images = self.find_similar_images(image_description, k)
        self.display_images(images)

    def find_similar_images(self, image_description: str, k: int):
        embedding = self.model.encode(image_description).tolist()
        with Session(self.engine) as session:
            query = (
                select(Img).order_by(Img.embedding.cosine_distance(embedding)).limit(k)
            )
            results = session.execute(query).scalars().all()
            print(f"🔎 Znaleziono {len(results)} podobnych obrazów")
            return [img.image_path for img in results]

    def display_images(self, image_paths):
        if not image_paths:
            print("Brak wyników- nie znaleziono żadnych podobnych obrazów.")
            return
        fig, axes = plt.subplots(1, len(image_paths), figsize=(15, 5))
        for i, path in enumerate(image_paths):
            img = Image.open(path)
            axes[i].imshow(img)
            axes[i].axis("off")
            axes[i].set_title(f"Image {i + 1}")
        plt.show()
