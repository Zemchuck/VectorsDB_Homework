from sqlalchemy.engine import URL
from sqlalchemy import create_engine, Integer, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from pgvector.sqlalchemy import Vector

db_url = URL.create(
    drivername="postgresql+psycopg",
    username="postgres",
    password="password",
    host="localhost",
    port=5555,
    database="similarity_search_service_db",
)

engine = create_engine(db_url)


class Base(DeclarativeBase):
    pass


class Img(Base):
    __tablename__ = "images"
    __table_args__ = {"extend_existing": True}

    VECTOR_LENGTH = 512

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    image_path: Mapped[str] = mapped_column(String(256))
    embedding: Mapped[list[float]] = mapped_column(Vector(VECTOR_LENGTH))


Base.metadata.create_all(engine)
