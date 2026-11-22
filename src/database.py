import sqlite3
import uuid

from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


class FaceDatabase:
    """
    Banco SQLite local para cadastro de identidades e embeddings faciais.
    As fotografias originais não são armazenadas.
    """

    def __init__(self, db_path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._initialize()

    @contextmanager
    def _connection(self):
        connection = sqlite3.connect(self.db_path)
        connection.execute("PRAGMA foreign_keys = ON")

        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self):
        with self._connection() as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS people (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)

            connection.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    person_id TEXT PRIMARY KEY,
                    vector BLOB NOT NULL,
                    dimension INTEGER NOT NULL,

                    FOREIGN KEY (person_id)
                        REFERENCES people(id)
                        ON DELETE CASCADE
                )
            """)

    @staticmethod
    def _build_prototype(embeddings):
        vectors = np.asarray(embeddings, dtype=np.float32)

        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)

        if vectors.ndim != 2 or len(vectors) == 0:
            raise ValueError("Nenhum embedding válido foi fornecido.")

        if not np.isfinite(vectors).all():
            raise ValueError("Os embeddings contêm valores inválidos.")

        norms = np.linalg.norm(vectors, axis=1, keepdims=True)

        if np.any(norms <= 1e-12):
            raise ValueError("Um ou mais embeddings possuem norma zero.")

        # Normaliza cada embedding antes de calcular a média.
        vectors = vectors / norms

        prototype = vectors.mean(axis=0)

        prototype_norm = np.linalg.norm(prototype)

        if prototype_norm <= 1e-12:
            raise ValueError("Não foi possível construir o protótipo.")

        prototype = prototype / prototype_norm

        return prototype.astype(np.float32)

    def add_person(self, name, embeddings):
        name = name.strip()

        if not name:
            raise ValueError("O nome não pode ficar vazio.")

        prototype = self._build_prototype(embeddings)

        person_id = str(uuid.uuid4())

        created_at = datetime.now(
            timezone.utc
        ).isoformat(timespec="seconds")

        vector_bytes = prototype.astype("<f4").tobytes()

        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO people (id, name, created_at)
                VALUES (?, ?, ?)
                """,
                (person_id, name, created_at),
            )

            connection.execute(
                """
                INSERT INTO embeddings (person_id, vector, dimension)
                VALUES (?, ?, ?)
                """,
                (person_id, vector_bytes, len(prototype)),
            )

        return person_id

    def list_people(self):
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT id, name, created_at
                FROM people
                ORDER BY name COLLATE NOCASE
                """
            ).fetchall()

        return [
            {
                "id": row[0],
                "name": row[1],
                "created_at": row[2],
            }
            for row in rows
        ]

    def get_prototypes(self):
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT people.id, people.name,
                       embeddings.vector, embeddings.dimension
                FROM people
                INNER JOIN embeddings
                    ON people.id = embeddings.person_id
                ORDER BY people.name COLLATE NOCASE
                """
            ).fetchall()

        results = []

        for person_id, name, vector_bytes, dimension in rows:
            vector = np.frombuffer(
                vector_bytes,
                dtype="<f4",
            ).copy()

            if len(vector) != dimension:
                raise ValueError(
                    f"Embedding inválido para a pessoa {person_id}."
                )

            if not np.isfinite(vector).all():
                raise ValueError(
                    f"Embedding contém valores inválidos: {person_id}."
                )

            results.append({
                "id": person_id,
                "name": name,
                "embedding": vector,
            })

        return results

    def delete_person(self, person_id):
        with self._connection() as connection:
            cursor = connection.execute(
                "DELETE FROM people WHERE id = ?",
                (person_id,),
            )

            deleted = cursor.rowcount > 0

        return deleted
