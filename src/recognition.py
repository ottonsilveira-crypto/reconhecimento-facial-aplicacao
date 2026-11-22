import numpy as np


class FaceRecognizer:
    """
    Compara um embedding facial com os protótipos cadastrados.
    """

    def __init__(self, database, threshold):
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("O threshold deve estar entre 0 e 1.")

        self.database = database
        self.threshold = threshold

    def recognize(self, embedding):
        query = np.asarray(embedding, dtype=np.float32).reshape(-1)

        if not np.isfinite(query).all():
            raise ValueError("Embedding de consulta inválido.")

        query_norm = np.linalg.norm(query)

        if query_norm <= 1e-12:
            raise ValueError("Embedding de consulta possui norma zero.")

        query = query / query_norm

        prototypes = self.database.get_prototypes()

        if not prototypes:
            return {
                "recognized": False,
                "reason": "empty_database",
                "name": None,
                "person_id": None,
                "similarity": None,
            }

        best_match = None
        best_similarity = -1.0

        for person in prototypes:
            prototype = person["embedding"]

            if prototype.shape != query.shape:
                raise ValueError(
                    "Dimensões incompatíveis entre os embeddings."
                )

            similarity = float(np.dot(query, prototype))

            if similarity > best_similarity:
                best_similarity = similarity
                best_match = person

        recognized = best_similarity >= self.threshold

        return {
            "recognized": recognized,
            "reason": "matched" if recognized else "unknown",
            "name": best_match["name"] if recognized else None,
            "person_id": best_match["id"] if recognized else None,
            "similarity": best_similarity,
        }
