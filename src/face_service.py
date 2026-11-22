import numpy as np

from src.database import FaceDatabase
from src.face_pipeline import FacePipeline
from src.recognition import FaceRecognizer
from src.logger import setup_logger


class FaceService:
    """
    Coordena o processamento facial, o banco de dados
    e o reconhecimento de identidades.
    """

    def __init__(
        self,
        model_path,
        yunet_path,
        db_path,
        threshold=0.3724803,
        device=None,
    ):
        self.logger = setup_logger()
        self.logger.info("Serviço de reconhecimento facial inicializado.")
        self.pipeline = FacePipeline(
            model_path=model_path,
            yunet_path=yunet_path,
            device=device,
        )

        self.database = FaceDatabase(db_path)

        self.recognizer = FaceRecognizer(
            database=self.database,
            threshold=threshold,
        )

    def enroll(self, name, image_paths, confirm_duplicate=False):
        """
        Cadastra uma pessoa usando uma ou mais fotografias.

        Não armazena as fotografias originais.
        """

        if not name or not name.strip():
            raise ValueError("Informe um nome válido.")

        if not image_paths:
            raise ValueError("Selecione pelo menos uma fotografia.")

        embeddings = []

        for image_path in image_paths:
            result = self.pipeline.process_image(image_path)
            embeddings.append(result["embedding"])

        vectors = np.asarray(embeddings, dtype=np.float32)
        prototype = vectors.mean(axis=0)

        norm = np.linalg.norm(prototype)

        if norm <= 1e-12:
            raise ValueError("Não foi possível construir o protótipo.")

        prototype = prototype / norm

        # Verifica se a identidade já parece estar cadastrada.
        duplicate = self.recognizer.recognize(prototype)

        if duplicate["recognized"] and not confirm_duplicate:
            raise ValueError(
                "Possível cadastro duplicado: "
                f"{duplicate['name']} "
                f"(similaridade: {duplicate['similarity']:.4f}). "
                "Confirme se deseja continuar."
            )

        person_id = self.database.add_person(name, embeddings)

        self.logger.info(
            "Cadastro realizado. ID interno: %s. Quantidade de imagens: %d.",
            person_id,
            len(embeddings)
        )

        return {
            "person_id": person_id,
            "name": name.strip(),
            "image_count": len(embeddings),
        }

    def recognize_image(self, image_path):
        """Reconhece uma pessoa a partir de uma fotografia."""

        result = self.pipeline.process_image(image_path)

        recognition = self.recognizer.recognize(
            result["embedding"]
        )

        self.logger.info(
            "Reconhecimento executado. Resultado: %s. Similaridade: %.4f.",
            recognition["reason"],
            recognition["similarity"]
        )
        
        recognition["detection_confidence"] = (
            result["detection_confidence"]
        )

        return recognition

    def list_people(self):
        return self.database.list_people()

    def delete_person(self, person_id):
        deleted = self.database.delete_person(person_id)

        if deleted:
            self.logger.info("Cadastro excluído. ID interno: %s.", person_id)
        else:
            self.logger.warning(
                "Tentativa de exclusão de cadastro inexistente. ID: %s.",
                person_id
            )

        return deleted
