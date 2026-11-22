from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image, ImageOps
from torchvision import transforms

from src.model import FaceEmbeddingNet


class FacePipeline:
    """
    Pipeline local de detecção, alinhamento e geração de embeddings faciais.
    """

    OUT_W = 94
    OUT_H = 125

    EYE_X = 47.6
    EYE_Y = 44.0
    EYE_DISTANCE = 44.2

    MAX_DIMENSION = 1280

    def __init__(self, model_path, yunet_path, device=None):
        self.model_path = Path(model_path)
        self.yunet_path = Path(yunet_path)

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 96), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225],
            ),
        ])

        self._load_model()
        self._load_detector()

    def _load_model(self):
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Checkpoint não encontrado: {self.model_path}"
            )

        checkpoint = torch.load(
            self.model_path,
            map_location=self.device,
            weights_only=True,
        )

        self.model = FaceEmbeddingNet(
            embedding_dim=checkpoint["embedding_dim"],
            pretrained=False,
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"],
            strict=True,
        )

        self.model.to(self.device)
        self.model.eval()

    def _load_detector(self):
        if not self.yunet_path.exists():
            raise FileNotFoundError(
                f"Modelo YuNet não encontrado: {self.yunet_path}"
            )

        self.detector = cv2.FaceDetectorYN.create(
            str(self.yunet_path),
            "",
            (320, 320),
            0.50,
            0.30,
            5000,
        )

    @staticmethod
    def read_image(image_path):
        with Image.open(image_path) as image:
            image = ImageOps.exif_transpose(image)
            return np.array(image.convert("RGB"))

    def align_face(self, image_rgb):
        height, width = image_rgb.shape[:2]

        scale = min(
            1.0,
            self.MAX_DIMENSION / max(height, width),
        )

        resized_width = round(width * scale)
        resized_height = round(height * scale)

        detection_image = cv2.resize(
            image_rgb,
            (resized_width, resized_height),
            interpolation=cv2.INTER_AREA,
        )

        self.detector.setInputSize(
            (resized_width, resized_height)
        )

        image_bgr = cv2.cvtColor(
            detection_image,
            cv2.COLOR_RGB2BGR,
        )

        _, faces = self.detector.detect(image_bgr)

        if faces is None or len(faces) == 0:
            raise ValueError("Nenhum rosto foi detectado.")

        if len(faces) > 1:
            raise ValueError(
                f"Foram detectados {len(faces)} rostos. "
                "Utilize uma imagem com apenas uma pessoa."
            )

        face = faces[0].copy()
        face[:14] /= scale

        eye1 = face[4:6]
        eye2 = face[6:8]

        if eye1[0] > eye2[0]:
            eye1, eye2 = eye2, eye1

        center = (eye1 + eye2) / 2

        dx = eye2[0] - eye1[0]
        dy = eye2[1] - eye1[1]

        distance = np.hypot(dx, dy)

        if distance <= 0:
            raise ValueError("Não foi possível localizar os olhos.")

        angle = np.degrees(np.arctan2(dy, dx))
        alignment_scale = self.EYE_DISTANCE / distance

        matrix = cv2.getRotationMatrix2D(
            tuple(center),
            angle,
            alignment_scale,
        )

        matrix[0, 2] += self.EYE_X - center[0]
        matrix[1, 2] += self.EYE_Y - center[1]

        aligned = cv2.warpAffine(
            image_rgb,
            matrix,
            (self.OUT_W, self.OUT_H),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

        confidence = float(face[14])

        return aligned, confidence

    def generate_embedding(self, face_rgb):
        image_tensor = self.transform(face_rgb)
        image_tensor = image_tensor.unsqueeze(0).to(self.device)

        with torch.inference_mode():
            embedding = self.model(image_tensor)
            embedding = F.normalize(
                embedding,
                p=2,
                dim=1,
            )

        return embedding.cpu().numpy()[0]

    def process_image(self, image_path):
        image_rgb = self.read_image(image_path)

        aligned_face, confidence = self.align_face(image_rgb)

        embedding = self.generate_embedding(aligned_face)

        return {
            "embedding": embedding,
            "detection_confidence": confidence,
        }
