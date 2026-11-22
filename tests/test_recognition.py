import unittest

import numpy as np

from src.recognition import FaceRecognizer


class FakeDatabase:
    """Banco fictício para testar o reconhecimento sem SQLite."""

    def __init__(self, prototypes):
        self.prototypes = prototypes

    def get_prototypes(self):
        return self.prototypes


class TestFaceRecognizer(unittest.TestCase):

    def setUp(self):
        self.person = {
            "id": "123",
            "name": "Pessoa cadastrada",
            "embedding": np.array([1.0, 0.0], dtype=np.float32),
        }

    def test_recognize_known_person(self):
        database = FakeDatabase([self.person])
        recognizer = FaceRecognizer(database, threshold=0.8)

        result = recognizer.recognize(
            np.array([0.99, 0.01], dtype=np.float32)
        )

        self.assertTrue(result["recognized"])
        self.assertEqual(result["name"], "Pessoa cadastrada")
        self.assertEqual(result["person_id"], "123")
        self.assertGreaterEqual(result["similarity"], 0.8)

    def test_reject_unknown_person(self):
        database = FakeDatabase([self.person])
        recognizer = FaceRecognizer(database, threshold=0.8)

        result = recognizer.recognize(
            np.array([0.0, 1.0], dtype=np.float32)
        )

        self.assertFalse(result["recognized"])
        self.assertEqual(result["reason"], "unknown")
        self.assertIsNone(result["name"])

    def test_empty_database(self):
        database = FakeDatabase([])
        recognizer = FaceRecognizer(database, threshold=0.8)

        result = recognizer.recognize(
            np.array([1.0, 0.0], dtype=np.float32)
        )

        self.assertFalse(result["recognized"])
        self.assertEqual(result["reason"], "empty_database")

    def test_reject_zero_embedding(self):
        database = FakeDatabase([self.person])
        recognizer = FaceRecognizer(database, threshold=0.8)

        with self.assertRaises(ValueError):
            recognizer.recognize(
                np.array([0.0, 0.0], dtype=np.float32)
            )

    def test_reject_invalid_embedding(self):
        database = FakeDatabase([self.person])
        recognizer = FaceRecognizer(database, threshold=0.8)

        with self.assertRaises(ValueError):
            recognizer.recognize(
                np.array([float("nan"), 0.0], dtype=np.float32)
            )


if __name__ == "__main__":
    unittest.main()
