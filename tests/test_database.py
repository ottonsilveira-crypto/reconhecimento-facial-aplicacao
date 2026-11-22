import tempfile
import unittest

import numpy as np

from pathlib import Path
from src.database import FaceDatabase


class TestFaceDatabase(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

        db_path = Path(self.temp_dir.name) / "test.sqlite3"

        self.database = FaceDatabase(db_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_add_and_read_person(self):
        embedding1 = np.zeros(128, dtype=np.float32)
        embedding1[0] = 1.0

        embedding2 = np.zeros(128, dtype=np.float32)
        embedding2[0] = 1.0

        person_id = self.database.add_person(
            "Pessoa de teste",
            [embedding1, embedding2],
        )

        people = self.database.list_people()
        prototypes = self.database.get_prototypes()

        self.assertEqual(len(people), 1)
        self.assertEqual(people[0]["id"], person_id)
        self.assertEqual(people[0]["name"], "Pessoa de teste")

        self.assertEqual(len(prototypes), 1)
        self.assertEqual(
            prototypes[0]["embedding"].shape,
            (128,),
        )

        self.assertAlmostEqual(
            float(np.linalg.norm(prototypes[0]["embedding"])),
            1.0,
            places=5,
        )

    def test_delete_person(self):
        embedding = np.zeros(128, dtype=np.float32)
        embedding[0] = 1.0

        person_id = self.database.add_person(
            "Pessoa para excluir",
            [embedding],
        )

        deleted = self.database.delete_person(person_id)

        self.assertTrue(deleted)
        self.assertEqual(len(self.database.list_people()), 0)
        self.assertEqual(len(self.database.get_prototypes()), 0)

    def test_reject_empty_name(self):
        embedding = np.zeros(128, dtype=np.float32)
        embedding[0] = 1.0

        with self.assertRaises(ValueError):
            self.database.add_person("", [embedding])

    def test_reject_invalid_embedding(self):
        embedding = np.zeros(128, dtype=np.float32)

        with self.assertRaises(ValueError):
            self.database.add_person(
                "Pessoa inválida",
                [embedding],
            )


if __name__ == "__main__":
    unittest.main()
