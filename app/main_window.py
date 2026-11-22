from pathlib import Path

import json
from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QTabWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QLineEdit,
    QListWidget,
    QFileDialog,
    QMessageBox,
)

from src.face_service import FaceService


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Reconhecimento Facial")
        self.resize(850, 600)

        root = Path(__file__).resolve().parent.parent

        config_path = root / "config.json"

        with open(config_path, "r", encoding="utf-8") as file:
            config = json.load(file)

        self.service = FaceService(
            model_path=root / config["model_path"],
            yunet_path=root / config["detector_path"],
            db_path=root / config["database_path"],
            threshold=float(config["threshold"]),
            device=config["device"],
        )

        self.enrollment_images = []

        self.tabs = QTabWidget()

        self.setCentralWidget(self.tabs)

        self.create_enrollment_tab()
        self.create_recognition_tab()
        self.create_people_tab()

    def create_enrollment_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        title = QLabel("Cadastrar pessoa")
        title.setStyleSheet("font-size: 22px; font-weight: bold;")

        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Nome da pessoa")

        select_button = QPushButton("Selecionar fotografias")
        select_button.clicked.connect(self.select_enrollment_images)

        self.enrollment_label = QLabel("Nenhuma fotografia selecionada")

        register_button = QPushButton("Cadastrar")
        register_button.clicked.connect(self.register_person)

        layout.addWidget(title)
        layout.addWidget(self.name_input)
        layout.addWidget(select_button)
        layout.addWidget(self.enrollment_label)
        layout.addWidget(register_button)
        layout.addStretch()

        self.tabs.addTab(tab, "Cadastrar")

    def create_recognition_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        title = QLabel("Reconhecer pessoa")
        title.setStyleSheet("font-size: 22px; font-weight: bold;")

        select_button = QPushButton("Selecionar fotografia")
        select_button.clicked.connect(self.recognize_person)

        self.recognition_label = QLabel(
            "Selecione uma fotografia para iniciar."
        )
        self.recognition_label.setWordWrap(True)

        layout.addWidget(title)
        layout.addWidget(select_button)
        layout.addWidget(self.recognition_label)
        layout.addStretch()

        self.tabs.addTab(tab, "Reconhecer")

    def create_people_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        title = QLabel("Pessoas cadastradas")
        title.setStyleSheet("font-size: 22px; font-weight: bold;")

        self.people_list = QListWidget()

        refresh_button = QPushButton("Atualizar lista")
        refresh_button.clicked.connect(self.refresh_people)

        delete_button = QPushButton("Excluir selecionado")
        delete_button.clicked.connect(self.delete_person)

        layout.addWidget(title)
        layout.addWidget(self.people_list)
        layout.addWidget(refresh_button)
        layout.addWidget(delete_button)

        self.tabs.addTab(tab, "Cadastros")

        self.refresh_people()

    def select_enrollment_images(self):
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Selecionar fotografias",
            "",
            "Imagens (*.png *.jpg *.jpeg *.bmp)",
        )

        if files:
            self.enrollment_images = files
            self.enrollment_label.setText(
                f"{len(files)} fotografia(s) selecionada(s)"
            )

    def register_person(self):
        name = self.name_input.text().strip()

        if not name:
            QMessageBox.warning(
                self, "Atenção", "Informe o nome da pessoa."
            )
            return

        if not self.enrollment_images:
            QMessageBox.warning(
                self, "Atenção", "Selecione pelo menos uma fotografia."
            )
            return

        try:
            result = self.service.enroll(
                name,
                self.enrollment_images,
            )

            QMessageBox.information(
                self,
                "Cadastro realizado",
                f"{result['name']} foi cadastrada com sucesso!\n"
                f"Fotografias processadas: {result['image_count']}",
            )

            self.name_input.clear()
            self.enrollment_images = []
            self.enrollment_label.setText(
                "Nenhuma fotografia selecionada"
            )

            self.refresh_people()

        except Exception as error:
            QMessageBox.critical(
                self,
                "Erro no cadastro",
                str(error),
            )

    def recognize_person(self):
        file, _ = QFileDialog.getOpenFileName(
            self,
            "Selecionar fotografia",
            "",
            "Imagens (*.png *.jpg *.jpeg *.bmp)",
        )

        if not file:
            return

        try:
            result = self.service.recognize_image(file)

            if result["recognized"]:
                message = (
                    f"Pessoa reconhecida: {result['name']}\n"
                    f"Similaridade: {result['similarity']:.4f}"
                )
            else:
                message = (
                    "Pessoa desconhecida.\n"
                    f"Similaridade: {result['similarity']:.4f}"
                    if result["similarity"] is not None
                    else "Nenhuma pessoa cadastrada."
                )

            self.recognition_label.setText(message)

        except Exception as error:
            QMessageBox.critical(
                self,
                "Erro no reconhecimento",
                str(error),
            )

    def refresh_people(self):
        self.people_list.clear()

        for person in self.service.list_people():
            self.people_list.addItem(
                f"{person['name']} | ID: {person['id']}"
            )

    def delete_person(self):
        selected = self.people_list.currentItem()

        if selected is None:
            QMessageBox.warning(
                self, "Atenção", "Selecione uma pessoa."
            )
            return

        person_id = selected.text().split("ID: ")[-1]

        answer = QMessageBox.question(
            self,
            "Confirmar exclusão",
            "Deseja realmente excluir este cadastro?",
        )

        if answer != QMessageBox.StandardButton.Yes:
            return

        if self.service.delete_person(person_id):
            QMessageBox.information(
                self,
                "Exclusão",
                "Cadastro excluído com sucesso.",
            )
        else:
            QMessageBox.warning(
                self,
                "Exclusão",
                "O cadastro não foi encontrado.",
            )

        self.refresh_people()
