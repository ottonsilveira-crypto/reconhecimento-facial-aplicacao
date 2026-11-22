# Sistema de Reconhecimento Facial

Aplicação desktop de reconhecimento facial desenvolvida em Python, com processamento local de imagens, detecção facial, geração de embeddings e comparação por similaridade de cosseno.

## Funcionalidades

- Cadastro de pessoas a partir de fotografias.
- Detecção e alinhamento facial utilizando YuNet.
- Extração de embeddings faciais com ResNet-18.
- Armazenamento local de protótipos biométricos em SQLite.
- Reconhecimento de pessoas cadastradas por imagem.
- Rejeição de pessoas desconhecidas por meio de um threshold configurável.
- Tratamento de imagens sem rostos ou com múltiplos rostos.
- Identificação de possíveis cadastros duplicados.
- Exclusão de cadastros e dados biométricos associados.
- Registro de eventos e erros por meio de logs.

## Tecnologias utilizadas

- Python
- PyTorch
- Torchvision
- OpenCV
- YuNet
- NumPy
- SQLite
- PySide6

## Organização do projeto

```text
reconhecimento-facial/
│
├── app/
│   ├── __init__.py
│   └── main_window.py
│
├── src/
│   ├── __init__.py
│   ├── database.py
│   ├── face_pipeline.py
│   ├── face_service.py
│   ├── logger.py
│   ├── model.py
│   ├── recognition.py
│   ├── data.py
│   └── metrics.py
│
├── tests/
│   ├── test_database.py
│   └── test_recognition.py
│
├── artifacts/
│   ├── lfw_resnet18_finetuned.pt
│   └── face_detection_yunet_2023mar.onnx
│
├── data/
├── logs/
├── config.json
├── requirements.txt
├── main.py
└── README.md
```

## Requisitos

- Windows 10 ou superior.
- Python 3.10 ou superior.
- Dependências listadas em `requirements.txt`.
- Arquivos dos modelos necessários para inferência.

## Instalação

Clone o repositório:

```powershell
git clone https://github.com/ottonsilveira-crypto/reconhecimento-facial-aplicacao
cd reconhecimento-facial-aplicacao
```

Crie e ative um ambiente virtual:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Instale as dependências:

```powershell
pip install -r requirements.txt
```

Caso o PySide6 não esteja incluído nas dependências:

```powershell
pip install PySide6
```

## Modelos necessários

Os arquivos de modelo não são incluídos automaticamente no repositório.

Disponibilize os seguintes arquivos na pasta `artifacts/`:

- `lfw_resnet18_finetuned.pt`: modelo de embeddings faciais.
- `face_detection_yunet_2023mar.onnx`: modelo de detecção facial YuNet.

Os caminhos podem ser alterados em `config.json`.

## Configuração

O arquivo `config.json` permite configurar os parâmetros principais:

```json
{
    "threshold": 0.45,
    "device": "cpu",
    "model_path": "artifacts/lfw_resnet18_finetuned.pt",
    "detector_path": "artifacts/face_detection_yunet_2023mar.onnx",
    "database_path": "data/app/database.sqlite3"
}
```

O threshold determina a similaridade mínima necessária para aceitar uma identificação.

O valor utilizado é experimental e pode precisar de calibração conforme o conjunto de dados e as condições de utilização.

## Execução

Inicie a aplicação com:

```powershell
python main.py
```

A interface permite cadastrar pessoas, reconhecer rostos em fotografias e gerenciar os cadastros existentes.

## Testes

Execute os testes automatizados com:

```powershell
python -m unittest discover -s tests -v
```

## Privacidade e proteção de dados

- O processamento das imagens é realizado localmente.
- A aplicação não depende de envio de fotografias para serviços em nuvem.
- As fotografias utilizadas no cadastro são processadas para gerar embeddings; as imagens originais não são armazenadas pelo banco de dados da aplicação.
- Os embeddings são dados biométricos sensíveis e ficam armazenados localmente em SQLite.
- O banco SQLite não possui criptografia própria.
- A exclusão de uma pessoa remove seu registro e os embeddings associados.
- Imagens privadas, bancos locais e logs não devem ser enviados ao repositório público.

A utilização do sistema deve respeitar a legislação aplicável, incluindo a Lei Geral de Proteção de Dados Pessoais (LGPD). O cadastro deve ocorrer com base legal adequada e transparência quanto à finalidade do tratamento.

## Limitações conhecidas

- O reconhecimento é realizado a partir de fotografias estáticas.
- A aplicação exige exatamente um rosto detectado por imagem.
- O threshold atual é experimental e não representa uma probabilidade de identificação.
- O sistema não deve ser considerado adequado para decisões de alto impacto ou utilização crítica sem validação adicional.

## Licença

A definir.
