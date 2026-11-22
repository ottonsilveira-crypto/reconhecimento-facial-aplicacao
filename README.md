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
reconhecimento-facial-aplicacao/
│
├── app/
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
│   ├── test_metrics.py
│   └── test_recognition.py
│
├── third_party_licenses/
│   └── YUNET-MIT-LICENSE.txt
│
├── artifacts/       # Modelos locais, não incluídos no repositório
├── data/            # Dados locais da aplicação
├── logs/            # Logs locais
├── .gitignore
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
git clone https://github.com/ottonsilveira-crypto/reconhecimento-facial-aplicacao.git
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

## Modelos necessários

Os arquivos dos modelos não são incluídos diretamente no repositório.

A aplicação requer os seguintes arquivos na pasta `artifacts/`:

- `lfw_resnet18_finetuned.pt`: modelo de embeddings faciais treinado com base no ResNet-18.
- `face_detection_yunet_2023mar.onnx`: modelo de detecção facial YuNet.

### Download do YuNet

O modelo YuNet está disponível na Release `v1.0.0` do projeto:

[Baixar YuNet – Release v1.0.0](https://github.com/ottonsilveira-crypto/reconhecimento-facial-aplicacao/releases/tag/v1.0.0)

Após o download, coloque o arquivo `face_detection_yunet_2023mar.onnx` na pasta `artifacts/`.

### Modelo ResNet-18

O checkpoint `lfw_resnet18_finetuned.pt` não é distribuído junto ao repositório ou à Release.

Para executar a aplicação, é necessário fornecer esse arquivo separadamente e colocá-lo na pasta `artifacts/`.

Os caminhos dos modelos podem ser alterados em `config.json`.

## Configuração

O arquivo `config.json` permite configurar os principais parâmetros da aplicação:

```json
{
    "threshold": 0.45,
    "device": "cpu",
    "model_path": "artifacts/lfw_resnet18_finetuned.pt",
    "detector_path": "artifacts/face_detection_yunet_2023mar.onnx",
    "database_path": "data/app/database.sqlite3"
}
```

O parâmetro `threshold` determina a similaridade mínima necessária para aceitar uma identificação.

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
- A aplicação não depende do envio de fotografias para serviços em nuvem.
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

A licença do código-fonte deste projeto ainda será definida.

A licença do código-fonte não se estende automaticamente aos modelos de terceiros utilizados pela aplicação.

## Modelos de terceiros

### YuNet

- Modelo: `face_detection_yunet_2023mar.onnx`
- Origem: [OpenCV Zoo – YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)
- Copyright: © 2020 Shiqi Yu
- Licença: MIT
- Texto da licença: [YUNET-MIT-LICENSE.txt](third_party_licenses/YUNET-MIT-LICENSE.txt)

O modelo YuNet é distribuído sob os termos da licença MIT. Consulte o arquivo de licença incluído no projeto para obter os termos completos.