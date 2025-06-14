# Projeto: Otimização de Hiperparâmetros para Predição de CBR

> Este projeto utiliza um Algoritmo Genético (AG) para otimizar os hiperparâmetros de modelos de Machine Learning (Rede Neural MLP e Random Forest) com o objetivo de prever o valor do California Bearing Ratio (CBR) de solos.

O framework está estruturado para ser modular, configurável e para registrar automaticamente os resultados de cada execução, facilitando a análise e a reprodutibilidade dos experimentos.

---

## 🚀 Tecnologias Utilizadas

* **Python 3.10+**
* **PyTorch**: Para a construção e treino do modelo de Rede Neural (MLP).
* **Scikit-learn**: Para o modelo Random Forest, pré-processamento de dados e métricas de avaliação.
* **Pandas & NumPy**: Para manipulação e análise de dados.
* **Rich**: Para a criação de saídas visualmente agradáveis no terminal.
* **Tqdm**: Para a exibição de barras de progresso.
* **Matplotlib & Seaborn**: Para a geração de gráficos de análise.

---

## 📂 Estrutura do Projeto

O projeto é organizado da seguinte forma para garantir a separação de responsabilidades:

PROJETOFINAL/
│
├── data/
│   └── SEU_ARQUIVO_DE_DADOS.xlsx
│
├── models/
│   ├── mlp_space.py        # Define a arquitetura e o espaço de busca da MLP
│   └── rf_space.py         # Define o espaço de busca do Random Forest
│
├── optimization/
│   └── ga_tuner.py         # Contém a classe do Algoritmo Genético
│
├── results/
│   └── run_YYYY-MM-DD_HH-MM-SS/ # Pasta para cada execução
│       ├── run_log.jsonl       # Log detalhado em formato JSON
│       └── *.png               # Gráficos de resultados salvos como imagem
│
├── .gitignore                # Arquivos e pastas a serem ignorados pelo Git
├── analysis.py               # Módulo para análise e plotagem final
├── config.py                 # Painel de controle para configurar o experimento
├── data_loader.py            # Módulo para carregar e preparar os dados
├── evaluation.py             # Lógica de avaliação dos modelos
├── main.py                   # Ponto de entrada principal para executar o projeto
└── requirements.txt          # Lista de dependências do Python


---

## ⚙️ Como Executar o Projeto (Passo a Passo)

Este guia foi feito para um ambiente **Linux (Ubuntu / WSL)**.

### Pré-requisitos

* Git
* Python 3.10 ou superior
* Acesso a um terminal (shell) Bash.

### Passo 1: Clonar o Repositório

Primeiro, clone este repositório para a sua máquina local.

```bash
git clone [https://github.com/iagoLopex/TCC.git](https://github.com/iagoLopex/TCC.git)
cd TCC
Passo 2: Configurar o Ambiente Virtual (venv)
É crucial usar um ambiente virtual para isolar as dependências do projeto.

Bash

# 1. Garanta que o pacote python3-venv está instalado (para Debian/Ubuntu)
sudo apt update && sudo apt install python3-venv -y

# 2. Crie o ambiente virtual na pasta do projeto
python3 -m venv venv

# 3. Ative o ambiente virtual
source venv/bin/activate
Após a ativação, você verá (venv) no início do prompt do seu terminal.

Passo 3: Instalar as Dependências
Instale todas as bibliotecas necessárias listadas no requirements.txt.

Bash

# Opcional, mas recomendado: atualize o pip
pip install --upgrade pip

# Instale os pacotes
pip install -r requirements.txt
Passo 4: Configurar o Experimento
Antes de executar, você pode customizar o experimento editando o arquivo config.py. Nele, você pode alterar:

O modelo a ser otimizado (MODEL_TO_OPTIMIZE).
O conjunto de dados a ser usado (DATASET).
O método de validação (VALIDATION_METHOD).
Os parâmetros do Algoritmo Genético (GA_PARAMS).
Passo 5: Executar a Otimização
Com tudo pronto, basta executar o script principal.

Bash

python3 main.py
O script começará a otimização. Você verá o progresso no terminal e, ao final, uma nova pasta será criada dentro de results/ com todos os logs e gráficos da execução.

Passo 6: Desativar o Ambiente
Quando terminar de trabalhar no projeto, você pode desativar o ambiente virtual.

Bash

deactivate
📊 Análise dos Resultados
Após cada execução, navegue até a pasta results/ e encontre a subpasta nomeada com a data e hora da sua execução (ex: run_2025-06-15_18-18-00/). Dentro dela, você encontrará:

run_log.jsonl: Um arquivo com o log completo e detalhado de cada geração do algoritmo genético, em formato JSON, ideal para análises programáticas.
*.png: Os gráficos de avaliação do melhor modelo, como a curva de aprendizado e a análise de resíduos, já salvos como imagens.
👤 Autor
Iago Lopes

GitHub: iagoLopex
<!-- end list -->
