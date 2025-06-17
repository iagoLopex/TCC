# Projeto: Otimização de Hiperparâmetros para Predição de CBR

> Este projeto utiliza um Algoritmo Genético (AG) para otimizar os hiperparâmetros de modelos de Machine Learning (MLP e Random Forest) visando prever o California Bearing Ratio (CBR) de solos.

---

## 📖 Fonte dos Dados e Tratamento

O dataset original (463 amostras) foi extraído da tese de doutorado de **José Gustavo Hermida de Mello Ferreira** (2008) e passou por:

1. Renomeação de variáveis para maior clareza (`CH`, `IP`, `CBR`, etc.).
2. Seleção de subconjuntos de features via `config.py`.
3. Remoção de todas as linhas com valores ausentes (`dropna()`).

---

## 🔬 Metodologia

Aplicamos um **Algoritmo Genético** para buscar a melhor configuração de hiperparâmetros em dois modelos de regressão:

1. **MLP (Multi-Layer Perceptron)**  
2. **Random Forest**

O pipeline executa cada experimento de forma sistemática, gerando:

- Métricas de performance
- Logs detalhados (JSONL)
- Gráficos comparativos

---

## 🚀 Tecnologias Utilizadas

- **Python 3.10+**  
- **PyTorch** (MLP)  
- **Scikit-learn** (Random Forest, pré-processamento, métricas)  
- **Pandas & NumPy** (manipulação de dados)  
- **Rich** (saídas no terminal)  
- **Tqdm** (barras de progresso)  
- **Matplotlib & Seaborn** (visualizações)  

---

## 📂 Estrutura do Projeto

PROJETOFINAL/
├── data/
│ └── SEU_ARQUIVO_DE_DADOS.xlsx
├── models/
│ ├── init.py
│ ├── mlp_space.py # Define arquitetura e espaço de busca da MLP
│ └── rf_space.py # Define espaço de busca do Random Forest
├── notebooks_de_analise/
│ └── analise_shap.ipynb # Notebook de interpretabilidade
├── optimization/
│ ├── init.py
│ └── ga_tuner.py # Classe do Algoritmo Genético
├── results/ # Gerada automaticamente
│ └── run_YYYY-MM-DD_HH-MM-SS/
│ ├── run_log.jsonl # Log em JSONL
│ └── *.png # Gráficos de resultados
├── venv/ # Ignorado pelo .gitignore
├── analysis.py # Análise e plotagem final
├── config.py # Painel de controle do experimento
├── data_loader.py # Carregamento e preparação dos dados
├── evaluation.py # Lógica de avaliação dos modelos
├── main.py # Ponto de entrada principal
├── .gitignore # Arquivos/pastas ignorados pelo Git
├── README.md # Este arquivo
└── requirements.txt # Dependências Python

yaml
Copiar
Editar

---

## ⚙️ Como Executar o Projeto

Este guia assume **Linux (Ubuntu / WSL)**.

### Pré-requisitos

- Git  
- Python 3.10+  
- Bash (terminal)

---

### Passo 1: Clonar o Repositório

```bash
git clone https://github.com/iagoLopex/TCC.git
cd TCC
Passo 2: Configurar o Ambiente Virtual
Instalar venv (Debian/Ubuntu):

bash
Copiar
Editar
sudo apt update && sudo apt install python3-venv -y
Criar o venv:

bash
Copiar
Editar
python3 -m venv venv
Ativar o venv:

bash
Copiar
Editar
source venv/bin/activate
Passo 3: Instalar Dependências
bash
Copiar
Editar
pip install --upgrade pip
pip install -r requirements.txt
Passo 4: Configurar o Experimento
Edite o config.py para ajustar:

MODEL_TO_OPTIMIZE

DATASET

VALIDATION_METHOD

GA_PARAMS

Passo 5: Executar a Otimização
bash
Copiar
Editar
python3 main.py
Ao final, uma nova pasta em results/ será criada com logs e gráficos.

Passo 6: Desativar o Ambiente
bash
Copiar
Editar
deactivate
📊 Análise dos Resultados
Acesse a pasta gerada em results/run_YYYY-MM-DD_HH-MM-SS/ e verifique:

run_log.jsonl: logs do AG por geração

*.png: curvas de aprendizado, análises de resíduos

final_model.joblib/.pth: modelo treinado para uso posterior

👤 Autor
Iago Lopes
GitHub: iagoLopex