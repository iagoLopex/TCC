# Otimização de Hiperparâmetros com Algoritmo Genético: Uma Análise Comparativa entre Artificial Neural Networks e Random Forest para Predição do Índice de Suporte Califórnia

> Este projeto utiliza um Algoritmo Genético (AG) para otimizar os hiperparâmetros de modelos de Machine Learning (Rede Neural MLP e Random Forest) com o objetivo de prever o valor do California Bearing Ratio (CBR) de solos.

---

## 📖 Fonte dos Dados e Tratamento

O conjunto de dados utilizado neste projeto foi extraído do estudo de doutorado de **José Gustavo Hermida de Mello Ferreira**, conforme a referência abaixo:

> FERREIRA, JOSÉ GUSTAVO HERMIDA DE MELLO. **Tratamento de Dados Geotécnicos Para Predição de Módulos de Resiliência de Solos e Britas Utilizando Ferramentas de Data Mining**. Tese (D.Sc., Engenharia Civil) - COPPE/UFRJ, Rio de Janeiro, 2008.

O dataset original, contendo 463 amostras, passou por um processo de tratamento e curadoria para esta análise, restando algo entorno de 330 amostras que possuem o CBR não nulo. As variáveis foram renomeadas para maior clareza (e.g., `CH`, `IP`, `CBR`). Para cada experimento, um subconjunto específico de features é selecionado (conforme definido em `config.py`). Subsequentemente, todas as amostras (linhas) que continham valores ausentes (`NaN`) em qualquer uma das colunas selecionadas foram removidas (`dropna()`) para garantir a qualidade e a integridade dos dados de entrada para os modelos.

## Análise breve do dataset

A variável target CBR nessa base de dados, com relação à quantidade de amostras, está mal distribuída no intervalo de 1 até 155, o que leva a problemas iniciais de treinamento dos modelos.

![Image](https://github.com/user-attachments/assets/522970d4-d667-46ae-ab90-552a991eaba4)

>A proposta utilizada é a possibilidade de atribuição de pesos para as amostras de CBR que tem o valor maior que um valor fixo que o usuário definir (Google Developers, 2025), para uma análise mais profunda de como o modelo lida com estes testes.

## 📚 Referências
* YABI, C. P. et al. **Prediction of CBR by Deep Artificial Neural Networks with Hyperparameter Optimization by Simulated Annealing**. Indian Geotechnical Journal, v. 54, n. 1, p. 121-137, fev. 2024. Disponível em: <https://doi.org/10.1007/s40098-024-00870-4>. Acesso em: 3 jun. 2025.

* TADO, N.; MEDIHAJIT, S.; PAL, D. **Forecasting California bearing ratio (CBR) of soil using machine learning algorithms: A review**. Research on Engineering Structures and Materials and Materials, v. 11, n. 1, p. 383-398, 2025. Disponível em: <http://dx.doi.org/10.17515/resm2025-623ml0115rv>. Acesso em: 3 jun. 2025.

* Google Developers. (s.d.). **Conjuntos de dados desequilibrados**. Machine Learning Crash Course. Disponível em: <https://developers.google.com/machine-learning/crash-course/classification/handling-imbalanced-classes>. Acesso em: 5 jun. 2025.

* BERNUCCI, Liedi Bariani et al. **Pavimentação asfáltica: formação básica para engenheiros**. 2. ed. Rio de Janeiro: Petrobras, 2022.

* ORTEGA, Julio Bizarreta; AVEROS, Sara Ochoa. **Manual Didático para a Execução do Ensaio Índice de Suporte Califórnia (ISC)**. Foz do Iguaçu: Edunila, 2022.

## 🔬 Metodologia

A metodologia central deste trabalho consiste em aplicar técnicas de otimização e aprendizado de máquina para prever o California Bearing Ratio (CBR) de solos. Utiliza-se um **Algoritmo Genético (AG)** para explorar um vasto espaço de hiperparâmetros e encontrar a configuração ótima para dois modelos de regressão:

1.  **Rede Neural Artificial (MLP - Multi-Layer Perceptron)**
2.  **Random Forest (Floresta Aleatória)**

>O framework é projetado para executar esses experimentos de forma sistemática, registrando métricas de performance, logs detalhados e visualizações para cada execução, permitindo uma análise comparativa robusta entre os modelos.

![Image](https://github.com/user-attachments/assets/0382240d-1163-43be-9d6c-810b9c30139a)


>Esse fluxograma apresenta a metodologia de testes que o usuário pode efetuar com base nas decisões do mesmo, como escolha de configuração dos dados de entrada com base no dataset dado (representado pelos Datasets 1, 2 e 3), atribuição ou não de pesos para algumas variáveis, e escolha do método de cross-validation (K-fold ou Holdout).

Exemplo de configurações de datasets (a partir do feature selection):

![Image](https://github.com/user-attachments/assets/4029f516-3bb9-496b-882b-4bacacad2b24)

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

O projeto é organizado com a seguinte estrutura para garantir a separação de responsabilidades:

- **PROJETOFINAL/**
  - 📂 **data/**
    - `SEU_ARQUIVO_DE_DADOS.xlsx`
  - 📂 **models/**
    - `__init__.py`
    - `mlp_space.py` _(Define a arquitetura e o espaço de busca da MLP)_
    - `rf_space.py` _(Define o espaço de busca do Random Forest)_
  - 📂 **notebooks_de_analise/**
    - `analise_shap.ipynb` _(Notebook para análise de interpretabilidade)_
  - 📂 **optimization/**
    - `__init__.py`
    - `ga_tuner.py` _(Contém a classe do Algoritmo Genético)_
  - 📂 **results/** _(Gerada automaticamente)_
    - `run_YYYY-MM-DD_HH-MM-SS/`
      - `run_log.jsonl` _(Log detalhado em formato JSON)_
      - `*.png` _(Gráficos de resultados)_
  - 📂 **venv/** _(Ignorada pelo .gitignore)_
  - 📄 `analysis.py` _(Módulo para análise e plotagem final)_
  - 📄 `config.py` _(Painel de controle para configurar o experimento)_
  - 📄 `data_loader.py` _(Módulo para carregar e preparar os dados)_
  - 📄 `evaluation.py` _(Lógica de avaliação dos modelos)_
  - 📄 `main.py` _(Ponto de entrada principal para executar o projeto)_
  - 📄 `.gitignore` _(Arquivos e pastas a serem ignorados pelo Git)_
  - 📄 `README.md` _(Este arquivo)_
  - 📄 `requirements.txt` _(Lista de dependências do Python)_

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
# clone do repositório
git clone https://github.com/iagoLopex/TCC.git

# entrar nele
cd TCC
```

### Passo 2: Configurar o Ambiente Virtual (venv) (É crucial usar um ambiente virtual para isolar as dependências do projeto)


```bash
# 1. Garanta que o pacote python3-venv está instalado (para Debian/Ubuntu)
sudo apt update && sudo apt install python3-venv -y

# 2. Crie o ambiente virtual na pasta do projeto
python3 -m venv venv

# 3. Ative o ambiente virtual
source venv/bin/activate
```
Após a ativação, você verá (venv) no início do prompt do seu terminal.

### Passo 3: Instalar as Dependências (Instale todas as bibliotecas necessárias listadas no requirements.txt)

```bash
# Opcional, mas recomendado: atualize o pip
pip install --upgrade pip

# Instale os pacotes
pip install -r requirements.txt
```

### Passo 4: Configurar o Experimento (Antes de executar, você pode customizar o experimento editando o arquivo config.py. Nele, você pode alterar)

 - O modelo a ser otimizado (MODEL_TO_OPTIMIZE).
 - O conjunto de dados a ser usado (DATASET).
 - O método de validação (VALIDATION_METHOD).
 - Os parâmetros do Algoritmo Genético (GA_PARAMS).
 - Passo 5: Executar a Otimização
 - Com tudo pronto, basta executar o script principal.

```bash
python3 main.py
```

>O script começará a otimização. Você verá o progresso no terminal e, ao final, uma nova pasta será criada dentro de results/ com todos os logs e gráficos da execução:


![Image](https://github.com/user-attachments/assets/7f96e19a-5100-4fac-a7c8-bae62bf89b13)


## 📊 Análise dos Resultados

>Após cada execução, navegue até a pasta results/ e encontre a subpasta nomeada com a data e hora da sua execução (ex: run_2025-06-17_14-45-00/). Dentro dela, você encontrará:

- run_log.jsonl: Um arquivo com o log completo e detalhado de cada geração do algoritmo genético, em formato JSON e os hiperparâmetros utilizados.
- *.png: Os gráficos de avaliação do melhor modelo, como a curva de aprendizado e a análise de resíduos, já salvos como imagens.
- final_model.joblib ou final_model.pth: O objeto do modelo final treinado, pronto para ser carregado e usado em outras análises (como no notebooks_analysis).

![Image](https://github.com/user-attachments/assets/029a8a76-2144-42de-bd01-7a4a863ce689)

## 💡 Dando Continuidade ao Projeto
A estrutura modular foi projetada para facilitar a expansão.

>Para Adicionar um Novo Modelo (ex: Gradient Boosting):
 1. Crie o Arquivo: Crie um novo arquivo models/gb_space.py.
 2. Implemente a Classe: Dentro dele, crie uma classe GradientBoostingSpace seguindo a mesma estrutura da MLPBlockSpace ou RandomForestSpace. Ela precisa ter os atributos bounds e types, e os métodos decode e evaluate.
 3. Registre o Modelo: No arquivo evaluation.py, importe sua nova classe e adicione-a ao dicionário MODEL_CLASSES:

```Python
MODEL_CLASSES = {"MLP": MLPBlockSpace, "RF": RandomForestSpace, "GB": GradientBoostingSpace}
```
 4. Configure e Rode: No config.py, mude MODEL_TO_OPTIMIZE = "GB" e execute o main.py.

>Para Testar um Novo Conjunto de Features:
 1. Edite o data_loader.py: Adicione uma nova entrada ao dicionário column_map, por exemplo, 'D4', com a lista de colunas desejada.
 2. Configure e Rode: No config.py, mude DATASET = "D4" e execute o main.py.

👤 Autor
Iago de Souza Lopes

GitHub: iagoLopex
<!-- end list -->