# data_loader.py
import pandas as pd

def load_and_prepare_data(csv_path, dataset_option):
    """Carrega dados de um arquivo Excel, seleciona colunas e remove NaNs."""
    print("--- Etapa 1: Preparando Dados ---")
    try:
        df_original = pd.read_excel(csv_path)
    except FileNotFoundError:
        print(f"Erro: Arquivo não encontrado em '{csv_path}'.")
        # Retornar DataFrames vazios para evitar que o resto do código quebre
        return pd.DataFrame(), pd.Series()

    if dataset_option == 'D1':
        colunas_selecionadas = ['D6', 'CH', 'CY', 'CBR']
    elif dataset_option == 'D2':
        colunas_selecionadas = ['D6', 'CH', 'CY', 'IP', 'CBR']
    else:
        colunas_selecionadas = ['D6', 'CH', 'CY', 'IP', 'LL', 'CBR']

    df = df_original[colunas_selecionadas].dropna().copy()
    
    Xdf = df.drop(columns='CBR')
    y = df['CBR']
    
    print(f"Features selecionadas para o dataset '{dataset_option}': {Xdf.columns.tolist()}")
    return Xdf, y