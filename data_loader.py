"""
Módulo de Carregamento e Preparação de Dados.

Este módulo contém a função responsável por ler o arquivo de dados,
selecionar as colunas relevantes com base na configuração e separar
as features (X) do alvo (y).
"""
from typing import Tuple

import pandas as pd


def load_and_prepare_data(
    csv_path: str, dataset_option: str
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Carrega os dados de um arquivo Excel e os prepara para o modelo.

    Lê o arquivo, seleciona as colunas de features com base na opção de
    dataset, remove linhas com valores nulos e separa as features (X)
    do alvo (y).

    Args:
        csv_path (str): O caminho para o arquivo .xlsx.
        dataset_option (str): A chave para selecionar o conjunto de colunas
            ('D1', 'D2', etc.).

    Returns:
        Tuple[pd.DataFrame, pd.Series]: Uma tupla contendo o DataFrame de
            features (X) e a Series do alvo (y).

    Raises:
        FileNotFoundError: Se o arquivo no `csv_path` não for encontrado.
        KeyError: Se as colunas selecionadas não existirem no DataFrame.
    """
    try:
        df_original = pd.read_excel(csv_path)
    except FileNotFoundError:
        print(f"ERRO: Arquivo de dados não encontrado em '{csv_path}'.")
        # Retorna objetos vazios para que o programa principal possa parar
        return pd.DataFrame(), pd.Series()

    # Mapeamento de datasets para colunas
    column_map = {
        "D1": ["D6", "CH", "CY", "CBR"],
        "D2": ["D6", "CH", "CY", "IP", "CBR"],
        "D3": ["IG", "EXP", "D3", "D4", "D5", "D6", "CH", "CY", "IP", "LL", "CBR"],
    }
    # Usa 'D3' como padrão se a opção não for encontrada
    selected_columns = column_map.get(dataset_option, column_map["D3"])

    try:
        df_processed = df_original[selected_columns].dropna().copy()
    except KeyError as e:
        print(f"ERRO: A coluna {e} não foi encontrada no arquivo Excel.")
        return pd.DataFrame(), pd.Series()

    if df_processed.empty:
        print("ALERTA: O DataFrame ficou vazio após remover linhas com valores nulos.")
        print("Verifique se as colunas para o dataset escolhido existem e estão preenchidas no Excel.")

    X = df_processed.drop(columns="CBR")
    y = df_processed["CBR"]

    return X, y