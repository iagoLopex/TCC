# main.py

import numpy as np
import multiprocessing as mp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
import logging
import os
from datetime import datetime
from pythonjsonlogger.jsonlogger import JsonFormatter
from rich.console import Console # <-- Adicionado
from rich.panel import Panel     # <-- Adicionado
from rich.syntax import Syntax   # <-- Adicionado
import json                       # <-- Adicionado

# Importa os módulos
import config
from data_loader import load_and_prepare_data
from evaluation import GenericSpace
from optimization.ga_tuner import BaseGATuner
from analysis import analyze_and_plot

# Ignorar avisos
warnings.filterwarnings('ignore')

def main():
    # --- INÍCIO DA MODIFICAÇÃO: Rich para Console, Logging para Arquivo ---
    console = Console() # Cria um console Rich para saídas bonitas

    run_timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    run_results_path = os.path.join('results', f'run_{run_timestamp}')
    os.makedirs(run_results_path, exist_ok=True)
    
    # Configura o logging para salvar APENAS no arquivo JSON. Nada irá para o console a partir daqui.
    json_log_filepath = os.path.join(run_results_path, 'run_log.jsonl')
    log_handler = logging.FileHandler(filename=json_log_filepath)
    log_handler.setFormatter(JsonFormatter())
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # Limpa handlers antigos e adiciona apenas o de arquivo
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.addHandler(log_handler)
    
    # --- FIM DA MODIFICAÇÃO ---

    config.set_seeds()

    # --- INÍCIO DA MODIFICAÇÃO: Usando Rich Print para o Console ---
    console.print(Panel(f"INICIANDO NOVA EXECUÇÃO\nResultados salvos em: [cyan]{run_results_path}[/cyan]", title="[bold green]Otimizador de CBR[/bold green]", expand=False))

    run_config = {
        'model_to_optimize': config.MODEL_TO_OPTIMIZE,
        'dataset': config.DATASET,
        'validation_method': config.VALIDATION_METHOD,
        'use_weights': config.USE_WEIGHTS,
        'seed': config.SEED,
        'ga_params': config.GA_PARAMS
    }
    # Loga no arquivo JSON
    logging.info("Configurações da Execução", extra={'json_fields': run_config})
    # Imprime bonito no console
    console.print("[bold]Configurações da Execução:[/bold]")
    console.print(Syntax(json.dumps(run_config, indent=4), "json", theme="monokai", line_numbers=False))

    console.print("\n[bold]--- Etapa 1: Preparando Dados ---[/bold]")
    Xdf, y = load_and_prepare_data(config.CSV_PATH, config.DATASET)
    if Xdf.empty:
        return
    
    console.print(f"[bold]--- Etapa 2: Configurando Estratégia de Validação: {config.VALIDATION_METHOD} ---[/bold]")
    # ... (o resto da lógica de divisão de dados continua a mesma, sem prints)
    X_dev, X_test, y_dev, y_test = train_test_split(
        Xdf.values.astype(np.float32), y.values.astype(np.float32).reshape(-1, 1),
        test_size=0.15, random_state=config.SEED
    )
    X_train_s, y_train_s, X_valid_s, y_valid_s = [None] * 4
    if config.VALIDATION_METHOD == 'Holdout':
        X_train, X_valid, y_train, y_valid = train_test_split(
            X_dev, y_dev, test_size=(0.15/0.85), random_state=config.SEED
        )
        scX = StandardScaler().fit(X_train); scY = StandardScaler().fit(y_train)
        X_train_s, y_train_s = scX.transform(X_train), scY.transform(y_train)
        X_valid_s, y_valid_s = scX.transform(X_valid), scY.transform(y_valid)

    console.print("\n[bold]--- Etapa 3: Iniciando Otimização de Hiperparâmetros ---[/bold]")
    eval_data = {
        'X_dev': X_dev, 'y_dev': y_dev,
        'X_train_s': X_train_s, 'y_train_s': y_train_s,
        'X_valid_s': X_valid_s, 'y_valid_s': y_valid_s,
    }
    ga_space = GenericSpace(model_type=config.MODEL_TO_OPTIMIZE, **eval_data)
    tuner = BaseGATuner(space=ga_space, seed=config.SEED, n_jobs=config.N_JOBS_GA, **config.GA_PARAMS)
    
    # A função run agora retorna o histórico das gerações
    best_gene, best_score_val, generation_history = tuner.run()
    
    # Imprime o resumo das gerações APÓS a barra de progresso terminar
    console.print("\n[bold]Resumo do Progresso das Gerações:[/bold]")
    for item in generation_history:
        console.print(f"  Geração {item['generation']}/{item['total_generations']} | Melhor R²: [bold yellow]{item['best_r2_in_gen']:.4f}[/bold yellow]")

    best_config = ga_space.decode(best_gene)
    final_results = {'best_validation_r2': -best_score_val, 'best_config_found': best_config}
    logging.info("Otimização Concluída", extra={'json_fields': final_results})
    console.print("\n[bold]--- Etapa 4: Otimização Concluída ---[/bold]")
    console.print(f"Melhor Score na Validação (R²): [bold green]{-best_score_val:.4f}[/bold green]")
    console.print("[bold]Melhor Configuração Encontrada:[/bold]")

    # --- INÍCIO DA CORREÇÃO ---
    # Cria uma cópia do dicionário para torná-lo 'imprimível'
    printable_config = best_config.copy()

    # Se a chave 'act' (de activation) existir, converte seu valor para o nome em texto
    if 'act' in printable_config:
        printable_config['act'] = printable_config['act'].__name__

    # Agora usamos a cópia imprimível para visualização
    console.print(Syntax(json.dumps(printable_config, indent=4), "json", theme="monokai", line_numbers=False))

    # ... (o resto do código continua o mesmo, usando console.print para mensagens)
    console.print("\n[bold]--- Etapa 5: Treinando o Modelo Final ---[/bold]")
    space_class = config.MODEL_CLASSES[config.MODEL_TO_OPTIMIZE]
    final_params = {'use_weights': config.USE_WEIGHTS, **config.WEIGHT_PARAMS}
    scX_final = StandardScaler().fit(X_dev); X_dev_s = scX_final.transform(X_dev); X_test_s = scX_final.transform(X_test)
    scY_final = StandardScaler().fit(y_dev); y_dev_s = scY_final.transform(y_dev); y_test_s_ignored = scY_final.transform(y_test)
    final_space = space_class(X_dev_s, y_dev_s, X_test_s, y_test_s_ignored, **final_params)
    final_model, train_hist, val_hist = final_space._train_model(best_config, rep=0)
    console.print("Treinamento finalizado.")
    
    analyze_and_plot(final_model, scY_final, X_dev_s, y_dev, X_test_s, y_test, run_results_path, train_hist, val_hist)

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()