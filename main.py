"""
Ponto de Entrada Principal do Framework de Otimização.

Este script orquestra todo o fluxo do experimento:
1. Configura o logging para console (Rich) e arquivo (JSON).
2. Carrega e prepara os dados.
3. Executa a otimização de hiperparâmetros com o Algoritmo Genético.
4. Salva o modelo treinado para uso futuro.
"""
import json
import logging
import multiprocessing as mp
import os
import time
import warnings
from datetime import datetime
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from pythonjsonlogger.jsonlogger import JsonFormatter
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import config
from analysis import analyze_and_plot
from data_loader import load_and_prepare_data
from evaluation import GenericSpace, MODEL_CLASSES
from optimization.ga_tuner import BaseGATuner

warnings.filterwarnings("ignore")


def setup_logging_and_results_dir() -> Tuple[str, Console]:
    """Configura o logging para a execução e cria uma pasta de resultados única."""
    console = Console()
    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_results_path = os.path.join("results", f"run_{run_timestamp}")
    os.makedirs(run_results_path, exist_ok=True)

    json_log_filepath = os.path.join(run_results_path, "run_log.jsonl")
    log_handler = logging.FileHandler(filename=json_log_filepath)
    log_handler.setFormatter(JsonFormatter())

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.addHandler(log_handler)

    console.print(Panel(f"INICIANDO NOVA EXECUÇÃO\nResultados salvos em: [cyan]{run_results_path}[/cyan]", title="[bold green]Otimizador de CBR[/bold green]", expand=False))
    return run_results_path, console


def log_initial_config(console: Console) -> None:
    """Imprime e loga a configuração inicial do experimento."""
    run_config = {
        "model_to_optimize": config.MODEL_TO_OPTIMIZE, "dataset": config.DATASET,
        "validation_method": config.VALIDATION_METHOD, "use_weights": config.USE_WEIGHTS,
        "seed": config.SEED, "ga_params": config.GA_PARAMS,
    }
    logging.info("Configurações da Execução", extra={"json_fields": run_config})
    console.print("[bold]Configurações da Execução:[/bold]")
    console.print(Syntax(json.dumps(run_config, indent=4), "json", theme="monokai"))


def run_optimization(
    console: Console, eval_data: Dict[str, Any]
) -> Tuple[Dict[str, Any], List[Dict], List[float]]:
    """
    Executa a otimização com o Algoritmo Genético.
    
    Returns:
        Tuple[Dict, List, List]: A melhor configuração, o histórico de 
            gerações e o histórico de fitness.
    """
    console.print("\n[bold]--- Etapa 3: Iniciando Otimização de Hiperparâmetros ---[/bold]")
    ga_space = GenericSpace(model_type=config.MODEL_TO_OPTIMIZE, **eval_data)
    tuner = BaseGATuner(
        space=ga_space, seed=config.SEED, n_jobs=config.N_JOBS_GA, **config.GA_PARAMS
    )
    best_gene, best_score, generation_history, fitness_history, termination_reason = tuner.run()

    console.print("\n[bold]Resumo do Progresso das Gerações:[/bold]")
    for item in generation_history:
        console.print(f"  Geração {item['generation']}/{item['total_generations']} | "f"Melhor R²: [bold yellow]{item['best_r2_in_gen']:.4f}[/bold yellow]")

    best_config = ga_space.decode(best_gene)
    final_results = {
        "best_validation_r2": -best_score, 
        "best_config_found": best_config,
        "termination_reason": termination_reason
    }
    logging.info("Otimização Concluída", extra={"json_fields": final_results})

    console.print("\n[bold]--- Etapa 4: Otimização Concluída ---[/bold]")
    console.print(f"Motivo da Parada: [italic]{termination_reason}[/italic]")
    console.print(f"Melhor Score na Validação (R²): [bold green]{-best_score:.4f}[/bold green]")
    console.print("[bold]Melhor Configuração Encontrada:[/bold]")
    
    printable_config = best_config.copy()
    if "act" in printable_config:
        printable_config["act"] = printable_config["act"].__name__
        
    console.print(Syntax(json.dumps(printable_config, indent=4), "json", theme="monokai"))
    return best_config, generation_history, fitness_history

def format_duration(seconds: float) -> str:
    """Formata um tempo em segundos para uma string legível (minutos e segundos)."""
    mins, secs = divmod(seconds, 60)
    return f"{int(mins)} minutos e {secs:.2f} segundos"

def main() -> None:
    """Ponto de entrada principal que orquestra todo o fluxo do experimento."""
    start_time = time.time()

    run_results_path, console = setup_logging_and_results_dir()
    config.set_seeds()
    log_initial_config(console)

    console.print("\n[bold]--- Etapa 1: Preparando Dados ---[/bold]")
    Xdf, y = load_and_prepare_data(config.CSV_PATH, config.DATASET)
    if Xdf.empty: return

    console.print(f"Dataset carregado com {Xdf.shape[0]} amostras e {Xdf.shape[1]} features.")
    data_stats = {"dataset_shape_after_cleaning": Xdf.shape, "target_variable_stats": y.describe().to_dict()}
    logging.info("Estatísticas do Dataset Utilizado", extra={"json_fields": data_stats})

    console.print(f"\n[bold]--- Etapa 2: Configurando Estratégia de Validação: {config.VALIDATION_METHOD} ---[/bold]")
    X_dev, X_test, y_dev, y_test = train_test_split(
        Xdf.values.astype(np.float32), y.values.astype(np.float32).reshape(-1, 1),
        test_size=0.15, random_state=config.SEED
    )
    X_train_s, y_train_s, X_valid_s, y_valid_s = (np.array([]),) * 4
    if config.VALIDATION_METHOD == "Holdout":
        X_train, X_valid, y_train, y_valid = train_test_split(X_dev, y_dev, test_size=(0.15 / 0.85), random_state=config.SEED)
        scaler_x_h = StandardScaler().fit(X_train); scaler_y_h = StandardScaler().fit(y_train)
        X_train_s, X_valid_s = scaler_x_h.transform(X_train), scaler_x_h.transform(X_valid)
        y_train_s, y_valid_s = scaler_y_h.transform(y_train), scaler_y_h.transform(y_valid)

    eval_data = {"X_dev": X_dev, "y_dev": y_dev, "X_train_s": X_train_s, "y_train_s": y_train_s, "X_valid_s": X_valid_s, "y_valid_s": y_valid_s}
    best_config, generation_history, fitness_history = run_optimization(console, eval_data)
    
    console.print("\n[bold]--- Etapa 5: Avaliação Robusta e Análise Gráfica ---[/bold]")
    scaler_x_final = StandardScaler().fit(X_dev)
    scaler_y_final = StandardScaler().fit(y_dev)

    analyze_and_plot(
        best_config=best_config, space_class=MODEL_CLASSES[config.MODEL_TO_OPTIMIZE],
        scaler_x=scaler_x_final, scaler_y=scaler_y_final, X_dev_real=X_dev,
        y_dev_real=y_dev, X_test_real=X_test, y_test_real=y_test,
        results_path=run_results_path, feature_names=Xdf.columns.tolist(),
        generation_history=generation_history, fitness_history=fitness_history
    )
    
    console.print("\n[bold]--- Etapa 6: Produção do Modelo Final para Distribuição ---[/bold]")
    X_dev_s = scaler_x_final.transform(X_dev)
    y_dev_s = scaler_y_final.transform(y_dev)
    
    final_space = MODEL_CLASSES[config.MODEL_TO_OPTIMIZE](
        X_dev_s, y_dev_s, X_dev_s, y_dev_s, 
        use_weights=config.USE_WEIGHTS, **config.WEIGHT_PARAMS
    )
    final_model_canonical, _, _ = final_space._train_model(best_config, rep=config.SEED)
    
    model_filename = "final_model.pth" if config.MODEL_TO_OPTIMIZE == "MLP" else "final_model.joblib"
    model_save_path = os.path.join(run_results_path, model_filename)
    if config.MODEL_TO_OPTIMIZE == "MLP":
        torch.save(final_model_canonical.state_dict(), model_save_path)
    else:
        joblib.dump(final_model_canonical, model_save_path)
    
    console.print(f"Modelo canônico final salvo em: [green]{model_save_path}[/green]")
    logging.info(f"Modelo canônico final salvo em: {model_save_path}")
    
    end_time = time.time()
    total_duration = end_time - start_time
    duration_str = format_duration(total_duration)

    console.print(Panel(f"Execução Concluída em: [bold yellow]{duration_str}[/bold yellow]", title="[bold]FIM[/bold]", expand=False))
    logging.info("Execução finalizada.", extra={"json_fields": {"total_execution_time_seconds": total_duration}})

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()