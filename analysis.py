"""
Módulo de Análise de Resultados e Visualização.

Este módulo contém uma suíte de funções para calcular métricas de performance,
comparar o desempenho de treino e teste, e gerar visualizações detalhadas
para a análise do modelo final. A avaliação final é feita de forma robusta,
treinando o melhor modelo encontrado múltiplas vezes.
"""
import logging
import os
from typing import Any, Dict, List, Optional, Type

import joblib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from numpy.typing import NDArray
from rich.console import Console
from rich.table import Table
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

import config

N_FINAL_EVALS = 10  # Número de vezes para treinar e avaliar o modelo final

def calculate_regression_metrics(y_true: NDArray, y_pred: NDArray) -> Dict[str, float]:
    """Calcula um conjunto de métricas de regressão (R², MAE, RMSE, MAPE)."""
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    r2 = r2_score(y_true_flat, y_pred_flat)
    mae = mean_absolute_error(y_true_flat, y_pred_flat)
    rmse = np.sqrt(mean_squared_error(y_true_flat, y_pred_flat))

    non_zero_mask = y_true_flat != 0
    mape = (
        np.mean(np.abs((y_true_flat[non_zero_mask] - y_pred_flat[non_zero_mask]) / y_true_flat[non_zero_mask])) * 100
        if np.any(non_zero_mask) else np.nan
    )
    return {"R²": r2, "MAE": mae, "RMSE": rmse, "MAPE (%)": mape}


def _get_predictions(
    model: Any, X_data_scaled: NDArray, scaler_y: StandardScaler
) -> NDArray:
    """Gera previsões e reverte a escala para os valores originais."""
    if config.MODEL_TO_OPTIMIZE == "MLP":
        model.eval()
        with torch.no_grad():
            tensor_X = torch.from_numpy(X_data_scaled).to(config.DEVICE)
            predictions_scaled = model(tensor_X)
        return scaler_y.inverse_transform(predictions_scaled.cpu().numpy())
    else:
        predictions_scaled = model.predict(X_data_scaled)
        return scaler_y.inverse_transform(predictions_scaled.reshape(-1, 1))


def _save_plot(figure: plt.Figure, results_path: str, filename: str) -> None:
    """Salva uma figura do Matplotlib em um arquivo e a fecha."""
    filepath = os.path.join(results_path, filename)
    figure.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close(figure)
    logging.info(f"Gráfico salvo em: {filepath}")


def _analyze_performance_by_bins(
    y_true: NDArray, y_pred: NDArray, console: Console
) -> None:
    """Analisa e exibe a performance do modelo por faixas de CBR."""
    df = pd.DataFrame({"Real": y_true.flatten(), "Predito": y_pred.flatten()})
    bins = [0, 15, 30, 50, 80, np.inf]
    labels = ["0-15", "16-30", "31-50", "51-80", ">80"]
    df["CBR_Faixa"] = pd.cut(df["Real"], bins=bins, labels=labels, right=True)

    console.print("\n[bold]--- Análise de Performance por Faixa de CBR (Teste) ---[/bold]")
    table = Table(title="Métricas por Faixa de CBR")
    table.add_column("Faixa CBR", style="cyan")
    table.add_column("Nº Amostras", style="magenta")
    table.add_column("R²", style="green")
    table.add_column("RMSE", style="yellow")

    for label in labels:
        subset = df[df["CBR_Faixa"] == label]
        if not subset.empty:
            metrics = calculate_regression_metrics(
                subset[["Real"]].values, subset[["Predito"]].values
            )
            table.add_row(
                label, str(len(subset)), f"{metrics['R²']:.3f}", f"{metrics['RMSE']:.2f}"
            )
    console.print(table)

def _plot_ga_convergence(fitness_history: List[float], results_path: str) -> None:
    """
    Gera e salva o gráfico de convergência do Algoritmo Genético com
    uma estética aprimorada.
    """
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(12, 7))

    r2_history = [-f for f in fitness_history]
    generations = range(1, len(r2_history) + 1)

    # Plota a linha de evolução
    ax.plot(
        generations, r2_history, marker='o', markersize=6,
        linestyle='-', linewidth=2, color='#007ACC',
        label='Melhor R² por Geração'
    )

    # Destaca o melhor ponto encontrado em toda a busca
    best_gen_idx = np.argmax(r2_history)
    best_r2_value = r2_history[best_gen_idx]
    ax.plot(
        generations[best_gen_idx], best_r2_value,
        marker='.', markersize=15, color="#FF0707",
        linestyle='none', label=f'Melhor R² Global: {best_r2_value:.4f}'
    )

    # Garante que os ticks do eixo X sejam inteiros e não muito aglomerados
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins='auto'))
    plt.xticks(rotation=-30, ha="right")  # Inclina os rótulos

    # Melhora os títulos e rótulos
    ax.set_xlabel("Geração", fontsize=12)
    ax.set_ylabel("Melhor Fitness (R²)", fontsize=12)
    ax.set_title("Curva de Convergência do Algoritmo Genético", fontsize=14, fontweight='bold')
    ax.legend()
    sns.despine(fig=fig, ax=ax)  # Remove as bordas superiores e direitas
    fig.tight_layout() # Ajusta o layout para evitar cortes
    
    _save_plot(fig, results_path, "ga_convergence_curve.png")

def _plot_prediction_tracking(y_true: NDArray, y_pred: NDArray, results_path: str) -> None:
    """Gera e salva o gráfico de rastreamento de predições vs. valores reais."""
    fig = plt.figure(figsize=(15, 7))
    sample_indices = range(len(y_true))
    plt.plot(sample_indices, y_true.flatten(), label='Valores Reais', color='gray', marker='o', linestyle='-')
    plt.plot(sample_indices, y_pred.flatten(), label='Valores Preditos', color='darkorange', marker='x', linestyle='--')
    plt.xlabel("Índice da Amostra de Teste")
    plt.ylabel("CBR")
    plt.title("Rastreamento de Valores Preditos vs. Reais")
    plt.legend()
    plt.grid(True)
    _save_plot(fig, results_path, "prediction_tracking.png")


def analyze_and_plot(
    best_config: Dict[str, Any],
    space_class: Type,
    scaler_x: StandardScaler,
    scaler_y: StandardScaler,
    X_dev_real: NDArray,
    y_dev_real: NDArray,
    X_test_real: NDArray,
    y_test_real: NDArray,
    results_path: str,
    feature_names: List[str],
    generation_history: List[Dict],
    fitness_history: List[float],
) -> None:
    """
    Executa a suíte completa de análise, incluindo múltiplos treinamentos do
    modelo final para uma avaliação robusta de performance.
    """
    console = Console()
    sns.set_style("whitegrid")
    logging.info("\n--- Etapa 6: Análise Detalhada e Robusta do Melhor Modelo ---")
    
    X_dev_scaled = scaler_x.transform(X_dev_real)
    X_test_scaled = scaler_x.transform(X_test_real)
    y_dev_scaled = scaler_y.transform(y_dev_real)
    y_test_scaled_ignored = scaler_y.transform(y_test_real)

    dev_metrics_list, test_metrics_list, models_and_scores = [], [], []
    
    for i in tqdm(range(N_FINAL_EVALS), desc="Avaliando robustez do modelo final"):
        final_space = space_class(
            X_dev_scaled, y_dev_scaled, X_test_scaled, y_test_scaled_ignored, 
            use_weights=config.USE_WEIGHTS, **config.WEIGHT_PARAMS
        )
        model, train_hist, val_hist = final_space._train_model(best_config, rep=config.SEED + i)
        
        pred_dev_real = _get_predictions(model, X_dev_scaled, scaler_y)
        pred_test_real = _get_predictions(model, X_test_scaled, scaler_y)

        dev_metrics_list.append(calculate_regression_metrics(y_dev_real, pred_dev_real))
        test_metrics = calculate_regression_metrics(y_test_real, pred_test_real)
        test_metrics_list.append(test_metrics)
        models_and_scores.append({'model': model, 'r2_test': test_metrics['R²'], 'train_hist': train_hist, 'val_hist': val_hist})

    df_dev_metrics = pd.DataFrame(dev_metrics_list)
    df_test_metrics = pd.DataFrame(test_metrics_list)
    dev_mean, dev_std = df_dev_metrics.mean(), df_dev_metrics.std()
    test_mean, test_std = df_test_metrics.mean(), df_test_metrics.std()

    logging.info("Métricas de Performance (Média de 10 execuções)", extra={"json_fields": {
        "dev_set_mean": dev_mean.to_dict(), "dev_set_std": dev_std.to_dict(),
        "test_set_mean": test_mean.to_dict(), "test_set_std": test_std.to_dict()
    }})

    console.print("\n[bold]--- Tabela de Performance Robusta (Média ± Desvio Padrão) ---[/bold]")
    table = Table(title=f"Comparativo de Métricas ({N_FINAL_EVALS} execuções)")
    table.add_column("Métrica", style="cyan"); table.add_column("Treino (Dev Set)", style="yellow"); table.add_column("Teste", style="green")
    for metric_name in dev_mean.index:
        table.add_row(metric_name, f"{dev_mean[metric_name]:.3f} ± {dev_std[metric_name]:.3f}", f"{test_mean[metric_name]:.3f} ± {test_std[metric_name]:.3f}")
    console.print(table)

    models_and_scores.sort(key=lambda x: x['r2_test'])
    median_model_data = models_and_scores[len(models_and_scores) // 2]
    representative_model = median_model_data['model']
    console.print(f"\nGerando gráficos com base no modelo mediano (R² no Teste = {median_model_data['r2_test']:.3f})")

    _plot_ga_convergence(fitness_history, results_path)
    rep_pred_test_real = _get_predictions(representative_model, X_test_scaled, scaler_y)
    _plot_prediction_tracking(y_test_real, rep_pred_test_real, results_path)

    if config.MODEL_TO_OPTIMIZE == "RF":
        importances = representative_model.feature_importances_
        feature_df = pd.DataFrame({"Feature": feature_names, "Importância": importances}).sort_values(by="Importância", ascending=False)
        fig = plt.figure(figsize=(10, max(6, len(feature_names) * 0.4)))
        sns.barplot(x="Importância", y="Feature", data=feature_df, palette="viridis"); plt.title("Importância das Features (Random Forest)"); fig.tight_layout()
        _save_plot(fig, results_path, "feature_importance.png")
    
    if config.MODEL_TO_OPTIMIZE == "MLP":
        train_hist_median, val_hist_median = median_model_data['train_hist'], median_model_data['val_hist']
        if train_hist_median and val_hist_median:
            fig = plt.figure(figsize=(8, 5))
            plt.plot(train_hist_median, label="Loss de Treino (Modelo Mediano)"); plt.plot(val_hist_median, label="Loss de Validação (Modelo Mediano)")
            plt.xlabel("Épocas"); plt.ylabel("Loss (MAE Normalizado)"); plt.title("Curva de Aprendizado do Modelo Final"); plt.legend(); _save_plot(fig, results_path, "learning_curve.png")

    fig_preds, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.scatterplot(x=y_test_real.flatten(), y=rep_pred_test_real.flatten(), ax=axes[0])
    axes[0].plot([y_test_real.min(), y_test_real.max()], [y_test_real.min(), y_test_real.max()], "r--")
    axes[0].set(xlabel="CBR Real", ylabel="CBR Predito", title="Real vs. Predito (Teste - Modelo Mediano)"); axes[0].set_aspect("equal", "box")
    residuals = y_test_real.flatten() - rep_pred_test_real.flatten()
    sns.histplot(residuals, kde=True, ax=axes[1])
    axes[1].set(xlabel="Resíduo (Real - Predito)", title="Distribuição dos Resíduos"); fig_preds.tight_layout()
    _save_plot(fig_preds, results_path, "predictions_and_residuals.png")

    _analyze_performance_by_bins(y_test_real, rep_pred_test_real, console)