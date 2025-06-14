# analysis.py

import torch
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import config
import logging
import os # <-- Adicionado

def analyze_and_plot(final_model, scY_final, X_dev_s, y_dev, X_test_s, y_test, results_path, train_hist=None, val_hist=None): # <-- results_path adicionado
    """Gera métricas de performance e gráficos de avaliação para o modelo final."""
    logging.info("\n--- Etapa 6: Análise Detalhada do Melhor Modelo ---")
    sns.set_style("whitegrid")
    
    # ... (Cálculo de previsões e métricas continua o mesmo) ...
    if config.MODEL_TO_OPTIMIZE == 'MLP':
        with torch.no_grad():
            pred_dev_s = final_model(torch.from_numpy(X_dev_s).to(config.DEVICE))
            pred_test_s = final_model(torch.from_numpy(X_test_s).to(config.DEVICE))
        pred_dev_real = scY_final.inverse_transform(pred_dev_s.cpu().numpy())
        pred_test_real = scY_final.inverse_transform(pred_test_s.cpu().numpy())
    else:  # RandomForest
        pred_dev_s = final_model.predict(X_dev_s)
        pred_test_s = final_model.predict(X_test_s)
        pred_dev_real = scY_final.inverse_transform(pred_dev_s.reshape(-1, 1))
        pred_test_real = scY_final.inverse_transform(pred_test_s.reshape(-1, 1))
    
    logging.info("\n--- Métricas de Performance ---")
    logging.info(f"Treino (Dev Set): R²={r2_score(y_dev, pred_dev_real):.4f} | MAE={mean_absolute_error(y_dev, pred_dev_real):.2f}")
    logging.info(f"Teste:            R²={r2_score(y_test, pred_test_real):.4f} | MAE={mean_absolute_error(y_test, pred_test_real):.2f}")

    # --- INÍCIO DA MODIFICAÇÃO: SALVAR GRÁFICOS EM VEZ DE MOSTRAR ---
    
    # Curva de Aprendizado (Apenas para MLP)
    if config.MODEL_TO_OPTIMIZE == 'MLP' and train_hist:
        plt.figure(figsize=(8, 5))
        plt.plot(train_hist, label='Loss de Treino')
        plt.plot(val_hist, label='Loss de Validação (Holdout)')
        plt.xlabel("Épocas"); plt.ylabel("Loss (MAE Normalizado)"); plt.title("Curva de Aprendizado do Modelo Final")
        plt.legend()
        filepath = os.path.join(results_path, "learning_curve.png")
        plt.savefig(filepath)
        plt.close() # Fecha a figura para não consumir memória
        logging.info(f"Gráfico 'Curva de Aprendizado' salvo em: {filepath}")

    # Real vs. Predito e Distribuição de Resíduos
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.scatterplot(x=y_test.flatten(), y=pred_test_real.flatten(), ax=axes[0])
    axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    axes[0].set_xlabel("CBR Real"); axes[0].set_ylabel("CBR Predito"); axes[0].set_title("Real vs. Predito (Teste)")
    axes[0].set_aspect('equal', adjustable='box')
    residuos = y_test.flatten() - pred_test_real.flatten()
    sns.histplot(residuos, kde=True, ax=axes[1])
    axes[1].set_xlabel("Resíduo (Real - Predito)"); axes[1].set_title("Distribuição dos Resíduos")
    plt.tight_layout()
    filepath = os.path.join(results_path, "real_vs_predicted_and_residuals.png")
    plt.savefig(filepath)
    plt.close()
    logging.info(f"Gráfico 'Real vs Predito e Resíduos' salvo em: {filepath}")

    # Resíduos vs. Predito
    plt.figure(figsize=(7, 6))
    sns.scatterplot(x=pred_test_real.flatten(), y=residuos)
    plt.axhline(0, color='r', ls='--'); plt.xlabel("CBR Predito"); plt.ylabel("Resíduo")
    plt.title("Resíduos vs. Valores Preditos")
    filepath = os.path.join(results_path, "residuals_vs_predicted.png")
    plt.savefig(filepath)
    plt.close()
    logging.info(f"Gráfico 'Resíduos vs Preditos' salvo em: {filepath}")
    # --- FIM DA MODIFICAÇÃO ---