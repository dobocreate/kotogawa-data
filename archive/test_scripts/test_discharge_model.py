#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
放流量予測モデルのテストと評価
"""

import pandas as pd
import numpy as np
from discharge_prediction_model import DischargePredictionModel
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# 日本語フォント設定
plt.rcParams['font.sans-serif'] = ['Yu Gothic', 'Hiragino Sans', 'Meirio', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def evaluate_model():
    """モデルの評価と結果の可視化"""
    print("放流量予測モデルの評価")
    print("=" * 60)
    
    # データ読み込み
    data_file = '統合データ_水位ダム_20250730_205325.csv'
    df = pd.read_csv(data_file, encoding='utf-8')
    print(f"データ読み込み完了: {len(df)}行")
    
    # モデルの初期化と訓練
    model = DischargePredictionModel()
    
    # データを訓練用とテスト用に分割
    split_idx = int(len(df) * 0.8)
    train_data = df.iloc[:split_idx]
    
    print(f"\n訓練データ: {len(train_data)}行")
    
    # モデル訓練
    model.train(train_data)
    
    # 訓練結果の表示
    print("\n=== 訓練結果サマリー ===")
    for step in model.prediction_steps:
        if step in model.validation_results:
            result = model.validation_results[step]
            print(f"\n{step*10}分先予測モデル:")
            print(f"  交差検証RMSE: {result['cv_rmse']:.2f} ± {result['cv_std']:.2f} m³/s")
            
            # 特徴量重要度（上位5つ）
            importance = result['feature_importance']
            sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
            print("  重要な特徴量（上位5つ）:")
            for feat, imp in sorted_features:
                print(f"    {feat}: {imp:.3f}")
    
    # 2時間先モデルの詳細分析
    print("\n=== 2時間先予測モデルの詳細分析 ===")
    step_2h = 12  # 120分 = 12ステップ
    
    if step_2h in model.validation_results:
        result = model.validation_results[step_2h]
        
        # 特徴量重要度の可視化
        importance = result['feature_importance']
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        features, importances = zip(*sorted_features)
        
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(features)), importances, color='steelblue')
        plt.yticks(range(len(features)), features)
        plt.xlabel('重要度')
        plt.title('特徴量重要度（2時間先予測モデル）')
        plt.tight_layout()
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plt.savefig(f'discharge_feature_importance_2h_{timestamp}.png', dpi=150)
        plt.close()
        print(f"\n特徴量重要度を保存: discharge_feature_importance_2h_{timestamp}.png")
    
    # 予測性能の時間変化
    plt.figure(figsize=(10, 6))
    
    steps = []
    rmse_means = []
    rmse_stds = []
    
    for step in sorted(model.prediction_steps):
        if step in model.validation_results:
            result = model.validation_results[step]
            steps.append(step * 10)  # 分に変換
            rmse_means.append(result['cv_rmse'])
            rmse_stds.append(result['cv_std'])
    
    steps = np.array(steps)
    rmse_means = np.array(rmse_means)
    rmse_stds = np.array(rmse_stds)
    
    plt.errorbar(steps, rmse_means, yerr=rmse_stds, marker='o', capsize=5, 
                linewidth=2, markersize=8)
    plt.xlabel('予測時間（分）')
    plt.ylabel('RMSE (m³/s)')
    plt.title('予測精度の時間変化')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(f'discharge_prediction_performance_{timestamp}.png', dpi=150)
    plt.close()
    print(f"予測性能グラフを保存: discharge_prediction_performance_{timestamp}.png")
    
    # 分析結果のまとめ
    print("\n=== 分析結果のまとめ ===")
    print("\n1. 遅延時間の効果:")
    print("   - 120分前の降雨強度が2時間先予測に寄与")
    print("   - 過去2時間の平均・最大降雨も重要な特徴量")
    
    print("\n2. 貯水位の影響:")
    print("   - 貯水位関連の特徴量は相対的に重要度が低い")
    print("   - 現在の放流量が最も重要な予測因子")
    
    print("\n3. 予測精度:")
    print(f"   - 10分先: RMSE約{model.validation_results[1]['cv_rmse']:.0f} m³/s")
    print(f"   - 2時間先: RMSE約{model.validation_results[12]['cv_rmse']:.0f} m³/s")
    
    # モデルの保存
    model_file = f'discharge_prediction_model_{timestamp}.pkl'
    model.save_model(model_file)
    print(f"\nモデルを保存: {model_file}")
    
    # サンプル予測の実行
    print("\n=== サンプル予測 ===")
    # 訓練データの最後の時点で予測
    sample_idx = len(train_data) - 1
    sample_time = pd.to_datetime(train_data.iloc[sample_idx]['時刻'])
    
    try:
        predictions = model.predict(sample_time, train_data, prediction_hours=2)
        print(f"\n{sample_time} からの2時間先予測:")
        print(predictions.head(12))  # 最初の2時間分
    except Exception as e:
        print(f"サンプル予測でエラー: {e}")
    
    return model


if __name__ == "__main__":
    model = evaluate_model()
    print("\n評価完了！")