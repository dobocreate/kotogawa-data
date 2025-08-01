#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
水位予測モデルの学習スクリプト
実データを使用してモデルを学習し、保存する
"""

import pandas as pd
import numpy as np
from datetime import datetime
import sys
from pathlib import Path

# モデルクラスのインポート
from water_level_prediction_model import WaterLevelPredictor


def load_and_preprocess_data(file_path):
    """データの読み込みと前処理"""
    print(f"\n=== データ読み込み ===")
    print(f"ファイル: {file_path}")
    
    # データ読み込み
    df = pd.read_csv(file_path, encoding='utf-8')
    print(f"読み込み完了: {len(df)}行")
    
    # 時刻をdatetimeに変換
    df['時刻'] = pd.to_datetime(df['時刻'])
    
    # 必要なカラムの確認
    required_cols = ['時刻', '水位_水位', 'ダム_全放流量']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"必要なカラム '{col}' が見つかりません")
    
    # 欠損値の処理
    print(f"\n欠損値処理前:")
    print(f"  水位欠損数: {df['水位_水位'].isna().sum()}")
    print(f"  放流量欠損数: {df['ダム_全放流量'].isna().sum()}")
    
    # 前方補完
    df['水位_水位'].fillna(method='ffill', inplace=True)
    df['ダム_全放流量'].fillna(method='ffill', inplace=True)
    
    # それでも残る欠損値は削除
    df.dropna(subset=['水位_水位', 'ダム_全放流量'], inplace=True)
    
    print(f"\n前処理後のデータ数: {len(df)}")
    
    # データの統計情報
    print(f"\n=== データ統計 ===")
    print(f"期間: {df['時刻'].min()} ～ {df['時刻'].max()}")
    print(f"放流量: {df['ダム_全放流量'].min():.1f} ～ {df['ダム_全放流量'].max():.1f} m³/s")
    print(f"水位: {df['水位_水位'].min():.2f} ～ {df['水位_水位'].max():.2f} m")
    
    # フィルタリング条件の適用
    mask = (df['ダム_全放流量'] >= 150) & (df['水位_水位'] >= 3.0)
    df_filtered = df[mask].copy()
    print(f"\nフィルタリング後（放流量≥150, 水位≥3.0）: {len(df_filtered)}行 ({len(df_filtered)/len(df)*100:.1f}%)")
    
    return df, df_filtered


def split_train_test_data(df, test_ratio=0.2):
    """時系列データの訓練・テスト分割"""
    # 時系列なので、後ろのデータをテストに使用
    split_idx = int(len(df) * (1 - test_ratio))
    
    df_train = df[:split_idx].copy()
    df_test = df[split_idx:].copy()
    
    print(f"\n=== データ分割 ===")
    print(f"訓練データ: {len(df_train)}行 ({df_train['時刻'].min()} ～ {df_train['時刻'].max()})")
    print(f"テストデータ: {len(df_test)}行 ({df_test['時刻'].min()} ～ {df_test['時刻'].max()})")
    
    return df_train, df_test


def evaluate_model(predictor, X_test, y_test):
    """モデルの評価"""
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    print("\n=== モデル評価 ===")
    
    # 予測
    y_pred = predictor.predict(X_test)
    
    # 各時刻での評価
    print("\n時刻別の予測精度:")
    print("時刻(分) | RMSE  | MAE   | R²")
    print("-" * 35)
    
    for i in range(y_test.shape[1]):
        rmse = np.sqrt(mean_squared_error(y_test[:, i], y_pred[:, i]))
        mae = mean_absolute_error(y_test[:, i], y_pred[:, i])
        r2 = r2_score(y_test[:, i], y_pred[:, i])
        
        print(f"{(i+1)*10:3d}     | {rmse:.3f} | {mae:.3f} | {r2:.3f}")
    
    # 全体の評価
    rmse_all = np.sqrt(mean_squared_error(y_test.flatten(), y_pred.flatten()))
    mae_all = mean_absolute_error(y_test.flatten(), y_pred.flatten())
    r2_all = r2_score(y_test.flatten(), y_pred.flatten())
    
    print(f"\n全体の精度:")
    print(f"RMSE: {rmse_all:.4f} m")
    print(f"MAE: {mae_all:.4f} m")
    print(f"R²: {r2_all:.4f}")
    
    return y_pred


def analyze_feature_importance(predictor):
    """特徴量の重要度分析（LightGBMの場合）- 修正版"""
    if predictor.model_type == 'lightgbm' and predictor.is_fitted:
        print("\n=== 特徴量重要度 ===")
        
        # splitタイプ（分岐回数）で集計 - より安定
        importance_sum = np.zeros(len(predictor.feature_names))
        valid_models = 0
        
        for i, model in enumerate(predictor.model):
            try:
                # splitタイプは全特徴量の重要度を返す
                importance = model.feature_importance(importance_type='split')
                if len(importance) == len(predictor.feature_names):
                    importance_sum += importance
                    valid_models += 1
            except:
                continue
        
        if valid_models > 0:
            # 平均化
            importance_avg = importance_sum / valid_models
            
            # ソートして表示
            feature_importance = pd.DataFrame({
                'feature': predictor.feature_names,
                'importance': importance_avg
            }).sort_values('importance', ascending=False)
            
            print(f"\nTop 10 重要な特徴量（{valid_models}/{len(predictor.model)}モデルの平均）:")
            for idx, row in feature_importance.head(10).iterrows():
                if row['importance'] > 0:
                    print(f"{row['feature']:30s}: {row['importance']:8.0f}")
            
            # カテゴリ別の重要度集計
            print("\n特徴量カテゴリ別の重要度:")
            category_importance = {
                '現在値': 0,
                '過去の放流量': 0,
                '過去の水位': 0,
                '統計量': 0,
                'その他': 0
            }
            
            for feature, importance in zip(predictor.feature_names, importance_avg):
                if 'current' in feature:
                    category_importance['現在値'] += importance
                elif 'discharge_lag' in feature:
                    category_importance['過去の放流量'] += importance
                elif 'water_level_lag' in feature:
                    category_importance['過去の水位'] += importance
                elif 'mean' in feature or 'std' in feature or 'trend' in feature:
                    category_importance['統計量'] += importance
                else:
                    category_importance['その他'] += importance
            
            total_importance = sum(category_importance.values())
            if total_importance > 0:
                for category, imp in sorted(category_importance.items(), key=lambda x: x[1], reverse=True):
                    percentage = imp / total_importance * 100
                    print(f"  {category:15s}: {percentage:5.1f}%")
        else:
            print("特徴量重要度を取得できませんでした")


def main():
    """メイン処理"""
    print("=== 水位予測モデル学習 ===")
    print(f"実行時刻: {datetime.now()}")
    
    # 1. データの読み込みと前処理
    file_path = "統合データ_水位ダム_20250730_205325.csv"
    
    if not Path(file_path).exists():
        print(f"エラー: ファイル '{file_path}' が見つかりません")
        print("適切なCSVファイルのパスを指定してください")
        return
    
    df_all, df_filtered = load_and_preprocess_data(file_path)
    
    # 2. 訓練・テストデータの分割
    df_train, df_test = split_train_test_data(df_filtered, test_ratio=0.2)
    
    # 3. モデルの初期化
    print("\n=== モデル初期化 ===")
    predictor = WaterLevelPredictor(model_type='lightgbm')
    
    # 4. 特徴量とターゲットの準備
    print("\n=== 特徴量作成 ===")
    print("訓練データの特徴量を作成中...")
    X_train, y_train = predictor.prepare_training_data(df_train)
    print(f"訓練データ: {X_train.shape[0]}サンプル, {X_train.shape[1]}特徴量")
    
    print("\nテストデータの特徴量を作成中...")
    X_test, y_test = predictor.prepare_training_data(df_test)
    print(f"テストデータ: {X_test.shape[0]}サンプル")
    
    # 5. クロスバリデーション（オプション）
    do_cv = input("\nクロスバリデーションを実行しますか？ (y/n): ").lower() == 'y'
    
    if do_cv:
        print("\n=== 時系列クロスバリデーション ===")
        cv_results = predictor.time_series_cross_validation(X_train, y_train, n_splits=3)
    
    # 6. モデルの学習
    print("\n=== モデル学習 ===")
    # 検証用データの分割（訓練データの最後20%）
    val_split = int(len(X_train) * 0.8)
    X_train_split = X_train[:val_split]
    y_train_split = y_train[:val_split]
    X_val = X_train[val_split:]
    y_val = y_train[val_split:]
    
    print(f"訓練: {len(X_train_split)}サンプル, 検証: {len(X_val)}サンプル")
    
    # 学習実行
    predictor.fit(X_train_split, y_train_split, X_val, y_val)
    
    # 7. モデルの評価
    y_pred = evaluate_model(predictor, X_test, y_test)
    
    # 8. 特徴量重要度の分析
    analyze_feature_importance(predictor)
    
    # 9. モデルの保存
    model_path = f"water_level_predictor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
    predictor.save_model(model_path)
    
    # 10. サンプル予測の表示
    print("\n=== サンプル予測（最初の5件） ===")
    print("実際の水位 vs 予測水位（3時間後）")
    for i in range(min(5, len(y_test))):
        actual = y_test[i, -1]  # 3時間後の実際の水位
        predicted = y_pred[i, -1]  # 3時間後の予測水位
        error = predicted - actual
        print(f"サンプル{i+1}: 実際={actual:.2f}m, 予測={predicted:.2f}m, 誤差={error:+.3f}m")
    
    print("\n=== 学習完了 ===")
    print(f"モデルを保存しました: {model_path}")
    
    # オンライン学習のデモ
    print("\n=== オンライン学習デモ ===")
    do_online = input("オンライン学習のデモを実行しますか？ (y/n): ").lower() == 'y'
    
    if do_online and len(X_test) >= 100:
        print("\n最新100サンプルでオンライン更新を実行...")
        X_online = X_test[-100:]
        y_online = y_test[-100:]
        
        # 更新前の学習率
        print(f"更新前の学習率: {predictor.get_current_learning_rate():.6f}")
        
        # オンライン更新
        predictor.online_update(X_online, y_online)
        
        # 更新後の学習率
        print(f"更新後の学習率: {predictor.get_current_learning_rate():.6f}")
        print(f"更新回数: {predictor.update_count}")
        
        # 更新後のモデルを保存
        updated_model_path = f"water_level_predictor_updated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        predictor.save_model(updated_model_path)
        print(f"更新後のモデルを保存: {updated_model_path}")


if __name__ == "__main__":
    main()