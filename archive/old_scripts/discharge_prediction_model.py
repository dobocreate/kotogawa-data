#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
放流量予測モデル
降雨強度（120分遅延）と貯水位を用いた機械学習による放流量予測
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from datetime import datetime, timedelta
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# 日本語フォント設定
plt.rcParams['font.sans-serif'] = ['Yu Gothic', 'Hiragino Sans', 'Meirio', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class DischargePredictionModel:
    """放流量予測モデル"""
    
    def __init__(self):
        """初期化"""
        # モデルパラメータ
        self.delay_minutes = 120  # 降雨強度の遅延時間（分）
        self.flood_prep_level = 38.00  # 洪水貯留準備水位（m）
        self.flood_max_level = 39.20   # 洪水時最高水位（m）
        
        # 機械学習モデル
        self.models = {}  # 予測時間ごとのモデル
        self.scalers = {}  # 特徴量の正規化
        
        # 特徴量の設定
        self.feature_names = [
            'rainfall_lag120',     # 120分前の降雨強度
            'rainfall_lag90',      # 90分前の降雨強度
            'rainfall_lag150',     # 150分前の降雨強度
            'rainfall_avg_2h',     # 過去2時間の平均降雨強度
            'rainfall_max_2h',     # 過去2時間の最大降雨強度
            'rainfall_change',     # 降雨強度の変化率
            'reservoir_level',     # 現在の貯水位
            'level_to_prep',       # 洪水貯留準備水位までの余裕
            'level_to_max',        # 洪水時最高水位までの余裕
            'level_ratio',         # 貯水位の正規化値
            'current_discharge',   # 現在の放流量
            'discharge_avg_1h',    # 過去1時間の平均放流量
            'discharge_change'     # 放流量の変化率
        ]
        
        # 予測対象時間（10分刻み）
        self.prediction_steps = [1, 2, 3, 6, 12]  # 10分, 20分, 30分, 1時間, 2時間先
        
        # データ保存用
        self.training_data = None
        self.validation_results = {}
        
    def prepare_features(self, df, target_time_idx):
        """
        特徴量の準備
        
        Parameters:
        -----------
        df : pd.DataFrame
            統合データ（時刻、降雨強度、貯水位、放流量を含む）
        target_time_idx : int
            予測対象時刻のインデックス
            
        Returns:
        --------
        features : dict
            特徴量の辞書
        """
        features = {}
        
        # 現在時刻の確認
        if target_time_idx < 0 or target_time_idx >= len(df):
            return None
            
        # 降雨強度関連の特徴量（遅延を考慮）
        lag_12 = int(self.delay_minutes / 10)  # 120分 = 12データポイント
        lag_9 = int(90 / 10)  # 90分
        lag_15 = int(150 / 10)  # 150分
        
        # 基本的な降雨特徴量
        if target_time_idx >= lag_12:
            features['rainfall_lag120'] = df.iloc[target_time_idx - lag_12]['ダム_60分雨量']
        else:
            features['rainfall_lag120'] = 0
            
        if target_time_idx >= lag_9:
            features['rainfall_lag90'] = df.iloc[target_time_idx - lag_9]['ダム_60分雨量']
        else:
            features['rainfall_lag90'] = 0
            
        if target_time_idx >= lag_15:
            features['rainfall_lag150'] = df.iloc[target_time_idx - lag_15]['ダム_60分雨量']
        else:
            features['rainfall_lag150'] = 0
        
        # 過去2時間の降雨統計
        if target_time_idx >= 12:
            past_2h_rainfall = df.iloc[target_time_idx-12:target_time_idx]['ダム_60分雨量']
            features['rainfall_avg_2h'] = past_2h_rainfall.mean()
            features['rainfall_max_2h'] = past_2h_rainfall.max()
        else:
            features['rainfall_avg_2h'] = 0
            features['rainfall_max_2h'] = 0
        
        # 降雨変化率（1時間前との比較）
        if target_time_idx >= lag_12 + 6:
            rain_current = features['rainfall_lag120']
            rain_1h_ago = df.iloc[target_time_idx - lag_12 - 6]['ダム_60分雨量']
            features['rainfall_change'] = rain_current - rain_1h_ago
        else:
            features['rainfall_change'] = 0
        
        # 貯水位関連の特徴量
        current_level = df.iloc[target_time_idx]['ダム_貯水位']
        features['reservoir_level'] = current_level
        features['level_to_prep'] = current_level - self.flood_prep_level
        features['level_to_max'] = self.flood_max_level - current_level
        features['level_ratio'] = (current_level - self.flood_prep_level) / (self.flood_max_level - self.flood_prep_level)
        
        # 放流量関連の特徴量
        features['current_discharge'] = df.iloc[target_time_idx]['ダム_全放流量']
        
        if target_time_idx >= 6:
            past_1h_discharge = df.iloc[target_time_idx-6:target_time_idx]['ダム_全放流量']
            features['discharge_avg_1h'] = past_1h_discharge.mean()
            features['discharge_change'] = features['current_discharge'] - df.iloc[target_time_idx-6]['ダム_全放流量']
        else:
            features['discharge_avg_1h'] = features['current_discharge']
            features['discharge_change'] = 0
            
        return features
    
    def create_training_data(self, df):
        """
        訓練データの作成
        
        Parameters:
        -----------
        df : pd.DataFrame
            統合データ
            
        Returns:
        --------
        X : np.array
            特徴量行列
        y : dict
            予測対象（各予測時間ごと）
        """
        # データフィルタリング（降雨強度>0、水位≥3.0、放流量≥150）
        mask = (df['ダム_60分雨量'] > 0) & (df['水位_水位'] >= 3.0) & (df['ダム_全放流量'] >= 150)
        valid_indices = df[mask].index
        
        X_list = []
        y_dict = {step: [] for step in self.prediction_steps}
        valid_indices_list = []
        
        for idx in valid_indices:
            # 特徴量の準備
            features = self.prepare_features(df, idx)
            if features is None:
                continue
                
            # 予測対象が存在するか確認
            valid_prediction = True
            for step in self.prediction_steps:
                if idx + step >= len(df):
                    valid_prediction = False
                    break
                    
            if not valid_prediction:
                continue
                
            # 特徴量を配列に変換
            feature_vector = [features[name] for name in self.feature_names]
            X_list.append(feature_vector)
            
            # 各予測時間の目標値
            for step in self.prediction_steps:
                y_dict[step].append(df.iloc[idx + step]['ダム_全放流量'])
                
            valid_indices_list.append(idx)
            
        X = np.array(X_list)
        y = {step: np.array(y_dict[step]) for step in self.prediction_steps}
        
        print(f"訓練データ作成完了: {len(X)}サンプル")
        
        return X, y, valid_indices_list
    
    def train(self, df, n_splits=5):
        """
        モデルの訓練
        
        Parameters:
        -----------
        df : pd.DataFrame
            統合データ
        n_splits : int
            時系列交差検証の分割数
        """
        print("=== モデル訓練開始 ===")
        
        # 訓練データの作成
        X, y, valid_indices = self.create_training_data(df)
        
        # 時系列交差検証
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        # 各予測時間ごとにモデルを訓練
        for step in self.prediction_steps:
            print(f"\n{step*10}分先予測モデルの訓練...")
            
            # スケーラーの初期化
            self.scalers[step] = StandardScaler()
            
            # 交差検証スコア
            cv_scores = []
            
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[step][train_idx], y[step][val_idx]
                
                # 正規化
                X_train_scaled = self.scalers[step].fit_transform(X_train)
                X_val_scaled = self.scalers[step].transform(X_val)
                
                # モデル訓練
                model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=15,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    random_state=42,
                    n_jobs=-1
                )
                model.fit(X_train_scaled, y_train)
                
                # 予測と評価
                y_pred = model.predict(X_val_scaled)
                rmse = np.sqrt(mean_squared_error(y_val, y_pred))
                cv_scores.append(rmse)
            
            # 最終モデルの訓練（全データ使用）
            X_scaled = self.scalers[step].fit_transform(X)
            self.models[step] = RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1
            )
            self.models[step].fit(X_scaled, y[step])
            
            # 訓練結果の保存
            self.validation_results[step] = {
                'cv_rmse': np.mean(cv_scores),
                'cv_std': np.std(cv_scores),
                'feature_importance': dict(zip(self.feature_names, self.models[step].feature_importances_))
            }
            
            print(f"  交差検証RMSE: {np.mean(cv_scores):.2f} ± {np.std(cv_scores):.2f} m³/s")
            
            # 特徴量重要度の表示（上位5つ）
            importance_sorted = sorted(
                self.validation_results[step]['feature_importance'].items(),
                key=lambda x: x[1],
                reverse=True
            )
            print("  重要な特徴量（上位5つ）:")
            for feat, imp in importance_sorted[:5]:
                print(f"    {feat}: {imp:.3f}")
    
    def predict(self, current_time, current_data, prediction_hours=2):
        """
        放流量予測
        
        Parameters:
        -----------
        current_time : datetime
            現在時刻
        current_data : pd.DataFrame
            現在までのデータ
        prediction_hours : float
            予測時間（時間）
        
        Returns:
        --------
        predictions : pd.DataFrame
            予測結果
        """
        # 現在時刻のインデックスを取得
        current_idx = len(current_data) - 1
        
        # 特徴量の準備
        features = self.prepare_features(current_data, current_idx)
        if features is None:
            raise ValueError("特徴量の準備に失敗しました")
        
        # 特徴量ベクトルの作成
        X = np.array([[features[name] for name in self.feature_names]])
        
        # 予測結果の格納
        predictions = []
        
        # 各時間ステップの予測
        max_steps = int(prediction_hours * 6)  # 10分刻みのステップ数
        
        for step in range(1, max_steps + 1):
            # 最も近いモデルを選択
            model_step = min(self.prediction_steps, key=lambda x: abs(x - step))
            
            if model_step in self.models:
                # 正規化
                X_scaled = self.scalers[model_step].transform(X)
                
                # 予測
                y_pred = self.models[model_step].predict(X_scaled)[0]
                
                # 予測結果の保存
                pred_time = current_time + timedelta(minutes=step*10)
                predictions.append({
                    '時刻': pred_time,
                    '予測放流量': y_pred,
                    '予測モデル': f'{model_step*10}分先モデル'
                })
        
        return pd.DataFrame(predictions)
    
    def evaluate_realtime(self, test_data, prediction_hours=2):
        """
        リアルタイム予測の評価
        
        Parameters:
        -----------
        test_data : pd.DataFrame
            テストデータ
        prediction_hours : float
            予測時間
        
        Returns:
        --------
        results : dict
            評価結果
        """
        # フィルタリング
        mask = (test_data['ダム_60分雨量'] > 0) & (test_data['水位_水位'] >= 3.0) & (test_data['ダム_全放流量'] >= 150)
        valid_indices = test_data[mask].index
        
        # 評価用データの準備
        predictions_all = []
        actuals_all = []
        
        print(f"評価対象データ数: {len(valid_indices)}")
        
        # 各時点での予測（最初の100件に制限）
        eval_indices = valid_indices[:min(100, len(valid_indices))]
        
        for i, idx in enumerate(eval_indices):
            if idx + int(prediction_hours * 6) >= len(test_data):
                continue
                
            try:
                # 現在時刻までのデータ（訓練データ＋テストデータの現在時刻まで）
                # テストデータのインデックスを全体データのインデックスに変換
                global_idx = len(test_data) - len(test_data) + idx
                
                # 予測に必要な過去データを含める
                start_idx = max(0, global_idx - 200)  # 過去200ステップ分のデータを使用
                current_data = test_data.iloc[start_idx:idx+1].copy()
                current_time = pd.to_datetime(test_data.iloc[idx]['時刻'])
                
                # 予測
                predictions = self.predict(current_time, current_data, prediction_hours)
                
                # 実測値との比較
                for _, pred in predictions.iterrows():
                    # 予測時刻に対応する実測値のインデックスを計算
                    pred_minutes_ahead = int((pred['時刻'] - current_time).total_seconds() / 60)
                    actual_idx = idx + pred_minutes_ahead // 10
                    
                    if actual_idx < len(test_data):
                        actual_discharge = test_data.iloc[actual_idx]['ダム_全放流量']
                        predictions_all.append(pred['予測放流量'])
                        actuals_all.append(actual_discharge)
                        
                if (i + 1) % 10 == 0:
                    print(f"  評価進捗: {i + 1}/{len(eval_indices)}")
                        
            except Exception as e:
                print(f"  予測エラー at index {idx}: {e}")
                continue
        
        # 評価指標の計算
        predictions_all = np.array(predictions_all)
        actuals_all = np.array(actuals_all)
        
        if len(predictions_all) == 0:
            print("警告: 有効な予測結果がありません")
            return {
                'rmse': np.nan,
                'mae': np.nan,
                'r2': np.nan,
                'peak_rmse': np.nan,
                'n_samples': 0
            }
        
        rmse = np.sqrt(mean_squared_error(actuals_all, predictions_all))
        mae = mean_absolute_error(actuals_all, predictions_all)
        r2 = r2_score(actuals_all, predictions_all)
        
        # ピーク時の評価
        peak_threshold = np.percentile(actuals_all, 90)
        peak_mask = actuals_all >= peak_threshold
        if peak_mask.sum() > 0:
            peak_rmse = np.sqrt(mean_squared_error(actuals_all[peak_mask], predictions_all[peak_mask]))
        else:
            peak_rmse = np.nan
        
        results = {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'peak_rmse': peak_rmse,
            'n_samples': len(predictions_all)
        }
        
        print(f"\nリアルタイム予測評価結果:")
        print(f"  RMSE: {rmse:.2f} m³/s")
        print(f"  MAE: {mae:.2f} m³/s")
        print(f"  R²: {r2:.3f}")
        print(f"  ピーク時RMSE: {peak_rmse:.2f} m³/s")
        
        return results
    
    def save_model(self, filepath):
        """モデルの保存"""
        model_data = {
            'models': self.models,
            'scalers': self.scalers,
            'feature_names': self.feature_names,
            'validation_results': self.validation_results,
            'delay_minutes': self.delay_minutes,
            'flood_prep_level': self.flood_prep_level,
            'flood_max_level': self.flood_max_level
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"モデルを保存しました: {filepath}")
    
    def load_model(self, filepath):
        """モデルの読み込み"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
            
        self.models = model_data['models']
        self.scalers = model_data['scalers']
        self.feature_names = model_data['feature_names']
        self.validation_results = model_data['validation_results']
        self.delay_minutes = model_data['delay_minutes']
        self.flood_prep_level = model_data['flood_prep_level']
        self.flood_max_level = model_data['flood_max_level']
        
        print(f"モデルを読み込みました: {filepath}")
        
    def plot_feature_importance(self, step=12):
        """
        特徴量重要度の可視化
        
        Parameters:
        -----------
        step : int
            表示するモデルのステップ（デフォルト: 2時間先）
        """
        if step not in self.validation_results:
            print(f"ステップ {step} のモデルが存在しません")
            return
            
        importance = self.validation_results[step]['feature_importance']
        
        # ソートして表示
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        features, importances = zip(*sorted_features)
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(features)), importances)
        plt.yticks(range(len(features)), features)
        plt.xlabel('重要度')
        plt.title(f'特徴量重要度（{step*10}分先予測モデル）')
        plt.tight_layout()
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plt.savefig(f'discharge_feature_importance_{timestamp}.png', dpi=150)
        plt.close()
        
        print(f"特徴量重要度を保存しました: discharge_feature_importance_{timestamp}.png")


def main():
    """メイン処理"""
    print("放流量予測モデル")
    print("=" * 60)
    
    # データ読み込み
    data_file = '統合データ_水位ダム_20250730_205325.csv'
    df = pd.read_csv(data_file, encoding='utf-8')
    print(f"データ読み込み完了: {len(df)}行")
    
    # モデルの初期化と訓練
    model = DischargePredictionModel()
    
    # データを訓練用とテスト用に分割（80:20）
    split_idx = int(len(df) * 0.8)
    train_data = df.iloc[:split_idx]
    test_data = df.iloc[split_idx:]
    
    print(f"\n訓練データ: {len(train_data)}行")
    print(f"テストデータ: {len(test_data)}行")
    
    # モデル訓練
    model.train(train_data)
    
    # リアルタイム予測の評価
    print("\n=== リアルタイム予測評価 ===")
    results = model.evaluate_realtime(test_data, prediction_hours=2)
    
    # 特徴量重要度の可視化
    model.plot_feature_importance(step=12)  # 2時間先モデル
    
    # モデルの保存
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_file = f'discharge_prediction_model_{timestamp}.pkl'
    model.save_model(model_file)
    
    print("\n処理完了！")


if __name__ == "__main__":
    main()