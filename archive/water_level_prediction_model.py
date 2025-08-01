#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
水位予測機械学習モデル
遅延時間と放流量から3時間先までの水位を予測
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import pickle
import json
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

# 深層学習ライブラリ
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorchが利用できません。LightGBMを使用します。")

# 機械学習ライブラリ
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("LightGBMが利用できません。")

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class WaterLevelPredictor:
    """水位予測モデルのベースクラス"""
    
    def __init__(self, model_type='lightgbm', config=None):
        """
        Parameters:
        -----------
        model_type : str
            'lightgbm' or 'lstm'
        config : dict
            モデル設定
        """
        self.model_type = model_type
        self.config = config or self._get_default_config()
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        self.is_fitted = False
        
        # 遅延時間計算用のパラメータ（Figure 8の全体トレンドライン）
        self.delay_params = {
            'a': 0.00008,  # 2次項の係数
            'b': 0.061,    # 1次項の係数
            'c': 55.5      # 定数項
        }
        
        # 学習率スケジューラーのパラメータ
        self.initial_learning_rate = self.config.get('initial_learning_rate', 0.01)
        self.learning_rate_decay = self.config.get('learning_rate_decay', 0.995)
        self.update_count = 0
        
    def _get_default_config(self):
        """デフォルト設定を返す"""
        if self.model_type == 'lightgbm':
            return {
                'num_leaves': 31,
                'max_depth': -1,
                'learning_rate': 0.01,
                'n_estimators': 1000,
                'objective': 'regression',
                'metric': 'rmse',
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'early_stopping_rounds': 50,
                'initial_learning_rate': 0.01,
                'learning_rate_decay': 0.995
            }
        else:
            return {
                'hidden_size': 128,
                'num_layers': 2,
                'dropout': 0.2,
                'learning_rate': 0.001,
                'batch_size': 32,
                'epochs': 100,
                'patience': 20,
                'initial_learning_rate': 0.001,
                'learning_rate_decay': 0.995
            }
    
    def calculate_delay_time(self, discharge):
        """
        放流量から遅延時間を計算（Figure 8の関係式）
        
        Parameters:
        -----------
        discharge : float or array
            放流量 (m³/s)
        
        Returns:
        --------
        delay_time : float or array
            遅延時間 (分)
        """
        return (self.delay_params['a'] * discharge**2 + 
                self.delay_params['b'] * discharge + 
                self.delay_params['c'])
    
    def create_features(self, df, target_idx, already_filtered=False):
        """
        特徴量を作成
        
        Parameters:
        -----------
        df : DataFrame
            時系列データ
        target_idx : int
            予測対象の時刻インデックス
        already_filtered : bool
            データがすでにフィルタリング済みかどうか
        
        Returns:
        --------
        features : dict
            特徴量の辞書
        """
        features = {}
        
        # 1. 現在の放流量
        current_discharge = df.iloc[target_idx]['ダム_全放流量']
        features['current_discharge'] = current_discharge
        
        # 2. 遅延時間（Figure 8の式から計算）
        features['delay_time'] = self.calculate_delay_time(current_discharge)
        
        # 3. 増加期/減少期フラグ（1時間前との差で判定）
        if target_idx >= 6:
            discharge_1h_ago = df.iloc[target_idx - 6]['ダム_全放流量']
            discharge_change = current_discharge - discharge_1h_ago
            features['flow_direction'] = 'increase' if discharge_change > 0 else 'decrease'
            features['discharge_change_1h'] = discharge_change
        else:
            features['flow_direction'] = 'stable'
            features['discharge_change_1h'] = 0
        
        # 4. 過去3時間の放流量（18ポイント）
        # フィルタリング条件を満たすデータのみを使用
        for i in range(18):
            if target_idx - i >= 0:
                discharge_val = df.iloc[target_idx - i]['ダム_全放流量']
                water_val = df.iloc[target_idx - i]['水位_水位']
                # すでにフィルタリング済みの場合はそのまま使用
                if already_filtered:
                    features[f'discharge_lag_{i*10}min'] = discharge_val
                else:
                    # フィルタリング条件チェック
                    if discharge_val >= 150 and water_val >= 3.0:
                        features[f'discharge_lag_{i*10}min'] = discharge_val
                    else:
                        features[f'discharge_lag_{i*10}min'] = np.nan
            else:
                features[f'discharge_lag_{i*10}min'] = np.nan
        
        # 5. 過去3時間の水位（18ポイント）
        # フィルタリング条件を満たすデータのみを使用
        for i in range(18):
            if target_idx - i >= 0:
                discharge_val = df.iloc[target_idx - i]['ダム_全放流量']
                water_val = df.iloc[target_idx - i]['水位_水位']
                # すでにフィルタリング済みの場合はそのまま使用
                if already_filtered:
                    features[f'water_level_lag_{i*10}min'] = water_val
                else:
                    # フィルタリング条件チェック
                    if discharge_val >= 150 and water_val >= 3.0:
                        features[f'water_level_lag_{i*10}min'] = water_val
                    else:
                        features[f'water_level_lag_{i*10}min'] = np.nan
            else:
                features[f'water_level_lag_{i*10}min'] = np.nan
        
        # 6. 変化規模カテゴリ
        abs_change = abs(features['discharge_change_1h'])
        if abs_change < 50:
            features['change_magnitude'] = 'small'
        elif abs_change < 200:
            features['change_magnitude'] = 'medium'
        else:
            features['change_magnitude'] = 'large'
        
        # 7. 初期水位レベル
        current_water_level = df.iloc[target_idx]['水位_水位']
        if current_water_level < 3.5:
            features['water_level_category'] = 'low'
        elif current_water_level < 4.5:
            features['water_level_category'] = 'medium'
        else:
            features['water_level_category'] = 'high'
        
        # 8. 統計的特徴量
        if target_idx >= 18:
            # フィルタリング条件を満たすデータのみを抽出
            recent_discharge = []
            recent_water = []
            
            for i in range(18):
                if target_idx - i >= 0:
                    discharge_val = df.iloc[target_idx - i]['ダム_全放流量']
                    water_val = df.iloc[target_idx - i]['水位_水位']
                    # すでにフィルタリング済みの場合はそのまま使用
                    if already_filtered:
                        recent_discharge.append(discharge_val)
                        recent_water.append(water_val)
                    else:
                        # フィルタリング条件チェック
                        if discharge_val >= 150 and water_val >= 3.0:
                            recent_discharge.append(discharge_val)
                            recent_water.append(water_val)
            
            # 有効なデータが3点以上ある場合のみ統計量を計算
            if len(recent_discharge) >= 3:
                features['discharge_mean_3h'] = np.mean(recent_discharge)
                features['discharge_std_3h'] = np.std(recent_discharge)
                features['discharge_trend_3h'] = recent_discharge[0] - recent_discharge[-1] if len(recent_discharge) > 1 else 0
                
                features['water_level_mean_3h'] = np.mean(recent_water)
                features['water_level_std_3h'] = np.std(recent_water)
                features['water_level_trend_3h'] = recent_water[0] - recent_water[-1] if len(recent_water) > 1 else 0
            else:
                # データが不足している場合は現在値を使用
                features['discharge_mean_3h'] = current_discharge
                features['discharge_std_3h'] = 0
                features['discharge_trend_3h'] = 0
                features['water_level_mean_3h'] = current_water_level
                features['water_level_std_3h'] = 0
                features['water_level_trend_3h'] = 0
        else:
            features['discharge_mean_3h'] = current_discharge
            features['discharge_std_3h'] = 0
            features['discharge_trend_3h'] = 0
            features['water_level_mean_3h'] = current_water_level
            features['water_level_std_3h'] = 0
            features['water_level_trend_3h'] = 0
        
        return features
    
    def prepare_training_data(self, df, lookback_hours=3, forecast_hours=3):
        """
        訓練データの準備
        
        Parameters:
        -----------
        df : DataFrame
            時系列データ
        lookback_hours : int
            過去データの参照時間
        forecast_hours : int
            予測時間
        
        Returns:
        --------
        X : array
            特徴量
        y : array
            目標値（3時間先までの水位）
        """
        # フィルタリング: 放流量≥150m³/s かつ 水位≥3.0m
        mask = (df['ダム_全放流量'] >= 150) & (df['水位_水位'] >= 3.0)
        df_filtered = df[mask].reset_index(drop=True)
        
        if len(df_filtered) < 36:  # 最低6時間分のデータが必要
            raise ValueError("フィルタリング後のデータが不十分です")
        
        X_list = []
        y_list = []
        
        lookback_steps = lookback_hours * 6  # 10分単位
        forecast_steps = forecast_hours * 6
        
        # 訓練データの作成
        for i in range(lookback_steps, len(df_filtered) - forecast_steps):
            # 特徴量の作成
            features = self.create_features(df_filtered, i)
            
            # 目標値：3時間先までの水位（18ポイント）
            target = []
            for j in range(1, forecast_steps + 1):
                if i + j < len(df_filtered):
                    target.append(df_filtered.iloc[i + j]['水位_水位'])
                else:
                    target.append(np.nan)
            
            # NaNを含むデータは除外
            if not any(pd.isna(list(features.values()))) and not any(pd.isna(target)):
                X_list.append(features)
                y_list.append(target)
        
        # DataFrameに変換
        X_df = pd.DataFrame(X_list)
        
        # カテゴリカル変数のエンコーディング
        categorical_cols = ['flow_direction', 'change_magnitude', 'water_level_category']
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                X_df[col] = self.label_encoders[col].fit_transform(X_df[col])
            else:
                X_df[col] = self.label_encoders[col].transform(X_df[col])
        
        # 特徴量名を保存
        self.feature_names = X_df.columns.tolist()
        
        # NumPy配列に変換
        X = X_df.values
        y = np.array(y_list)
        
        return X, y
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        モデルの学習
        
        Parameters:
        -----------
        X_train : array
            訓練用特徴量
        y_train : array
            訓練用目標値
        X_val : array, optional
            検証用特徴量
        y_val : array, optional
            検証用目標値
        """
        # データの正規化
        X_train_scaled = self.scaler_X.fit_transform(X_train)
        y_train_scaled = self.scaler_y.fit_transform(y_train)
        
        if X_val is not None:
            X_val_scaled = self.scaler_X.transform(X_val)
            y_val_scaled = self.scaler_y.transform(y_val)
        
        if self.model_type == 'lightgbm' and LIGHTGBM_AVAILABLE:
            self._fit_lightgbm(X_train_scaled, y_train_scaled, 
                             X_val_scaled if X_val is not None else None,
                             y_val_scaled if y_val is not None else None)
        elif self.model_type == 'lstm' and TORCH_AVAILABLE:
            self._fit_lstm(X_train_scaled, y_train_scaled,
                          X_val_scaled if X_val is not None else None,
                          y_val_scaled if y_val is not None else None)
        else:
            raise ValueError(f"モデルタイプ {self.model_type} は利用できません")
        
        self.is_fitted = True
    
    def _fit_lightgbm(self, X_train, y_train, X_val=None, y_val=None):
        """LightGBMモデルの学習"""
        # マルチアウトプット対応：各時刻を個別に学習
        self.model = []
        
        for i in range(y_train.shape[1]):
            print(f"時刻 {(i+1)*10}分先のモデルを学習中...")
            
            train_data = lgb.Dataset(X_train, y_train[:, i])
            valid_data = lgb.Dataset(X_val, y_val[:, i]) if X_val is not None else None
            
            params = self.config.copy()
            params['learning_rate'] = self.get_current_learning_rate()
            
            model = lgb.train(
                params,
                train_data,
                valid_sets=[valid_data] if valid_data else None,
                callbacks=[lgb.early_stopping(self.config['early_stopping_rounds'])] if valid_data else None
            )
            
            self.model.append(model)
    
    def predict(self, X):
        """
        予測
        
        Parameters:
        -----------
        X : array
            特徴量
        
        Returns:
        --------
        predictions : array
            予測値（3時間先までの水位）
        """
        if not self.is_fitted:
            raise ValueError("モデルが学習されていません")
        
        X_scaled = self.scaler_X.transform(X)
        
        if self.model_type == 'lightgbm':
            predictions_scaled = np.column_stack([
                model.predict(X_scaled) for model in self.model
            ])
        else:
            # LSTM実装は省略
            raise NotImplementedError("LSTM予測は未実装です")
        
        predictions = self.scaler_y.inverse_transform(predictions_scaled)
        return predictions
    
    def online_update(self, X_new, y_new):
        """
        オンライン学習による更新（1日1回）
        
        Parameters:
        -----------
        X_new : array
            新しい特徴量
        y_new : array
            新しい目標値
        """
        self.update_count += 1
        
        # 学習率の更新
        current_lr = self.get_current_learning_rate()
        print(f"オンライン更新 #{self.update_count}, 学習率: {current_lr:.6f}")
        
        # データの正規化
        X_new_scaled = self.scaler_X.transform(X_new)
        y_new_scaled = self.scaler_y.transform(y_new)
        
        if self.model_type == 'lightgbm':
            # LightGBMの場合：追加学習
            for i, model in enumerate(self.model):
                # 既存モデルを初期値として新しいデータで追加学習
                train_data = lgb.Dataset(X_new_scaled, y_new_scaled[:, i])
                
                # 新しいパラメータで追加学習
                params = self.config.copy()
                params['learning_rate'] = current_lr
                params['n_estimators'] = 50  # 少ないイテレーションで更新
                
                # 追加学習
                self.model[i] = lgb.train(
                    params,
                    train_data,
                    init_model=model,
                    keep_training_booster=True
                )
    
    def get_current_learning_rate(self):
        """現在の学習率を取得"""
        return self.initial_learning_rate * (self.learning_rate_decay ** self.update_count)
    
    def time_series_cross_validation(self, X, y, n_splits=5):
        """
        時系列クロスバリデーション
        
        Parameters:
        -----------
        X : array
            特徴量
        y : array
            目標値
        n_splits : int
            分割数
        
        Returns:
        --------
        cv_results : dict
            クロスバリデーション結果
        """
        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_results = {
            'rmse': [],
            'mae': [],
            'r2': []
        }
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            print(f"\nFold {fold + 1}/{n_splits}")
            
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # モデルの初期化と学習
            self.__init__(self.model_type, self.config)
            self.fit(X_train, y_train, X_val, y_val)
            
            # 予測
            y_pred = self.predict(X_val)
            
            # 評価指標の計算
            rmse = np.sqrt(mean_squared_error(y_val.flatten(), y_pred.flatten()))
            mae = mean_absolute_error(y_val.flatten(), y_pred.flatten())
            r2 = r2_score(y_val.flatten(), y_pred.flatten())
            
            cv_results['rmse'].append(rmse)
            cv_results['mae'].append(mae)
            cv_results['r2'].append(r2)
            
            print(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
        
        # 平均と標準偏差
        print(f"\n=== クロスバリデーション結果 ===")
        print(f"RMSE: {np.mean(cv_results['rmse']):.4f} ± {np.std(cv_results['rmse']):.4f}")
        print(f"MAE: {np.mean(cv_results['mae']):.4f} ± {np.std(cv_results['mae']):.4f}")
        print(f"R²: {np.mean(cv_results['r2']):.4f} ± {np.std(cv_results['r2']):.4f}")
        
        return cv_results
    
    def save_model(self, filepath):
        """モデルの保存"""
        model_data = {
            'model_type': self.model_type,
            'config': self.config,
            'model': self.model,
            'scaler_X': self.scaler_X,
            'scaler_y': self.scaler_y,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted,
            'delay_params': self.delay_params,
            'update_count': self.update_count,
            'initial_learning_rate': self.initial_learning_rate,
            'learning_rate_decay': self.learning_rate_decay
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"モデルを保存しました: {filepath}")
    
    def load_model(self, filepath):
        """モデルの読み込み"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model_type = model_data['model_type']
        self.config = model_data['config']
        self.model = model_data['model']
        self.scaler_X = model_data['scaler_X']
        self.scaler_y = model_data['scaler_y']
        self.label_encoders = model_data['label_encoders']
        self.feature_names = model_data['feature_names']
        self.is_fitted = model_data['is_fitted']
        self.delay_params = model_data['delay_params']
        self.update_count = model_data.get('update_count', 0)
        self.initial_learning_rate = model_data.get('initial_learning_rate', 0.01)
        self.learning_rate_decay = model_data.get('learning_rate_decay', 0.995)
        
        # feature_namesが空の場合、期待される特徴量名を生成
        if not self.feature_names:
            print("警告: feature_namesが空です。期待される特徴量名を生成します。")
            self.feature_names = self._generate_expected_feature_names()
        
        # label_encodersが空の場合、期待されるエンコーダーを生成
        if not self.label_encoders:
            print("警告: label_encodersが空です。期待されるエンコーダーを生成します。")
            self.label_encoders = self._generate_expected_label_encoders()
        
        print(f"モデルを読み込みました: {filepath}")
        print(f"特徴量数: {len(self.feature_names)}")
    
    def _generate_expected_feature_names(self):
        """期待される特徴量名を生成"""
        feature_names = []
        
        # 基本特徴量
        feature_names.append('current_discharge')
        feature_names.append('delay_time')
        feature_names.append('discharge_change_1h')
        
        # 過去3時間の放流量（18ポイント）
        for i in range(18):
            feature_names.append(f'discharge_lag_{i*10}min')
        
        # 過去3時間の水位（18ポイント）
        for i in range(18):
            feature_names.append(f'water_level_lag_{i*10}min')
        
        # 統計的特徴量
        feature_names.extend([
            'discharge_mean_3h',
            'discharge_std_3h',
            'discharge_trend_3h',
            'water_level_mean_3h',
            'water_level_std_3h',
            'water_level_trend_3h'
        ])
        
        # カテゴリカル特徴量（エンコード後は数値）
        feature_names.extend([
            'flow_direction',
            'change_magnitude',
            'water_level_category'
        ])
        
        return feature_names
    
    def _generate_expected_label_encoders(self):
        """期待されるラベルエンコーダーを生成"""
        from sklearn.preprocessing import LabelEncoder
        
        label_encoders = {}
        
        # flow_direction エンコーダー
        le_flow = LabelEncoder()
        le_flow.fit(['decrease', 'increase', 'stable'])
        label_encoders['flow_direction'] = le_flow
        
        # change_magnitude エンコーダー
        le_mag = LabelEncoder()
        le_mag.fit(['large', 'medium', 'small'])
        label_encoders['change_magnitude'] = le_mag
        
        # water_level_category エンコーダー
        le_water = LabelEncoder()
        le_water.fit(['high', 'low', 'medium'])
        label_encoders['water_level_category'] = le_water
        
        return label_encoders


def main():
    """使用例"""
    print("=== 水位予測モデルのデモ ===")
    
    # データの読み込み（仮想データ）
    print("\n1. データ準備")
    # 実際の使用では、CSVファイルから読み込む
    # df = pd.read_csv('統合データ_水位ダム_20250730_205325.csv', encoding='utf-8')
    
    # モデルの初期化
    print("\n2. モデル初期化")
    predictor = WaterLevelPredictor(model_type='lightgbm')
    
    # 特徴量の作成例
    print("\n3. 特徴量作成の例")
    # X, y = predictor.prepare_training_data(df)
    
    # クロスバリデーション
    # print("\n4. 時系列クロスバリデーション")
    # cv_results = predictor.time_series_cross_validation(X, y)
    
    # モデルの保存
    # print("\n5. モデル保存")
    # predictor.save_model('water_level_predictor.pkl')
    
    print("\n実装完了！")


if __name__ == "__main__":
    main()