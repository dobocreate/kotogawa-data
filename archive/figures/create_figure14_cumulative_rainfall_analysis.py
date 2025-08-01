#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 14: 累積雨量と放流量の関係分析
累積雨量を用いた放流量予測のための包括的分析
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from datetime import datetime, timedelta
from scipy import stats, signal
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# 日本語フォント設定
plt.rcParams['font.sans-serif'] = ['Yu Gothic', 'Hiragino Sans', 'Meirio', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class Figure14CumulativeRainfallAnalysis:
    """累積雨量-放流量関係分析クラス"""
    
    def __init__(self, data_file='統合データ_水位ダム_20250730_205325.csv'):
        """初期化"""
        self.data_file = data_file
        self.df = None
        self.df_filtered = None
        self.correlation_results = None
        self.response_patterns = None
        self.prediction_models = None
        self.change_rate_results = None
        self.lag_correlation_results = None
        
    def load_and_preprocess_data(self):
        """データの読み込みと前処理"""
        print("=== データ読み込みと前処理 ===")
        
        # データ読み込み
        self.df = pd.read_csv(self.data_file, encoding='utf-8')
        print(f"読み込み完了: {len(self.df)}行")
        
        # 時刻をdatetimeに変換
        self.df['時刻'] = pd.to_datetime(self.df['時刻'])
        
        # 必要なカラムの確認
        required_cols = ['時刻', '水位_水位', 'ダム_全放流量', 'ダム_累加雨量', 'ダム_60分雨量']
        for col in required_cols:
            if col not in self.df.columns:
                raise ValueError(f"必要なカラム '{col}' が見つかりません")
        
        # カラム名を簡潔に
        self.df = self.df[required_cols].copy()
        self.df.columns = ['時刻', '水位', '放流量', '累積雨量', '降雨強度']
        
        # フィルタリング（水位≥3m、放流量≥150m³/s、累積雨量>0mm）
        print("\nフィルタリング前:")
        print(f"  全データ数: {len(self.df)}")
        
        mask = (self.df['水位'] >= 3.0) & (self.df['放流量'] >= 150.0) & (self.df['累積雨量'] > 0.0)
        self.df_filtered = self.df[mask].copy()
        
        print("\nフィルタリング後:")
        print(f"  データ数: {len(self.df_filtered)} ({len(self.df_filtered)/len(self.df)*100:.1f}%)")
        
        # 欠損値の処理
        print("\n欠損値の確認:")
        print(f"  累積雨量: {self.df_filtered['累積雨量'].isna().sum()}件")
        print(f"  放流量: {self.df_filtered['放流量'].isna().sum()}件")
        
        # 欠損値を前方補完
        self.df_filtered['累積雨量'].fillna(method='ffill', inplace=True)
        self.df_filtered['放流量'].fillna(method='ffill', inplace=True)
        
        # 統計情報
        print("\n基本統計:")
        print(self.df_filtered[['水位', '放流量', '累積雨量', '降雨強度']].describe())
        
        return self.df_filtered
    
    def analyze_cross_correlation(self):
        """累積雨量と放流量の相互相関分析（0-6時間）"""
        print("\n=== 相互相関分析 ===")
        
        # データ準備
        cumulative_rain = self.df_filtered['累積雨量'].values
        discharge = self.df_filtered['放流量'].values
        
        # 相互相関計算（最大遅延: 6時間 = 36データ点）
        max_lag = 36  # 6時間 × 6データ/時間
        correlations = []
        lags = range(0, max_lag + 1)
        
        for lag in lags:
            if lag == 0:
                corr = np.corrcoef(cumulative_rain, discharge)[0, 1]
            else:
                # 累積雨量が先行する場合の相関
                if len(cumulative_rain) > lag:
                    corr = np.corrcoef(cumulative_rain[:-lag], discharge[lag:])[0, 1]
                else:
                    corr = np.nan
            correlations.append(corr)
        
        correlations = np.array(correlations)
        
        # 最大相関とその遅延
        valid_corrs = ~np.isnan(correlations)
        if valid_corrs.any():
            max_corr_idx = np.nanargmax(correlations)
            max_corr_lag = lags[max_corr_idx]
            max_corr_value = correlations[max_corr_idx]
        else:
            max_corr_idx = 0
            max_corr_lag = 0
            max_corr_value = 0
        
        print(f"最大相関係数: {max_corr_value:.3f}")
        print(f"最適遅延時間: {max_corr_lag * 10}分 ({max_corr_lag / 6:.1f}時間)")
        
        # 累積雨量レベル別の相関分析
        cumulative_levels = [(0, 50), (50, 100), (100, 200), (200, 300), (300, 500), (500, 1000)]
        level_results = []
        
        for min_r, max_r in cumulative_levels:
            mask_level = (cumulative_rain >= min_r) & (cumulative_rain < max_r)
            if mask_level.sum() > 20:  # 最低20データポイント
                cumulative_level = cumulative_rain[mask_level]
                discharge_level = discharge[mask_level]
                
                # 各遅延での相関
                best_lag = 0
                best_corr = 0
                
                for lag in range(0, min(31, len(cumulative_level))):  # 最大5時間
                    if lag > 0 and len(cumulative_level) > lag:
                        try:
                            corr = np.corrcoef(cumulative_level[:-lag], discharge_level[lag:])[0, 1]
                            if not np.isnan(corr) and abs(corr) > abs(best_corr):
                                best_corr = corr
                                best_lag = lag
                        except:
                            continue
                
                level_results.append({
                    'cumulative_range': f"{min_r}-{max_r}",
                    'optimal_lag_minutes': best_lag * 10,
                    'correlation': best_corr,
                    'data_points': mask_level.sum()
                })
                print(f"  {min_r}-{max_r} mm: {mask_level.sum()}件, "
                      f"遅延{best_lag * 10}分, 相関{best_corr:.3f}")
        
        self.correlation_results = {
            'overall': {
                'lags': [lag * 10 for lag in lags],
                'correlations': correlations,
                'optimal_lag_minutes': max_corr_lag * 10,
                'max_correlation': max_corr_value
            },
            'by_level': pd.DataFrame(level_results)
        }
        
        return self.correlation_results
    
    def analyze_response_patterns(self):
        """遅延時間別の応答特性分析"""
        print("\n=== 遅延時間別応答特性分析 ===")
        
        # 分析する遅延時間（0, 1, 2, 3, 4, 5, 6時間）
        lag_hours = [0, 1, 2, 3, 4, 5, 6]
        response_data = {}
        
        for lag_h in lag_hours:
            lag_points = lag_h * 6  # 10分間隔データ
            
            if lag_points < len(self.df_filtered) - 1:
                if lag_points == 0:
                    x = self.df_filtered['累積雨量'].values
                    y = self.df_filtered['放流量'].values
                else:
                    x = self.df_filtered['累積雨量'].values[:-lag_points]
                    y = self.df_filtered['放流量'].values[lag_points:]
                
                # 有効なデータのみ使用
                valid_mask = ~(np.isnan(x) | np.isnan(y)) & (x >= 0) & (y >= 0)
                x_valid = x[valid_mask]
                y_valid = y[valid_mask]
                
                if len(x_valid) > 10:
                    # 回帰分析
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x_valid, y_valid)
                    
                    # べき乗則フィッティング（x > 0の場合）
                    positive_mask = x_valid > 0
                    if positive_mask.sum() > 10:
                        try:
                            def power_law(x, a, b):
                                return a * np.power(x, b)
                            
                            popt, _ = curve_fit(power_law, x_valid[positive_mask], 
                                              y_valid[positive_mask], p0=[100, 0.5])
                            power_params = popt
                        except:
                            power_params = None
                    else:
                        power_params = None
                    
                    response_data[lag_h] = {
                        'x': x_valid,
                        'y': y_valid,
                        'linear': {
                            'slope': slope,
                            'intercept': intercept,
                            'r_squared': r_value**2,
                            'p_value': p_value
                        },
                        'power': power_params,
                        'stats': {
                            'mean_x': np.mean(x_valid),
                            'mean_y': np.mean(y_valid),
                            'std_x': np.std(x_valid),
                            'std_y': np.std(y_valid),
                            'n_points': len(x_valid)
                        }
                    }
                    
                    print(f"\n遅延{lag_h}時間:")
                    print(f"  データ数: {len(x_valid)}")
                    print(f"  線形回帰: y = {slope:.2f}x + {intercept:.0f}, R²={r_value**2:.3f}")
                    if power_params is not None:
                        print(f"  べき乗則: y = {power_params[0]:.1f} × x^{power_params[1]:.3f}")
        
        self.response_patterns = response_data
        return self.response_patterns
    
    def build_prediction_models(self):
        """4時間先までの予測モデル構築"""
        print("\n=== 予測モデル構築 ===")
        
        # 各予測時間（1, 2, 3, 4時間先）のモデル
        prediction_hours = [1, 2, 3, 4]
        models = {}
        
        # 最適遅延時間の取得
        optimal_lag = self.correlation_results['overall']['optimal_lag_minutes'] // 10  # データポイント数
        
        for pred_h in prediction_hours:
            print(f"\n{pred_h}時間先予測モデル:")
            
            # データ準備
            pred_points = pred_h * 6  # 予測時間のデータポイント数
            input_data = []
            output_data = []
            
            for i in range(len(self.df_filtered) - pred_points - 6):
                # 入力: 現在の累積雨量、降雨強度、現在の放流量
                cum_rain = self.df_filtered['累積雨量'].iloc[i]
                rain_intensity = self.df_filtered['降雨強度'].iloc[i]
                current_discharge = self.df_filtered['放流量'].iloc[i]
                
                # 累積雨量の変化率（過去1時間）
                if i >= 6:
                    cum_rain_change = self.df_filtered['累積雨量'].iloc[i] - self.df_filtered['累積雨量'].iloc[i-6]
                else:
                    cum_rain_change = 0
                
                # 出力: pred_h時間後の放流量
                future_discharge = self.df_filtered['放流量'].iloc[i + pred_points]
                
                input_data.append([cum_rain, rain_intensity, current_discharge, cum_rain_change])
                output_data.append(future_discharge)
            
            input_data = np.array(input_data)
            output_data = np.array(output_data)
            
            # 訓練・テスト分割
            X_train, X_test, y_train, y_test = train_test_split(
                input_data, output_data, test_size=0.2, random_state=42
            )
            
            # モデル訓練
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            # 予測
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # 評価
            train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            
            # 係数の解釈
            coef_names = ['累積雨量', '降雨強度', '現在放流量', '累積雨量変化']
            
            models[pred_h] = {
                'model': model,
                'coefficients': dict(zip(coef_names, model.coef_)),
                'intercept': model.intercept_,
                'metrics': {
                    'train_rmse': train_rmse,
                    'test_rmse': test_rmse,
                    'train_r2': train_r2,
                    'test_r2': test_r2,
                    'n_train': len(X_train),
                    'n_test': len(X_test)
                },
                'data': {
                    'X_test': X_test,
                    'y_test': y_test,
                    'y_pred': y_pred_test
                }
            }
            
            print(f"  モデル式: Q_future = {model.intercept_:.1f}")
            for name, coef in zip(coef_names, model.coef_):
                print(f"           + {coef:.3f} × {name}")
            print(f"  訓練RMSE: {train_rmse:.1f} m³/s, R²: {train_r2:.3f}")
            print(f"  テストRMSE: {test_rmse:.1f} m³/s, R²: {test_r2:.3f}")
        
        self.prediction_models = models
        return self.prediction_models
    
    def evaluate_predictions(self):
        """予測精度の詳細評価"""
        print("\n=== 予測精度評価 ===")
        
        # 累積雨量別の精度評価
        cumulative_bins = [0, 50, 100, 200, 300, 500, 1000]
        evaluation_results = {}
        
        for pred_h, model_info in self.prediction_models.items():
            print(f"\n{pred_h}時間先予測の詳細評価:")
            
            X_test = model_info['data']['X_test']
            y_test = model_info['data']['y_test']
            y_pred = model_info['data']['y_pred']
            
            # 累積雨量別評価
            cumulative_rain = X_test[:, 0]  # 累積雨量
            
            level_results = []
            for i in range(len(cumulative_bins) - 1):
                mask = (cumulative_rain >= cumulative_bins[i]) & (cumulative_rain < cumulative_bins[i+1])
                if mask.sum() > 5:
                    rmse = np.sqrt(mean_squared_error(y_test[mask], y_pred[mask]))
                    mae = np.mean(np.abs(y_test[mask] - y_pred[mask]))
                    mape = np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100
                    
                    level_results.append({
                        'cumulative_range': f"{cumulative_bins[i]}-{cumulative_bins[i+1]}",
                        'n_samples': mask.sum(),
                        'rmse': rmse,
                        'mae': mae,
                        'mape': mape
                    })
                    
                    print(f"  {cumulative_bins[i]}-{cumulative_bins[i+1]} mm: "
                          f"n={mask.sum()}, RMSE={rmse:.1f}, MAE={mae:.1f}, MAPE={mape:.1f}%")
            
            evaluation_results[pred_h] = pd.DataFrame(level_results)
        
        self.evaluation_results = evaluation_results
        return self.evaluation_results
    
    def analyze_change_rate_correlation(self):
        """変化率ベースの時間遅れ相関分析"""
        print("\n=== 変化率ベースの時間遅れ相関分析 ===")
        
        # データ準備
        cumulative_rain = self.df_filtered['累積雨量'].values
        discharge = self.df_filtered['放流量'].values
        
        # 変化率の計算（60分間隔 = 6データポイント）
        interval = 6
        
        # 各遅延時間での相関分析
        max_lag = 48  # 最大8時間 = 48データポイント
        lag_results = []
        
        for lag in range(0, max_lag + 1):
            correlations_inc = []  # 増加期
            correlations_dec = []  # 減少期
            correlations_all = []  # 全期間
            
            # 変化率計算
            for i in range(interval, len(cumulative_rain) - interval - lag):
                # 累積雨量の変化率
                dC_dt = (cumulative_rain[i + interval] - cumulative_rain[i - interval]) / (2 * interval * 10 / 60)  # mm/h
                
                # 放流量の変化率（遅延を考慮）
                if i + lag + interval < len(discharge):
                    dQ_dt = (discharge[i + lag + interval] - discharge[i + lag - interval]) / (2 * interval * 10 / 60)  # m³/s/h
                    
                    correlations_all.append((dC_dt, dQ_dt))
                    
                    # 放流量の変化方向で分類
                    if dQ_dt > 0:
                        correlations_inc.append((dC_dt, dQ_dt))
                    else:
                        correlations_dec.append((dC_dt, dQ_dt))
            
            # 相関係数計算
            if len(correlations_all) > 10:
                all_data = np.array(correlations_all)
                r_all = np.corrcoef(all_data[:, 0], all_data[:, 1])[0, 1]
            else:
                r_all = np.nan
                
            if len(correlations_inc) > 10:
                inc_data = np.array(correlations_inc)
                r_inc = np.corrcoef(inc_data[:, 0], inc_data[:, 1])[0, 1]
            else:
                r_inc = np.nan
                
            if len(correlations_dec) > 10:
                dec_data = np.array(correlations_dec)
                r_dec = np.corrcoef(dec_data[:, 0], dec_data[:, 1])[0, 1]
            else:
                r_dec = np.nan
            
            lag_results.append({
                'lag_minutes': lag * 10,
                'correlation_all': r_all,
                'correlation_increase': r_inc,
                'correlation_decrease': r_dec,
                'n_all': len(correlations_all),
                'n_increase': len(correlations_inc),
                'n_decrease': len(correlations_dec)
            })
        
        self.lag_correlation_results = pd.DataFrame(lag_results)
        
        # 最適遅延時間の特定
        max_corr_all = self.lag_correlation_results['correlation_all'].max()
        optimal_lag_all = self.lag_correlation_results.loc[
            self.lag_correlation_results['correlation_all'].idxmax(), 'lag_minutes'
        ]
        
        print(f"\n全期間での最大相関: {max_corr_all:.3f} at {optimal_lag_all}分")
        
        # 変化率ベースの相関分析結果保存
        change_rate_data = []
        optimal_lag_idx = int(optimal_lag_all / 10)
        
        for i in range(interval, len(cumulative_rain) - interval - optimal_lag_idx):
            dC_dt = (cumulative_rain[i + interval] - cumulative_rain[i - interval]) / (2 * interval * 10 / 60)
            if i + optimal_lag_idx + interval < len(discharge):
                dQ_dt = (discharge[i + optimal_lag_idx + interval] - discharge[i + optimal_lag_idx - interval]) / (2 * interval * 10 / 60)
                
                change_rate_data.append({
                    'dC_dt': dC_dt,
                    'dQ_dt': dQ_dt,
                    'direction': 'increase' if dQ_dt > 0 else 'decrease'
                })
        
        self.change_rate_results = pd.DataFrame(change_rate_data)
        
        return self.lag_correlation_results
    
    def create_figure14(self):
        """Figure 14: 累積雨量-放流量関係の包括的分析"""
        print("\n=== Figure 14 作成 ===")
        
        # 3行4列のレイアウト
        fig = plt.figure(figsize=(24, 18))
        
        # 1行目: 1-4時間先予測の結果を表示
        pred_hours = [1, 2, 3, 4]
        
        for idx, pred_h in enumerate(pred_hours):
            ax = plt.subplot(3, 4, idx + 1)
            
            if pred_h in self.prediction_models:
                model_info = self.prediction_models[pred_h]
                y_test = model_info['data']['y_test']
                y_pred = model_info['data']['y_pred']
                
                # 散布図で実測値と予測値を比較
                ax.scatter(y_test, y_pred, alpha=0.3, s=10, c='blue')
                
                # 理想的な予測線（y=x）
                min_val = min(y_test.min(), y_pred.min())
                max_val = max(y_test.max(), y_pred.max())
                ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='理想線(y=x)')
                
                # 回帰線
                z = np.polyfit(y_test, y_pred, 1)
                p = np.poly1d(z)
                ax.plot([min_val, max_val], p([min_val, max_val]), 'g-', linewidth=2, 
                       label=f'回帰線: y={z[0]:.3f}x+{z[1]:.1f}')
                
                # メトリクスを表示
                metrics = model_info['metrics']
                textstr = f'RMSE: {metrics["test_rmse"]:.1f} m³/s\n' + \
                         f'R²: {metrics["test_r2"]:.3f}\n' + \
                         f'n: {metrics["n_test"]}'
                
                # テキストボックスを配置
                props = dict(boxstyle='round', facecolor='white', alpha=0.8)
                ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                       verticalalignment='top', bbox=props)
                
                # 係数情報を下部に表示
                coef_text = f'係数:\n'
                for name, coef in model_info['coefficients'].items():
                    coef_text += f'{name}: {coef:.3f}\n'
                coef_text += f'切片: {model_info["intercept"]:.1f}'
                
                ax.text(0.05, 0.25, coef_text, transform=ax.transAxes, fontsize=8,
                       verticalalignment='top', bbox=props)
                
                ax.set_xlabel('実測値 (m³/s)')
                ax.set_ylabel('予測値 (m³/s)')
                ax.set_title(f'{pred_h}時間先予測', fontsize=14)
                ax.legend(loc='lower right')
                ax.grid(True, alpha=0.3)
                
                # 軸の範囲を揃える
                ax.set_xlim(min_val - 50, max_val + 50)
                ax.set_ylim(min_val - 50, max_val + 50)
                ax.set_aspect('equal', adjustable='box')
        
        # 2行目のグラフ
        # グラフ5: 累積雨量別の遅延時間
        ax5 = plt.subplot(3, 4, 5)
        if hasattr(self, 'correlation_results') and 'by_level' in self.correlation_results:
            level_data = self.correlation_results['by_level']
            
            # 累積雨量の中央値を計算
            cumulative_centers = []
            for range_str in level_data['cumulative_range']:
                min_val, max_val = map(float, range_str.split('-'))
                cumulative_centers.append((min_val + max_val) / 2)
            
            ax5.plot(cumulative_centers, level_data['optimal_lag_minutes'], 'bo-', linewidth=2, markersize=8)
            ax5.set_xlabel('累積雨量 (mm)')
            ax5.set_ylabel('最適遅延時間 (分)')
            ax5.set_title('累積雨量別の最適遅延時間')
            ax5.grid(True, alpha=0.3)
            ax5.set_xlim(0, max(cumulative_centers) + 50)
            
            # データポイント数を表示
            for i, (x, y, n) in enumerate(zip(cumulative_centers, level_data['optimal_lag_minutes'], level_data['data_points'])):
                ax5.text(x, y + 5, f'n={n}', ha='center', va='bottom', fontsize=8)
        
        # グラフ6: 累積雨量別の予測精度（RMSE）
        ax6 = plt.subplot(3, 4, 6)
        if hasattr(self, 'evaluation_results'):
            # 1時間先予測の結果を使用
            if 1 in self.evaluation_results:
                eval_1h = self.evaluation_results[1]
                
                # 累積雨量の中央値を計算
                cumulative_centers = []
                for range_str in eval_1h['cumulative_range']:
                    min_val, max_val = map(float, range_str.split('-'))
                    cumulative_centers.append((min_val + max_val) / 2)
                
                # RMSEとMAPEを両軸で表示
                ax6_rmse = ax6
                ax6_mape = ax6.twinx()
                
                # RMSE
                p1 = ax6_rmse.plot(cumulative_centers, eval_1h['rmse'], 'ro-', linewidth=2, markersize=8, label='RMSE')
                ax6_rmse.set_xlabel('累積雨量 (mm)')
                ax6_rmse.set_ylabel('RMSE (m³/s)', color='r')
                ax6_rmse.tick_params(axis='y', labelcolor='r')
                
                # MAPE
                p2 = ax6_mape.plot(cumulative_centers, eval_1h['mape'], 'bs-', linewidth=2, markersize=8, label='MAPE')
                ax6_mape.set_ylabel('MAPE (%)', color='b')
                ax6_mape.tick_params(axis='y', labelcolor='b')
                
                ax6.set_title('累積雨量別の予測精度（1時間先）')
                ax6.grid(True, alpha=0.3)
                ax6.set_xlim(0, max(cumulative_centers) + 50)
                
                # 凡例
                lines = p1 + p2
                labels = [l.get_label() for l in lines]
                ax6.legend(lines, labels, loc='upper left')
        
        # グラフ7: 累積雨量別のデータ分布
        ax7 = plt.subplot(3, 4, 7)
        
        # 累積雨量を50mmごとに区切ってデータ数をカウント
        cumulative_data = self.df_filtered['累積雨量'].values
        bins = np.arange(0, max(cumulative_data) + 50, 50)
        counts, _ = np.histogram(cumulative_data, bins=bins)
        
        # ビンの中央値を計算
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        # 棒グラフで表示
        bars = ax7.bar(bin_centers, counts, width=40, alpha=0.7, color='steelblue', edgecolor='black')
        
        # 各棒の上にデータ数を表示
        for bar, count in zip(bars, counts):
            if count > 0:
                ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                        f'{count}', ha='center', va='bottom', fontsize=9)
        
        ax7.set_xlabel('累積雨量 (mm)')
        ax7.set_ylabel('データ数')
        ax7.set_title('累積雨量別のデータ分布')
        ax7.grid(True, alpha=0.3)
        ax7.set_xlim(-25, max(bin_centers) + 50)
        
        # 統計情報を追加
        total_data = len(cumulative_data)
        mean_cumulative = np.mean(cumulative_data)
        median_cumulative = np.median(cumulative_data)
        
        stats_text = f'総データ数: {total_data:,}\n平均: {mean_cumulative:.1f} mm\n中央値: {median_cumulative:.1f} mm'
        props = dict(boxstyle='round', facecolor='white', alpha=0.8)
        ax7.text(0.65, 0.95, stats_text, transform=ax7.transAxes, fontsize=9,
                verticalalignment='top', bbox=props)
        
        # グラフ8: 予測精度の時間変化（1-4時間）
        ax8 = plt.subplot(3, 4, 8)
        if hasattr(self, 'prediction_models'):
            pred_hours_list = []
            rmse_list = []
            r2_list = []
            
            for h in [1, 2, 3, 4]:
                if h in self.prediction_models:
                    pred_hours_list.append(h)
                    rmse_list.append(self.prediction_models[h]['metrics']['test_rmse'])
                    r2_list.append(self.prediction_models[h]['metrics']['test_r2'])
            
            # RMSE
            ax8_rmse = ax8
            ax8_r2 = ax8.twinx()
            
            p1 = ax8_rmse.plot(pred_hours_list, rmse_list, 'ro-', linewidth=2, markersize=10, label='RMSE')
            ax8_rmse.set_xlabel('予測時間 (時間先)')
            ax8_rmse.set_ylabel('RMSE (m³/s)', color='r')
            ax8_rmse.tick_params(axis='y', labelcolor='r')
            
            # R²
            p2 = ax8_r2.plot(pred_hours_list, r2_list, 'bs-', linewidth=2, markersize=10, label='R²')
            ax8_r2.set_ylabel('R²', color='b')
            ax8_r2.tick_params(axis='y', labelcolor='b')
            ax8_r2.set_ylim(0, 1)
            
            ax8.set_title('予測精度の時間変化')
            ax8.grid(True, alpha=0.3)
            ax8.set_xticks(pred_hours_list)
            ax8.set_xlim(0.5, 4.5)
            
            # 凡例
            lines = p1 + p2
            labels = [l.get_label() for l in lines]
            ax8.legend(lines, labels, loc='center right')
        
        # 3行目: 変化率ベースの分析
        # 変化率相関分析が未実行の場合は実行
        if not hasattr(self, 'lag_correlation_results'):
            self.analyze_change_rate_correlation()
        
        # グラフ9: 時間遅れ相関分析（変化率ベース）
        ax9 = plt.subplot(3, 4, 9)
        if hasattr(self, 'lag_correlation_results'):
            lag_data = self.lag_correlation_results
            
            # 全期間、増加期、減少期の相関
            ax9.plot(lag_data['lag_minutes'], lag_data['correlation_all'], 
                    'k-', linewidth=2, label='全期間')
            ax9.plot(lag_data['lag_minutes'], lag_data['correlation_increase'], 
                    'r--', linewidth=2, label='放流量増加期')
            ax9.plot(lag_data['lag_minutes'], lag_data['correlation_decrease'], 
                    'b--', linewidth=2, label='放流量減少期')
            
            # 最大相関点をマーク（全期間）
            max_idx = lag_data['correlation_all'].idxmax()
            max_lag = lag_data.loc[max_idx, 'lag_minutes']
            max_corr = lag_data.loc[max_idx, 'correlation_all']
            ax9.plot(max_lag, max_corr, 'ko', markersize=10)
            ax9.text(max_lag + 10, max_corr, f'{max_lag}分\nr={max_corr:.3f}', 
                    fontsize=10, ha='left')
            
            # 増加期の最大相関点
            if not lag_data['correlation_increase'].isna().all():
                max_idx_inc = lag_data['correlation_increase'].idxmax()
                max_lag_inc = lag_data.loc[max_idx_inc, 'lag_minutes']
                max_corr_inc = lag_data.loc[max_idx_inc, 'correlation_increase']
                ax9.plot(max_lag_inc, max_corr_inc, 'ro', markersize=8)
                ax9.text(max_lag_inc + 10, max_corr_inc - 0.05, f'{max_lag_inc}分', 
                        fontsize=9, ha='left', color='red')
            
            # 減少期の最大相関点
            if not lag_data['correlation_decrease'].isna().all():
                max_idx_dec = lag_data['correlation_decrease'].idxmax()
                max_lag_dec = lag_data.loc[max_idx_dec, 'lag_minutes']
                max_corr_dec = lag_data.loc[max_idx_dec, 'correlation_decrease']
                ax9.plot(max_lag_dec, max_corr_dec, 'bo', markersize=8)
                ax9.text(max_lag_dec + 10, max_corr_dec + 0.05, f'{max_lag_dec}分', 
                        fontsize=9, ha='left', color='blue')
            
            ax9.set_xlabel('遅延時間 (分)')
            ax9.set_ylabel('相関係数')
            ax9.set_title('変化率ベースの時間遅れ相関分析')
            ax9.grid(True, alpha=0.3)
            ax9.legend(loc='best')
            ax9.set_xlim(0, 480)
            ax9.set_ylim(-0.5, 1.0)
        
        # グラフ10: 変化率の散布図（最適遅延時間）
        ax10 = plt.subplot(3, 4, 10)
        if hasattr(self, 'change_rate_results'):
            data = self.change_rate_results
            increase_mask = data['direction'] == 'increase'
            decrease_mask = data['direction'] == 'decrease'
            
            # 増加期
            if increase_mask.sum() > 0:
                ax10.scatter(data.loc[increase_mask, 'dC_dt'],
                           data.loc[increase_mask, 'dQ_dt'],
                           c='red', alpha=0.3, s=10, label=f'増加期 (n={increase_mask.sum()})')
            
            # 減少期
            if decrease_mask.sum() > 0:
                ax10.scatter(data.loc[decrease_mask, 'dC_dt'],
                           data.loc[decrease_mask, 'dQ_dt'],
                           c='blue', alpha=0.3, s=10, label=f'減少期 (n={decrease_mask.sum()})')
            
            # 原点を通る線
            ax10.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
            ax10.axvline(x=0, color='gray', linestyle='-', alpha=0.5)
            
            ax10.set_xlabel('累積雨量変化率 (mm/h)')
            ax10.set_ylabel('放流量変化率 (m³/s/h)')
            ax10.set_title('変化率の関係（最適遅延時間）')
            ax10.legend()
            ax10.grid(True, alpha=0.3)
        
        # グラフ11: 累積雨量別の変化率相関
        ax11 = plt.subplot(3, 4, 11)
        if hasattr(self, 'change_rate_results') and hasattr(self, 'lag_correlation_results'):
            # 最適遅延時間を取得
            optimal_lag_all = self.lag_correlation_results.loc[
                self.lag_correlation_results['correlation_all'].idxmax(), 'lag_minutes'
            ]
            
            # 累積雨量ビンごとの相関分析
            cumulative_bins = [0, 100, 200, 300, 500, 1000]
            bin_correlations = []
            
            for i in range(len(cumulative_bins) - 1):
                min_c, max_c = cumulative_bins[i], cumulative_bins[i+1]
                
                # 該当する累積雨量のデータを抽出
                bin_mask = (self.df_filtered['累積雨量'].iloc[6:-6-int(optimal_lag_all/10)] >= min_c) & \
                          (self.df_filtered['累積雨量'].iloc[6:-6-int(optimal_lag_all/10)] < max_c)
                
                if bin_mask.sum() > 20:
                    bin_data = self.change_rate_results[bin_mask.values]
                    if len(bin_data) > 10:
                        corr = np.corrcoef(bin_data['dC_dt'], bin_data['dQ_dt'])[0, 1]
                        bin_correlations.append({
                            'cumulative_center': (min_c + max_c) / 2,
                            'correlation': corr,
                            'n_points': len(bin_data)
                        })
            
            if bin_correlations:
                bin_df = pd.DataFrame(bin_correlations)
                ax11.bar(bin_df['cumulative_center'], bin_df['correlation'], 
                        width=80, alpha=0.7, color='green', edgecolor='black')
                
                # データ数を表示
                for i, row in bin_df.iterrows():
                    ax11.text(row['cumulative_center'], row['correlation'] + 0.02,
                            f"n={row['n_points']}", ha='center', va='bottom', fontsize=8)
                
                ax11.set_xlabel('累積雨量 (mm)')
                ax11.set_ylabel('変化率の相関係数')
                ax11.set_title('累積雨量別の変化率相関')
                ax11.grid(True, alpha=0.3)
                ax11.set_ylim(-0.5, 1.0)
        
        # グラフ12: 変化率ベースの予測モデル性能
        ax12 = plt.subplot(3, 4, 12)
        if hasattr(self, 'change_rate_results'):
            # 変化率を用いた簡易予測モデルの評価
            data = self.change_rate_results
            
            # 線形回帰
            X = data['dC_dt'].values.reshape(-1, 1)
            y = data['dQ_dt'].values
            
            model = LinearRegression()
            model.fit(X, y)
            y_pred = model.predict(X)
            
            # R²スコア
            r2 = model.score(X, y)
            
            # 予測値 vs 実測値
            ax12.scatter(y, y_pred, alpha=0.3, s=5)
            
            # 理想線
            min_val = min(y.min(), y_pred.min())
            max_val = max(y.max(), y_pred.max())
            ax12.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
            
            ax12.set_xlabel('実測変化率 (m³/s/h)')
            ax12.set_ylabel('予測変化率 (m³/s/h)')
            ax12.set_title(f'変化率モデルの予測性能\nR²={r2:.3f}')
            ax12.grid(True, alpha=0.3)
            ax12.set_aspect('equal', adjustable='box')
        
        # 全体タイトル
        fig.suptitle('Figure 14: 累積雨量を用いた放流量予測の包括的分析', fontsize=16, y=0.995)
        
        plt.tight_layout()
        
        # 保存
        import os
        os.makedirs('figures', exist_ok=True)
        output_path = f"figures/figure14_cumulative_rainfall_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nFigure 14 を保存しました: {output_path}")
        
        plt.close()
        
        return fig
    
    def save_analysis_results(self):
        """分析結果の保存"""
        print("\n=== 分析結果の保存 ===")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. 相関分析結果
        corr_file = f"cumulative_rainfall_discharge_correlation_{timestamp}.csv"
        self.correlation_results['by_level'].to_csv(corr_file, index=False, encoding='utf-8-sig')
        print(f"相関分析結果を保存: {corr_file}")
        
        # 2. 予測モデルの要約
        model_summary = []
        for h, model_info in self.prediction_models.items():
            summary = {
                '予測時間': f'{h}時間先',
                **model_info['coefficients'],
                '切片': model_info['intercept'],
                **model_info['metrics']
            }
            model_summary.append(summary)
        
        model_file = f"cumulative_rainfall_discharge_models_{timestamp}.csv"
        pd.DataFrame(model_summary).to_csv(model_file, index=False, encoding='utf-8-sig')
        print(f"予測モデル要約を保存: {model_file}")
        
        # 3. 評価結果
        for h, eval_df in self.evaluation_results.items():
            eval_file = f"cumulative_rainfall_discharge_evaluation_{h}h_{timestamp}.csv"
            eval_df.to_csv(eval_file, index=False, encoding='utf-8-sig')
            print(f"{h}時間先予測の評価結果を保存: {eval_file}")

def main():
    """メイン処理"""
    print("累積雨量-放流量関係分析プログラム")
    print("=" * 60)
    
    analyzer = Figure14CumulativeRainfallAnalysis()
    
    try:
        # 1. データ読み込みと前処理
        analyzer.load_and_preprocess_data()
        
        # 2. 相互相関分析
        analyzer.analyze_cross_correlation()
        
        # 3. 応答特性分析
        analyzer.analyze_response_patterns()
        
        # 4. 予測モデル構築
        analyzer.build_prediction_models()
        
        # 5. 予測精度評価
        analyzer.evaluate_predictions()
        
        # 6. 変化率ベースの相関分析
        analyzer.analyze_change_rate_correlation()
        
        # 7. Figure 14作成
        analyzer.create_figure14()
        analyzer.save_analysis_results()
        
        print("\n分析完了！")
        
    except Exception as e:
        print(f"\nエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()