#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
河川水位遅延時間分析スクリプト
放流量と水位の関係から、変化点検出と水位推定のための遅延時間を分析
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, stats
from scipy.optimize import curve_fit
import tkinter as tk
from tkinter import filedialog
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 日本語フォント設定
plt.rcParams['font.sans-serif'] = ['Yu Gothic', 'Hiragino Sans', 'Meirio', 'DejaVu Sans']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.unicode_minus'] = False

class RiverDelayAnalysis:
    def __init__(self):
        self.df = None
        self.df_processed = None
        self.change_points = None
        self.delay_times = None
        self.correlation_results = None
        self.model_params = None
        
    def load_data(self):
        """データの読み込み"""
        print("=== データ読み込み ===")
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(
            title="統合データCSVファイルを選択",
            filetypes=[("CSV files", "*.csv")]
        )
        if not file_path:
            raise ValueError("ファイルが選択されませんでした")
        
        self.df = pd.read_csv(file_path, encoding='utf-8')
        print(f"読み込み完了: {len(self.df)}行")
        print(f"カラム: {self.df.columns.tolist()}")
        
        # 時刻をdatetimeに変換
        self.df['時刻'] = pd.to_datetime(self.df['時刻'])
        
        # 必要なカラムの確認
        required_cols = ['時刻', '水位_水位', 'ダム_全放流量']
        for col in required_cols:
            if col not in self.df.columns:
                raise ValueError(f"必要なカラム '{col}' が見つかりません")
        
        return self.df
    
    def preprocess_data(self):
        """データの前処理"""
        print("\n=== データ前処理 ===")
        self.df_processed = self.df.copy()
        
        # 1. 欠測値の処理（前方補完）
        print("欠測値処理前:")
        print(f"  水位欠測数: {self.df_processed['水位_水位'].isna().sum()}")
        print(f"  放流量欠測数: {self.df_processed['ダム_全放流量'].isna().sum()}")
        
        self.df_processed['水位_水位'].fillna(method='ffill', inplace=True)
        self.df_processed['ダム_全放流量'].fillna(method='ffill', inplace=True)
        
        # 2. 水位変化の計算
        self.df_processed['水位_変化'] = self.df_processed['水位_水位'].diff()
        
        # 3. 異常値の除去（10分間で1m以上の水位変化）
        abnormal_mask = abs(self.df_processed['水位_変化']) > 1.0
        abnormal_count = abnormal_mask.sum()
        print(f"\n異常値検出: {abnormal_count}件（10分間で1m以上の水位変化）")
        
        if abnormal_count > 0:
            print("異常値の例:")
            print(self.df_processed[abnormal_mask][['時刻', '水位_水位', '水位_変化']].head())
            # 異常値を前の値で置換
            self.df_processed.loc[abnormal_mask, '水位_水位'] = np.nan
            self.df_processed['水位_水位'].fillna(method='ffill', inplace=True)
            self.df_processed['水位_変化'] = self.df_processed['水位_水位'].diff()
        
        print(f"\n前処理後のデータ数: {len(self.df_processed)}")
        
        # 4. 移動平均の計算
        self.df_processed['放流量_MA30'] = self.df_processed['ダム_全放流量'].rolling(3, center=True).mean()
        self.df_processed['放流量_MA60'] = self.df_processed['ダム_全放流量'].rolling(6, center=True).mean()
        self.df_processed['水位_MA30'] = self.df_processed['水位_水位'].rolling(3, center=True).mean()
        
        return self.df_processed
    
    def detect_change_points(self, ma_window='30'):
        """放流量の変化点（ピーク・ボトム）検出"""
        print(f"\n=== 変化点検出（{ma_window}分移動平均） ===")
        
        if ma_window == '30':
            discharge_ma = self.df_processed['放流量_MA30'].values
        else:
            discharge_ma = self.df_processed['放流量_MA60'].values
        
        # NaNを除去
        valid_idx = ~np.isnan(discharge_ma)
        discharge_clean = discharge_ma[valid_idx]
        time_clean = self.df_processed['時刻'].values[valid_idx]
        
        # ピークとボトムの検出
        peaks, _ = signal.find_peaks(discharge_clean, distance=6)  # 最小60分間隔
        bottoms, _ = signal.find_peaks(-discharge_clean, distance=6)
        
        # 結果を保存
        self.change_points = {
            'peaks': {
                'indices': np.where(valid_idx)[0][peaks],
                'times': time_clean[peaks],
                'values': discharge_clean[peaks]
            },
            'bottoms': {
                'indices': np.where(valid_idx)[0][bottoms],
                'times': time_clean[bottoms],
                'values': discharge_clean[bottoms]
            }
        }
        
        print(f"検出されたピーク数: {len(peaks)}")
        print(f"検出されたボトム数: {len(bottoms)}")
        
        return self.change_points
    
    def calculate_change_point_delays(self):
        """変化点での遅延時間計算"""
        print("\n=== 変化点での遅延時間計算 ===")
        
        delay_results = []
        
        # ピーク（放流量増加→水位増加）の遅延
        for i, idx in enumerate(self.change_points['peaks']['indices']):
            # 放流量がピークに達した時刻から、水位が増加に転じるまでの時間
            water_level = self.df_processed['水位_MA30'].values[idx:]
            
            # 水位の変化率を計算
            water_diff = np.diff(water_level)
            
            # 水位が明確に増加し始める点を探す（ノイズを考慮）
            for j in range(len(water_diff)-3):
                if all(water_diff[j:j+3] > 0.001):  # 3点連続で増加
                    delay_minutes = (j + 1) * 10
                    delay_results.append({
                        'type': 'peak',
                        'time': self.change_points['peaks']['times'][i],
                        'discharge': self.change_points['peaks']['values'][i],
                        'delay_minutes': delay_minutes,
                        'water_level': self.df_processed['水位_水位'].iloc[idx]
                    })
                    break
        
        # ボトム（放流量減少→水位減少）の遅延
        for i, idx in enumerate(self.change_points['bottoms']['indices']):
            water_level = self.df_processed['水位_MA30'].values[idx:]
            water_diff = np.diff(water_level)
            
            for j in range(len(water_diff)-3):
                if all(water_diff[j:j+3] < -0.001):  # 3点連続で減少
                    delay_minutes = (j + 1) * 10
                    delay_results.append({
                        'type': 'bottom',
                        'time': self.change_points['bottoms']['times'][i],
                        'discharge': self.change_points['bottoms']['values'][i],
                        'delay_minutes': delay_minutes,
                        'water_level': self.df_processed['水位_水位'].iloc[idx]
                    })
                    break
        
        self.delay_times = pd.DataFrame(delay_results)
        
        if len(self.delay_times) > 0:
            print(f"計算された遅延時間数: {len(self.delay_times)}")
            print(f"平均遅延時間: {self.delay_times['delay_minutes'].mean():.1f}分")
            print(f"遅延時間範囲: {self.delay_times['delay_minutes'].min()}-{self.delay_times['delay_minutes'].max()}分")
        else:
            print("警告: 遅延時間が計算できませんでした")
        
        return self.delay_times
    
    def calculate_cross_correlation(self):
        """相互相関による遅延時間分析"""
        print("\n=== 相互相関分析 ===")
        
        # 全体での相互相関
        discharge = self.df_processed['ダム_全放流量'].values
        water_level = self.df_processed['水位_水位'].values
        
        # NaNを除去
        mask = ~(np.isnan(discharge) | np.isnan(water_level))
        discharge_clean = discharge[mask]
        water_level_clean = water_level[mask]
        
        # 相互相関計算（最大遅延: 300分 = 30データ点）
        max_lag = 30
        correlations = []
        lags = range(-max_lag, max_lag + 1)
        
        for lag in lags:
            if lag < 0:
                corr = np.corrcoef(discharge_clean[:lag], water_level_clean[-lag:])[0, 1]
            elif lag > 0:
                corr = np.corrcoef(discharge_clean[lag:], water_level_clean[:-lag])[0, 1]
            else:
                corr = np.corrcoef(discharge_clean, water_level_clean)[0, 1]
            correlations.append(corr)
        
        # 最大相関とその遅延
        correlations = np.array(correlations)
        max_corr_idx = np.argmax(correlations)
        max_corr_lag = lags[max_corr_idx]
        max_corr_value = correlations[max_corr_idx]
        
        print(f"最大相関係数: {max_corr_value:.3f}")
        print(f"最適遅延時間: {max_corr_lag * 10}分")
        
        # 放流量レベル別の相互相関（100 m³/sピッチ）
        discharge_levels = []
        for i in range(0, 1100, 100):  # 0から1000まで100刻み
            discharge_levels.append((i, i + 100))
        
        level_results = []
        
        for min_q, max_q in discharge_levels:
            mask_level = (discharge_clean >= min_q) & (discharge_clean < max_q)
            if mask_level.sum() > 10:  # データが10個以上あれば分析（閾値を下げる）
                discharge_level = discharge_clean[mask_level]
                water_level_level = water_level_clean[mask_level]
                
                # 簡易的に最大相関の遅延を計算
                best_lag = 0
                best_corr = 0
                for lag in range(0, 31):  # 0-300分の遅延
                    if lag > 0 and len(discharge_level) > lag:
                        corr = np.corrcoef(discharge_level[:-lag], water_level_level[lag:])[0, 1]
                        if corr > best_corr:
                            best_corr = corr
                            best_lag = lag
                
                level_results.append({
                    'discharge_range': f"{min_q}-{max_q}",
                    'optimal_lag_minutes': best_lag * 10,
                    'max_correlation': best_corr,
                    'data_points': mask_level.sum()
                })
        
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
    
    def fit_nonlinear_model(self):
        """非線形モデルのフィッティング"""
        print("\n=== 非線形モデル構築 ===")
        
        if self.delay_times is None or len(self.delay_times) == 0:
            print("警告: 遅延時間データがないため、モデル構築をスキップします")
            return None
        
        # モデル1: べき乗則
        def power_law(x, a, b):
            return a * np.power(x, b)
        
        # モデル2: シグモイド関数を含む非線形モデル（氾濫水位考慮）
        def sigmoid_model(x, a, b, c, d):
            # 基本のべき乗則 + 氾濫水位での飽和効果
            base = a * np.power(x, b)
            # 水位5.5m付近での飽和をモデル化
            saturation = c / (1 + np.exp(-d * (x - 300)))  # 300 m³/s付近で変化
            return base + saturation
        
        # データ準備
        x_data = self.delay_times['discharge'].values
        y_data = self.delay_times['delay_minutes'].values
        
        try:
            # べき乗則フィッティング
            popt_power, _ = curve_fit(power_law, x_data, y_data, p0=[100, -0.3])
            y_pred_power = power_law(x_data, *popt_power)
            rmse_power = np.sqrt(np.mean((y_data - y_pred_power)**2))
            r2_power = 1 - np.sum((y_data - y_pred_power)**2) / np.sum((y_data - y_data.mean())**2)
            
            # シグモイドモデルフィッティング
            popt_sigmoid, _ = curve_fit(sigmoid_model, x_data, y_data, p0=[100, -0.3, 50, 0.01])
            y_pred_sigmoid = sigmoid_model(x_data, *popt_sigmoid)
            rmse_sigmoid = np.sqrt(np.mean((y_data - y_pred_sigmoid)**2))
            r2_sigmoid = 1 - np.sum((y_data - y_pred_sigmoid)**2) / np.sum((y_data - y_data.mean())**2)
            
            self.model_params = {
                'power_law': {
                    'params': popt_power,
                    'equation': f"τ = {popt_power[0]:.2f} × Q^({popt_power[1]:.3f})",
                    'rmse': rmse_power,
                    'r2': r2_power
                },
                'sigmoid': {
                    'params': popt_sigmoid,
                    'equation': f"τ = {popt_sigmoid[0]:.2f} × Q^({popt_sigmoid[1]:.3f}) + {popt_sigmoid[2]:.2f}/(1+exp(-{popt_sigmoid[3]:.4f}×(Q-300)))",
                    'rmse': rmse_sigmoid,
                    'r2': r2_sigmoid
                }
            }
            
            print(f"べき乗則モデル: {self.model_params['power_law']['equation']}")
            print(f"  RMSE: {rmse_power:.2f}分, R²: {r2_power:.3f}")
            print(f"非線形モデル: RMSE: {rmse_sigmoid:.2f}分, R²: {r2_sigmoid:.3f}")
            
        except Exception as e:
            print(f"モデルフィッティングエラー: {e}")
            self.model_params = None
        
        return self.model_params
    
    def visualize_results(self):
        """結果の可視化"""
        print("\n=== 結果の可視化 ===")
        
        fig = plt.figure(figsize=(16, 12))
        
        # 1. 時系列データ
        ax1 = plt.subplot(3, 2, 1)
        time_range = slice(0, min(1440, len(self.df_processed)))  # 最初の10日間
        ax1.plot(self.df_processed['時刻'].iloc[time_range], 
                self.df_processed['ダム_全放流量'].iloc[time_range], 
                'b-', label='放流量', alpha=0.7)
        ax1.set_ylabel('放流量 (m³/s)', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        ax1.set_xlabel('時刻')
        ax1.grid(True, alpha=0.3)
        
        ax1_twin = ax1.twinx()
        ax1_twin.plot(self.df_processed['時刻'].iloc[time_range], 
                     self.df_processed['水位_水位'].iloc[time_range], 
                     'r-', label='水位', alpha=0.7)
        ax1_twin.axhline(y=5.5, color='r', linestyle='--', alpha=0.5, label='氾濫水位')
        ax1_twin.set_ylabel('水位 (m)', color='r')
        ax1_twin.tick_params(axis='y', labelcolor='r')
        
        # 変化点をマーク
        if self.change_points:
            peak_times = pd.to_datetime(self.change_points['peaks']['times'])
            peak_mask = peak_times < self.df_processed['時刻'].iloc[time_range].max()
            ax1.scatter(peak_times[peak_mask], 
                       self.change_points['peaks']['values'][peak_mask],
                       color='blue', s=50, marker='^', label='ピーク')
        
        ax1.set_title('時系列データと変化点')
        ax1.legend(loc='upper left')
        ax1_twin.legend(loc='upper right')
        
        # 2. 変化点遅延時間の散布図
        if self.delay_times is not None and len(self.delay_times) > 0:
            ax2 = plt.subplot(3, 2, 2)
            peaks_data = self.delay_times[self.delay_times['type'] == 'peak']
            bottoms_data = self.delay_times[self.delay_times['type'] == 'bottom']
            
            ax2.scatter(peaks_data['discharge'], peaks_data['delay_minutes'], 
                       color='red', label='上昇時', alpha=0.6, s=50)
            ax2.scatter(bottoms_data['discharge'], bottoms_data['delay_minutes'], 
                       color='blue', label='下降時', alpha=0.6, s=50)
            
            # モデル曲線
            if self.model_params:
                x_model = np.linspace(10, self.delay_times['discharge'].max(), 100)
                y_power = self.model_params['power_law']['params'][0] * np.power(x_model, self.model_params['power_law']['params'][1])
                ax2.plot(x_model, y_power, 'k--', label='べき乗則', linewidth=2)
            
            ax2.set_xlabel('放流量 (m³/s)')
            ax2.set_ylabel('遅延時間 (分)')
            ax2.set_title('変化点検出の遅延時間')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_xscale('log')
        
        # 3. 相互相関
        if self.correlation_results:
            ax3 = plt.subplot(3, 2, 3)
            ax3.plot(self.correlation_results['overall']['lags'], 
                    self.correlation_results['overall']['correlations'])
            opt_lag = self.correlation_results['overall']['optimal_lag_minutes']
            opt_corr = self.correlation_results['overall']['max_correlation']
            ax3.axvline(x=opt_lag, color='r', linestyle='--', 
                       label=f'最適遅延: {opt_lag}分 (r={opt_corr:.3f})')
            ax3.set_xlabel('遅延時間 (分)')
            ax3.set_ylabel('相関係数')
            ax3.set_title('相互相関分析（水位推定）')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 4. 放流量レベル別遅延時間
        if self.correlation_results and 'by_level' in self.correlation_results:
            ax4 = plt.subplot(3, 2, 4)
            level_data = self.correlation_results['by_level']
            # 固定のX軸ラベル（0〜1200の100刻み）
            x_labels = [f"{i}-{i+100}" for i in range(0, 1200, 100)]
            x_pos = range(len(x_labels))

            # データの辞書化 → 欠損ラベルに対応（値がない場合は0）
            lag_dict = dict(zip(level_data['discharge_range'], level_data['optimal_lag_minutes']))
            display_values = [lag_dict.get(label, 0) for label in x_labels]

            ax4.bar(x_pos, display_values)
            ax4.set_xticks(x_pos)
            ax4.set_xticklabels(x_labels, rotation=45)
            ax4.set_xlabel('放流量範囲 (m³/s)')
            ax4.set_ylabel('最適遅延時間 (分)')
            ax4.set_title('放流量レベル別の遅延時間')
            ax4.grid(True, alpha=0.3, axis='y')
        
        # 5. 水位vs放流量（氾濫水位の影響）
        ax5 = plt.subplot(3, 2, 5)
        mask = ~(np.isnan(self.df_processed['ダム_全放流量']) | np.isnan(self.df_processed['水位_水位']))
        discharge_clean = self.df_processed['ダム_全放流量'][mask]
        water_clean = self.df_processed['水位_水位'][mask]
        
        # 氾濫水位以下と以上で色分け
        below_flood = water_clean < 5.5
        ax5.scatter(discharge_clean[below_flood], water_clean[below_flood], 
                   alpha=0.3, s=1, color='blue', label='通常時')
        ax5.scatter(discharge_clean[~below_flood], water_clean[~below_flood], 
                   alpha=0.5, s=2, color='red', label='氾濫水位超過')
        ax5.axhline(y=5.5, color='red', linestyle='--', alpha=0.7, label='氾濫水位')
        ax5.set_xlabel('放流量 (m³/s)')
        ax5.set_ylabel('水位 (m)')
        ax5.set_title('放流量と水位の関係')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. 統計情報
        ax6 = plt.subplot(3, 2, 6)
        ax6.axis('off')
        stats_text = "=== 分析結果サマリー ===\n\n"
        
        if self.delay_times is not None and len(self.delay_times) > 0:
            stats_text += f"【変化点検出の遅延時間】\n"
            stats_text += f"平均: {self.delay_times['delay_minutes'].mean():.1f}分\n"
            stats_text += f"中央値: {self.delay_times['delay_minutes'].median():.1f}分\n"
            stats_text += f"範囲: {self.delay_times['delay_minutes'].min()}-{self.delay_times['delay_minutes'].max()}分\n\n"
        
        if self.correlation_results:
            stats_text += f"【相互相関（水位推定）】\n"
            stats_text += f"最適遅延: {self.correlation_results['overall']['optimal_lag_minutes']}分\n"
            stats_text += f"最大相関: {self.correlation_results['overall']['max_correlation']:.3f}\n\n"
        
        if self.model_params:
            stats_text += f"【回帰モデル】\n"
            stats_text += f"{self.model_params['power_law']['equation']}\n"
            stats_text += f"R² = {self.model_params['power_law']['r2']:.3f}\n"
        
        ax6.text(0.1, 0.9, stats_text, transform=ax6.transAxes, 
                fontsize=11, verticalalignment='top', family='monospace')
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def save_results(self):
        """結果の保存"""
        print("\n=== 結果の保存 ===")
        
        root = tk.Tk()
        root.withdraw()
        
        # タイムスタンプ付きファイル名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"遅延時間分析結果_{timestamp}.csv"
        
        file_path = filedialog.asksaveasfilename(
            title="結果を保存",
            defaultextension=".csv",
            initialfile=default_name,
            filetypes=[("CSV files", "*.csv")]
        )
        
        if file_path:
            # 遅延時間テーブルの保存
            if self.delay_times is not None and len(self.delay_times) > 0:
                self.delay_times.to_csv(file_path, index=False, encoding='utf-8-sig')
                print(f"変化点遅延時間を保存: {file_path}")
            
            # 相関分析結果の保存
            if self.correlation_results and 'by_level' in self.correlation_results:
                corr_path = file_path.replace('.csv', '_相関分析.csv')
                self.correlation_results['by_level'].to_csv(corr_path, index=False, encoding='utf-8-sig')
                print(f"相関分析結果を保存: {corr_path}")
            
            # モデルパラメータの保存
            if self.model_params:
                model_path = file_path.replace('.csv', '_モデル.txt')
                with open(model_path, 'w', encoding='utf-8') as f:
                    f.write("=== 遅延時間予測モデル ===\n\n")
                    f.write("【べき乗則モデル】\n")
                    f.write(f"式: {self.model_params['power_law']['equation']}\n")
                    f.write(f"RMSE: {self.model_params['power_law']['rmse']:.2f}分\n")
                    f.write(f"R²: {self.model_params['power_law']['r2']:.3f}\n")
                print(f"モデルパラメータを保存: {model_path}")

def main():
    """メイン処理"""
    print("河川水位遅延時間分析プログラム")
    print("=" * 50)
    
    analyzer = RiverDelayAnalysis()
    
    try:
        # 1. データ読み込み
        analyzer.load_data()
        
        # 2. 前処理
        analyzer.preprocess_data()
        
        # 3. 変化点検出
        analyzer.detect_change_points(ma_window='30')
        
        # 4. 変化点での遅延時間計算
        analyzer.calculate_change_point_delays()
        
        # 5. 相互相関分析
        analyzer.calculate_cross_correlation()
        
        # 6. 非線形モデル構築
        analyzer.fit_nonlinear_model()
        
        # 7. 可視化
        analyzer.visualize_results()
        
        # 8. 結果保存
        analyzer.save_results()
        
        print("\n分析完了！")
        
    except Exception as e:
        print(f"\nエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()