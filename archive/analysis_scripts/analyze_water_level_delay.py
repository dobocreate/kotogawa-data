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
        
        print(f"\n=== 相互相関分析（Figure1）のデータ確認 ===")
        print(f"元データ数: {len(discharge)}")
        print(f"NaN除去後データ数: {len(discharge_clean)}")
        print(f"データのメモリアドレス: {id(discharge_clean)}")
        
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
        for i in range(0, 1200, 100):  # 0から1100まで100刻み（1100-1200も含める）
            discharge_levels.append((i, i + 100))
        
        level_results = []
        
        # 動的閾値：5個固定（データが少ないレベルも分析対象にする）
        dynamic_threshold = 5
        print(f"動的閾値: {dynamic_threshold}個")
        
        for min_q, max_q in discharge_levels:
            mask_level = (discharge_clean >= min_q) & (discharge_clean < max_q)
            data_count = mask_level.sum()
            
            if data_count >= dynamic_threshold:  # 動的閾値を使用
                discharge_level = discharge_clean[mask_level]
                water_level_level = water_level_clean[mask_level]
                
                # 簡易的に最大相関の遅延を計算
                best_lag = 0
                best_corr = 0
                for lag in range(0, 31):  # 0-300分の遅延
                    if lag > 0 and len(discharge_level) > lag:
                        try:
                            corr = np.corrcoef(discharge_level[:-lag], water_level_level[lag:])[0, 1]
                            if not np.isnan(corr) and corr > best_corr:
                                best_corr = corr
                                best_lag = lag
                        except:
                            continue
                
                level_results.append({
                    'discharge_range': f"{min_q}-{max_q}",
                    'optimal_lag_minutes': best_lag * 10,
                    'max_correlation': best_corr,
                    'data_points': data_count
                })
                print(f"  {min_q}-{max_q} m³/s: {data_count}件 → 遅延{best_lag * 10}分 (r={best_corr:.3f})")
            else:
                print(f"  {min_q}-{max_q} m³/s: {data_count}件 (閾値未満、スキップ)")
        
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
    
    def event_separation_analysis(self):
        """イベント分離分析: 放流量の急変イベントを抽出し、前後1時間の安定期間での遅延時間を分析"""
        print("\n=== イベント分離分析 ===")
        
        if self.df_processed is None:
            print("データが読み込まれていません。")
            return None
        
        # データ準備
        df = self.df_processed.copy()
        df = df.dropna(subset=['ダム_全放流量', '水位_水位'])
        df = df.reset_index(drop=True)
        
        discharge = df['ダム_全放流量'].values
        water_level = df['水位_水位'].values
        
        print(f"分析対象データ数: {len(discharge)}")
        
        # 1. 放流量の変化点検出
        events = self._detect_discharge_events(discharge)
        print(f"検出されたイベント数: {len(events)}")
        
        if len(events) == 0:
            print("有効なイベントが検出されませんでした。")
            return None
        
        # 2. 各イベントでの遅延時間計算
        event_results = []
        valid_events = 0
        
        for i, event in enumerate(events):
            result = self._analyze_single_event(event, discharge, water_level, i)
            if result is not None:
                event_results.append(result)
                valid_events += 1
        
        print(f"有効なイベント数: {valid_events}")
        
        if len(event_results) == 0:
            print("分析可能なイベントがありませんでした。")
            return None
        
        # 3. 結果の統計処理
        event_df = pd.DataFrame(event_results)
        
        # 統計サマリー
        print("\n=== イベント分析結果統計 ===")
        print(f"平均遅延時間: {event_df['delay_minutes'].mean():.1f} ± {event_df['delay_minutes'].std():.1f}分")
        print(f"中央値遅延時間: {event_df['delay_minutes'].median():.1f}分")
        print(f"遅延時間範囲: {event_df['delay_minutes'].min():.1f} - {event_df['delay_minutes'].max():.1f}分")
        
        # 放流量規模別の統計
        print("\n=== 放流量規模別遅延時間 ===")
        event_df['discharge_level'] = pd.cut(event_df['discharge_change'], 
                                           bins=[0, 50, 100, 200, 500, 1000, np.inf],
                                           labels=['~50', '50-100', '100-200', '200-500', '500-1000', '1000+'])
        
        level_stats = event_df.groupby('discharge_level')['delay_minutes'].agg(['count', 'mean', 'std']).round(1)
        print(level_stats)
        
        # 500-700 m³/s範囲の詳細分析
        print("\n=== 500-700 m³/s範囲の詳細分析 ===")
        mask_500_700 = ((event_df['discharge_before'] >= 500) & (event_df['discharge_before'] < 700)) | \
                       ((event_df['discharge_after'] >= 500) & (event_df['discharge_after'] < 700))
        
        if mask_500_700.any():
            events_500_700 = event_df[mask_500_700]
            print(f"500-700範囲のイベント数: {len(events_500_700)}")
            print(f"平均遅延時間: {events_500_700['delay_minutes'].mean():.1f}分")
            print(f"標準偏差: {events_500_700['delay_minutes'].std():.1f}分")
            print("\n個別イベント:")
            for idx, event in events_500_700.iterrows():
                print(f"  Event {event['event_id']}: {event['discharge_before']:.0f} → {event['discharge_after']:.0f} m³/s, "
                      f"遅延{event['delay_minutes']}分, スコア{event['response_score']:.3f}")
        
        self.event_results = {
            'events': event_df,
            'summary': {
                'total_events': len(events),
                'valid_events': valid_events,
                'mean_delay': event_df['delay_minutes'].mean(),
                'median_delay': event_df['delay_minutes'].median(),
                'std_delay': event_df['delay_minutes'].std()
            }
        }
        
        return self.event_results
    
    def classify_stable_variable_periods(self, window_hours=2, cv_threshold=0.05):
        """データを安定期間と変動期間に分類（改善版）"""
        print("\n=== 安定期間・変動期間の分類（改善版） ===")
        
        if self.df_processed is None:
            print("データが読み込まれていません。")
            return None
        
        window_points = int(window_hours * 6)  # 2時間 = 12ポイント
        discharge = self.df_processed['ダム_全放流量'].values
        water_level = self.df_processed['水位_水位'].values
        
        # 各時点での変動係数を計算
        cv_values = []
        period_types = []
        discharge_changes = []  # 放流量変化率
        level_changes = []      # 水位変化率
        
        for i in range(len(discharge)):
            start_idx = max(0, i - window_points // 2)
            end_idx = min(len(discharge), i + window_points // 2)
            
            window_discharge = discharge[start_idx:end_idx]
            window_level = water_level[start_idx:end_idx]
            
            if len(window_discharge) > 0 and np.mean(window_discharge) > 0:
                # 1. 放流量の変動係数
                cv_discharge = np.std(window_discharge) / np.mean(window_discharge)
                cv_values.append(cv_discharge)
                
                # 2. 放流量の変化率（前後1時間の変化）
                if i >= 6 and i < len(discharge) - 6:
                    discharge_change = abs(discharge[i+6] - discharge[i-6]) / max(discharge[i], 1)
                    discharge_changes.append(discharge_change)
                else:
                    discharge_change = 0
                    discharge_changes.append(0)
                
                # 3. 水位の変化率
                if i >= 6 and i < len(water_level) - 6:
                    level_change = abs(water_level[i+6] - water_level[i-6])
                    level_changes.append(level_change)
                else:
                    level_change = 0
                    level_changes.append(0)
                
                # 4. 安定期間の改善された定義：
                # 重要: 単なる低変動ではなく、「遅延時間分析に適した安定期間」を抽出
                
                mean_discharge = np.mean(window_discharge)
                
                # a. 基本的な安定性条件（放流量レベル別）
                if mean_discharge < 50:
                    # 超低流量: 非常に緩い条件（ノイズが多いため）
                    cv_threshold = 0.2
                    change_threshold = 0.2
                elif mean_discharge < 100:
                    # 低流量: やや緩い条件
                    cv_threshold = 0.1
                    change_threshold = 0.1
                elif mean_discharge < 300:
                    # 中流量: 標準的な条件
                    cv_threshold = 0.05
                    change_threshold = 0.05
                elif mean_discharge < 500:
                    # 高流量: 厳しい条件
                    cv_threshold = 0.03
                    change_threshold = 0.03
                else:
                    # 超高流量: 非常に厳しい条件
                    cv_threshold = 0.02
                    change_threshold = 0.02
                
                # b. 追加条件：前後に大きな変化がある場合は優先的に安定期間とする
                # （遅延時間分析に最適）
                future_change = False
                past_change = False
                
                if i < len(discharge) - 18:  # 3時間後を確認
                    future_change = abs(discharge[i+18] - discharge[i]) / max(discharge[i], 1) > 0.2
                if i >= 18:  # 3時間前を確認
                    past_change = abs(discharge[i] - discharge[i-18]) / max(discharge[i-18], 1) > 0.2
                
                # イベント前後の安定期間は条件を緩和
                if future_change or past_change:
                    cv_threshold *= 1.5
                    change_threshold *= 1.5
                
                # c. 最終判定
                is_stable = (cv_discharge < cv_threshold and 
                           discharge_change < change_threshold and 
                           level_change < 0.15)  # 水位変化の閾値も緩和
                
                period_types.append('stable' if is_stable else 'variable')
            else:
                cv_values.append(np.nan)
                discharge_changes.append(np.nan)
                level_changes.append(np.nan)
                period_types.append('unknown')
        
        self.df_processed['cv'] = cv_values
        self.df_processed['period_type'] = period_types
        
        # 期間の統計
        stable_count = sum(1 for p in period_types if p == 'stable')
        variable_count = sum(1 for p in period_types if p == 'variable')
        
        print(f"分類結果:")
        print(f"  安定期間: {stable_count}ポイント ({stable_count/len(period_types)*100:.1f}%)")
        print(f"  変動期間: {variable_count}ポイント ({variable_count/len(period_types)*100:.1f}%)")
        
        # 連続する期間を抽出
        self.stable_periods = []
        self.variable_periods = []
        
        current_type = None
        start_idx = 0
        
        for i, period_type in enumerate(period_types):
            if period_type != current_type:
                if current_type == 'stable' and i - start_idx >= 6:  # 最低1時間継続
                    self.stable_periods.append((start_idx, i))
                elif current_type == 'variable' and i - start_idx >= 6:
                    self.variable_periods.append((start_idx, i))
                current_type = period_type
                start_idx = i
        
        print(f"\n連続期間数:")
        print(f"  安定期間: {len(self.stable_periods)}個")
        print(f"  変動期間: {len(self.variable_periods)}個")
        
        return self.df_processed
    
    def classify_stable_variable_periods_detailed(self):
        """より詳細な期間分類（イベントベース）"""
        print("\n=== 詳細な期間分類（イベントベース） ===")
        
        if self.df_processed is None:
            print("データが読み込まれていません。")
            return None
            
        discharge = self.df_processed['ダム_全放流量'].values
        water_level = self.df_processed['水位_水位'].values
        
        # 1. 放流量の変化点を検出
        discharge_diff = np.diff(discharge)
        discharge_diff = np.concatenate([[0], discharge_diff])
        
        # 2. 大きな変化イベントを特定（標準偏差の2倍以上）
        change_threshold = np.std(discharge_diff) * 2
        significant_changes = np.abs(discharge_diff) > max(change_threshold, 10)  # 最低10m³/s
        
        # 3. イベントを検出
        events = []
        in_event = False
        event_start = 0
        
        for i in range(len(significant_changes)):
            if significant_changes[i] and not in_event:
                in_event = True
                event_start = i
            elif not significant_changes[i] and in_event:
                in_event = False
                if i - event_start >= 3:  # 最低30分継続
                    events.append((event_start, i))
        
        # 4. 詳細な期間タイプを割り当て
        detailed_types = ['unknown'] * len(discharge)
        
        for event_start, event_end in events:
            # イベント期間
            for j in range(event_start, event_end):
                detailed_types[j] = 'event'
            
            # イベント前の安定期間（最大6時間）
            pre_start = max(0, event_start - 36)
            pre_end = event_start
            
            if pre_end - pre_start >= 12:  # 最低2時間
                # 安定性チェック
                pre_discharge = discharge[pre_start:pre_end]
                pre_cv = np.std(pre_discharge) / (np.mean(pre_discharge) + 1)
                
                if pre_cv < 0.1:  # 安定している
                    for j in range(pre_start, pre_end):
                        if detailed_types[j] == 'unknown':
                            detailed_types[j] = 'pre_stable'
            
            # イベント後の安定期間（水位応答の収束を待つ）
            post_start = event_end
            post_end = min(len(discharge), event_end + 72)  # 最大12時間
            
            # 水位の変化率が収束する点を探す
            for i in range(post_start + 12, post_end):
                if i + 6 < len(water_level):
                    window_level = water_level[i-6:i+6]
                    level_change_rate = np.std(np.diff(window_level))
                    
                    if level_change_rate < 0.01:  # 収束
                        # この時点から安定期間
                        stable_end = min(i + 36, len(discharge))
                        for j in range(i, stable_end):
                            if j < len(discharge) - 6:
                                local_change = abs(discharge[j+6] - discharge[j-6]) / max(discharge[j], 1)
                                if local_change < 0.05:
                                    detailed_types[j] = 'post_stable'
                                else:
                                    break
                        break
        
        # 5. 残りの期間を分類
        for i in range(len(detailed_types)):
            if detailed_types[i] == 'unknown':
                if 'period_type' in self.df_processed.columns:
                    # 既存の分類を使用
                    if self.df_processed['period_type'].iloc[i] == 'stable':
                        detailed_types[i] = 'low_activity'
                    else:
                        detailed_types[i] = 'variable'
                else:
                    detailed_types[i] = 'variable'
        
        # 結果を保存
        self.df_processed['detailed_period_type'] = detailed_types
        
        # 統計を表示
        type_counts = pd.Series(detailed_types).value_counts()
        print("\n詳細な期間分類結果:")
        for ptype, count in type_counts.items():
            percentage = count / len(detailed_types) * 100
            print(f"  {ptype}: {count}ポイント ({percentage:.1f}%)")
        
        # 高放流量域での分布を確認
        high_discharge_mask = discharge >= 300
        high_discharge_types = [detailed_types[i] for i in range(len(discharge)) if high_discharge_mask[i]]
        
        if high_discharge_types:
            print(f"\n高放流量域（≥300 m³/s）での分布:")
            high_type_counts = pd.Series(high_discharge_types).value_counts()
            for ptype, count in high_type_counts.items():
                percentage = count / len(high_discharge_types) * 100
                print(f"  {ptype}: {count}ポイント ({percentage:.1f}%)")
        
        return self.df_processed
    
    def analyze_stable_periods(self):
        """安定期間での遅延時間分析"""
        print("\n=== 安定期間の分析 ===")
        
        if not hasattr(self, 'stable_periods'):
            print("期間分類が実行されていません。")
            return None
        
        stable_results = []
        
        for period_idx, (start, end) in enumerate(self.stable_periods):
            # 期間内のデータ
            discharge = self.df_processed['ダム_全放流量'].iloc[start:end].values
            water_level = self.df_processed['水位_水位'].iloc[start:end].values
            
            # 欠損値を除去
            mask = ~(np.isnan(discharge) | np.isnan(water_level))
            if mask.sum() < 20:  # 最低200分のデータが必要
                continue
            
            discharge_clean = discharge[mask]
            water_level_clean = water_level[mask]
            
            # 相互相関による遅延時間計算
            max_lag = 15  # 最大150分
            correlations = []
            
            for lag in range(0, max_lag):
                if lag > 0 and len(discharge_clean) > lag:
                    corr = np.corrcoef(discharge_clean[:-lag], water_level_clean[lag:])[0, 1]
                else:
                    corr = np.corrcoef(discharge_clean, water_level_clean)[0, 1]
                correlations.append(corr if not np.isnan(corr) else 0)
            
            # 最適遅延を見つける
            best_lag = np.argmax(correlations)
            best_corr = correlations[best_lag]
            
            if best_corr > 0.3:  # 有意な相関
                avg_discharge = np.mean(discharge_clean)
                stable_results.append({
                    'period_idx': period_idx,
                    'start': start,
                    'end': end,
                    'duration_hours': (end - start) / 6,
                    'avg_discharge': avg_discharge,
                    'delay_minutes': best_lag * 10,
                    'correlation': best_corr,
                    'discharge_range': f"{int(avg_discharge//100)*100}-{int(avg_discharge//100+1)*100}"
                })
        
        self.stable_analysis_results = pd.DataFrame(stable_results)
        
        if len(stable_results) > 0:
            print(f"分析された安定期間数: {len(stable_results)}")
            print(f"平均遅延時間: {self.stable_analysis_results['delay_minutes'].mean():.1f}分")
            
            # 放流量範囲別の統計
            print("\n放流量範囲別遅延時間（安定期間）:")
            range_stats = self.stable_analysis_results.groupby('discharge_range')['delay_minutes'].agg(['count', 'mean', 'std'])
            print(range_stats)
        
        return self.stable_analysis_results
    
    def analyze_stable_periods_detailed(self):
        """詳細な期間タイプに基づく安定期間分析"""
        print("\n=== 詳細な安定期間分析 ===")
        
        if 'detailed_period_type' not in self.df_processed.columns:
            print("詳細な期間分類が実行されていません。")
            return None
        
        detailed_results = {
            'pre_stable': [],
            'post_stable': [],
            'low_activity': []
        }
        
        # 各期間タイプごとに連続する期間を抽出
        for period_type in ['pre_stable', 'post_stable', 'low_activity']:
            mask = self.df_processed['detailed_period_type'] == period_type
            
            # 連続する期間を見つける
            periods = []
            in_period = False
            start_idx = 0
            
            for i in range(len(mask)):
                if mask.iloc[i] and not in_period:
                    in_period = True
                    start_idx = i
                elif not mask.iloc[i] and in_period:
                    in_period = False
                    if i - start_idx >= 12:  # 最低2時間
                        periods.append((start_idx, i))
            
            # 各期間を分析
            for period_idx, (start, end) in enumerate(periods):
                discharge = self.df_processed['ダム_全放流量'].iloc[start:end].values
                water_level = self.df_processed['水位_水位'].iloc[start:end].values
                
                # 欠損値を除去
                valid_mask = ~(np.isnan(discharge) | np.isnan(water_level))
                if valid_mask.sum() < 12:
                    continue
                
                discharge_clean = discharge[valid_mask]
                water_level_clean = water_level[valid_mask]
                
                # 相互相関による遅延時間計算
                max_lag = 20  # 最大200分
                correlations = []
                
                for lag in range(0, max_lag):
                    if lag > 0 and len(discharge_clean) > lag:
                        corr = np.corrcoef(discharge_clean[:-lag], water_level_clean[lag:])[0, 1]
                    else:
                        corr = np.corrcoef(discharge_clean, water_level_clean)[0, 1]
                    correlations.append(corr if not np.isnan(corr) else 0)
                
                # 最適遅延を見つける
                best_lag = np.argmax(correlations)
                best_corr = correlations[best_lag]
                
                avg_discharge = np.mean(discharge_clean)
                
                # 期間前後の変化を確認
                if period_type == 'pre_stable' and end < len(self.df_processed) - 6:
                    future_change = abs(self.df_processed['ダム_全放流量'].iloc[end+6] - 
                                      self.df_processed['ダム_全放流量'].iloc[end]) / max(avg_discharge, 1)
                else:
                    future_change = 0
                
                if period_type == 'post_stable' and start >= 6:
                    past_change = abs(self.df_processed['ダム_全放流量'].iloc[start] - 
                                    self.df_processed['ダム_全放流量'].iloc[start-6]) / max(avg_discharge, 1)
                else:
                    past_change = 0
                
                result = {
                    'period_idx': period_idx,
                    'start': start,
                    'end': end,
                    'duration_hours': (end - start) / 6,
                    'avg_discharge': avg_discharge,
                    'delay_minutes': best_lag * 10,
                    'correlation': best_corr,
                    'discharge_range': f"{int(avg_discharge//100)*100}-{int(avg_discharge//100+1)*100}",
                    'future_change': future_change,
                    'past_change': past_change
                }
                
                detailed_results[period_type].append(result)
        
        # 結果をDataFrameに変換
        self.detailed_stable_results = {}
        for period_type, results in detailed_results.items():
            if results:
                self.detailed_stable_results[period_type] = pd.DataFrame(results)
                
                print(f"\n{period_type}の分析結果:")
                print(f"  期間数: {len(results)}")
                print(f"  平均遅延時間: {np.mean([r['delay_minutes'] for r in results]):.1f}分")
                print(f"  平均相関係数: {np.mean([r['correlation'] for r in results]):.3f}")
                
                # 放流量範囲別
                df_temp = pd.DataFrame(results)
                range_stats = df_temp.groupby('discharge_range').agg({
                    'delay_minutes': ['mean', 'std', 'count'],
                    'correlation': 'mean'
                })
                print(f"\n  放流量範囲別統計:")
                print(range_stats)
        
        return self.detailed_stable_results
    
    def analyze_variable_periods(self):
        """変動期間での微分・累積応答分析"""
        print("\n=== 変動期間の分析 ===")
        
        if not hasattr(self, 'variable_periods'):
            print("期間分類が実行されていません。")
            return None
        
        variable_results = []
        
        for period_idx, (start, end) in enumerate(self.variable_periods):
            if end - start < 12:  # 最低2時間
                continue
                
            # 期間内のデータ
            discharge = self.df_processed['ダム_全放流量'].iloc[start:end].values
            water_level = self.df_processed['水位_水位'].iloc[start:end].values
            
            # 欠損値を除去
            mask = ~(np.isnan(discharge) | np.isnan(water_level))
            if mask.sum() < 12:
                continue
            
            discharge_clean = discharge[mask]
            water_level_clean = water_level[mask]
            
            # 1. 通常の相関分析（安定期間と同じ手法）
            correlations = []
            for lag in range(0, 15):  # 0-140分
                if lag > 0 and len(discharge_clean) > lag:
                    corr = np.corrcoef(discharge_clean[:-lag], water_level_clean[lag:])[0, 1]
                else:
                    corr = np.corrcoef(discharge_clean, water_level_clean)[0, 1]
                correlations.append(corr if not np.isnan(corr) else 0)
            
            best_lag = np.argmax(correlations)
            best_corr = correlations[best_lag]
            
            # 2. 微分応答分析（補助的な分析として保持）
            discharge_diff = np.diff(discharge_clean)
            water_diff = np.diff(water_level_clean)
            
            # 遅延を考慮した微分相関
            diff_correlations = []
            for lag in range(0, 15):  # 最大150分
                if lag > 0 and len(discharge_diff) > lag:
                    corr = np.corrcoef(discharge_diff[:-lag], water_diff[lag:])[0, 1]
                else:
                    corr = np.corrcoef(discharge_diff, water_diff)[0, 1]
                diff_correlations.append(corr if not np.isnan(corr) else 0)
            
            best_diff_lag = np.argmax(np.abs(diff_correlations))
            best_diff_corr = diff_correlations[best_diff_lag]
            
            # 3. 累積応答分析（補助的な分析として保持）
            # 累積変化量
            discharge_cumsum = np.cumsum(discharge_clean - discharge_clean[0])
            water_cumsum = water_level_clean - water_level_clean[0]
            
            # 累積相関
            cumsum_correlations = []
            for lag in range(0, 15):
                if lag > 0 and len(discharge_cumsum) > lag:
                    corr = np.corrcoef(discharge_cumsum[:-lag], water_cumsum[lag:])[0, 1]
                else:
                    corr = np.corrcoef(discharge_cumsum, water_cumsum)[0, 1]
                cumsum_correlations.append(corr if not np.isnan(corr) else 0)
            
            best_cumsum_lag = np.argmax(cumsum_correlations)
            best_cumsum_corr = cumsum_correlations[best_cumsum_lag]
            
            # 増加期か減少期かを判定
            net_change = discharge_clean[-1] - discharge_clean[0]
            period_direction = 'increase' if net_change > 0 else 'decrease'
            
            avg_discharge = np.mean(discharge_clean)
            variable_results.append({
                'period_idx': period_idx,
                'start': start,
                'end': end,
                'duration_hours': (end - start) / 6,
                'avg_discharge': avg_discharge,
                'direction': period_direction,
                'net_change': net_change,
                'delay_minutes': best_lag * 10,  # 通常の遅延時間（安定期間と同じ）
                'correlation': best_corr,  # 通常の相関係数
                'diff_delay_minutes': best_diff_lag * 10,
                'diff_correlation': best_diff_corr,
                'cumsum_delay_minutes': best_cumsum_lag * 10,
                'cumsum_correlation': best_cumsum_corr,
                'discharge_range': f"{int(avg_discharge//100)*100}-{int(avg_discharge//100+1)*100}"
            })
        
        self.variable_analysis_results = pd.DataFrame(variable_results)
        
        if len(variable_results) > 0:
            print(f"分析された変動期間数: {len(variable_results)}")
            print(f"\n通常の相関分析（安定期間と同じ手法）:")
            print(f"  平均遅延時間: {self.variable_analysis_results['delay_minutes'].mean():.1f}分")
            print(f"  平均相関係数: {self.variable_analysis_results['correlation'].mean():.3f}")
            print(f"\n微分応答:")
            print(f"  平均遅延時間: {self.variable_analysis_results['diff_delay_minutes'].mean():.1f}分")
            print(f"\n累積応答:")
            print(f"  平均遅延時間: {self.variable_analysis_results['cumsum_delay_minutes'].mean():.1f}分")
            
            # 増加期と減少期の比較
            print("\n増加期 vs 減少期:")
            direction_stats = self.variable_analysis_results.groupby('direction')[['delay_minutes', 'diff_delay_minutes', 'cumsum_delay_minutes']].mean()
            print(direction_stats)
        
        return self.variable_analysis_results
    
    
    def _detect_discharge_events(self, discharge, min_change=None, stability_hours=1.0):
        """放流量の急変イベントを検出"""
        events = []
        stability_points = int(stability_hours * 6)  # 10分間隔データでの1時間 = 6ポイント
        
        # 変化量の計算（移動平均を使用してノイズを軽減）
        window_size = 3
        discharge_smooth = np.convolve(discharge, np.ones(window_size)/window_size, mode='same')
        
        # 最小変化量を動的に設定（データの標準偏差の1/2）
        if min_change is None:
            discharge_std = np.std(discharge_smooth)
            min_change = discharge_std * 0.5
            print(f"  動的最小変化量: {min_change:.1f} m³/s（標準偏差の50%）")
        
        for i in range(stability_points, len(discharge) - stability_points - 30):  # 30は最大遅延
            # 現在の変化量を計算
            change = abs(discharge_smooth[i+1] - discharge_smooth[i])
            
            if change >= min_change:
                # 前後の安定性をチェック
                pre_stable = self._check_stability(discharge_smooth, i-stability_points, i)
                post_stable = self._check_stability(discharge_smooth, i+1, i+1+stability_points)
                
                if pre_stable and post_stable:
                    events.append({
                        'event_time': i,
                        'discharge_before': np.mean(discharge_smooth[i-stability_points:i]),
                        'discharge_after': np.mean(discharge_smooth[i+1:i+1+stability_points]),
                        'change_magnitude': change,
                        'direction': 'increase' if discharge_smooth[i+1] > discharge_smooth[i] else 'decrease'
                    })
        
        return events
    
    def _check_stability(self, data, start_idx, end_idx, max_cv=0.1):
        """期間内のデータの安定性をチェック（変動係数で評価）"""
        if start_idx < 0 or end_idx >= len(data) or start_idx >= end_idx:
            return False
        
        segment = data[start_idx:end_idx]
        if len(segment) < 3:
            return False
        
        mean_val = np.mean(segment)
        if mean_val == 0:
            return np.std(segment) < 1  # 絶対値での小さな変動
        
        cv = np.std(segment) / mean_val  # 変動係数
        return cv <= max_cv
    
    def _analyze_single_event(self, event, discharge, water_level, event_id):
        """単一イベントの遅延時間分析"""
        event_time = event['event_time']
        
        # 分析範囲の設定（前後1時間 + 最大遅延時間30分）
        analysis_start = max(0, event_time - 6)
        analysis_end = min(len(discharge), event_time + 36)  # イベント後36ポイント（6時間）
        
        if analysis_end - analysis_start < 20:  # 最小分析窓
            return None
        
        # イベント前後のデータ
        discharge_segment = discharge[analysis_start:analysis_end]
        water_level_segment = water_level[analysis_start:analysis_end]
        
        # 遅延時間の探索（10-100分、10分刻み）
        best_delay = 0
        best_score = 0
        delay_scores = []
        
        for delay_minutes in range(10, 110, 10):  # 10-100分、10分刻み
            delay_points = delay_minutes // 10  # 10分間隔データでの遅延ポイント数
            
            if len(discharge_segment) <= delay_points:
                continue
            
            # イベント前後の対応関係を評価
            score = self._calculate_event_response_score(
                discharge_segment, water_level_segment, 
                event_time - analysis_start, delay_points
            )
            
            delay_scores.append(score)
            
            if score > best_score:
                best_score = score
                best_delay = delay_minutes
        
        # 有効性の判定
        if best_score < 0.3:  # 相関係数0.3以上
            return None
        
        return {
            'event_id': event_id,
            'event_time': event_time,
            'discharge_before': event['discharge_before'],
            'discharge_after': event['discharge_after'],
            'discharge_change': event['change_magnitude'],
            'direction': event['direction'],
            'delay_minutes': best_delay,
            'response_score': best_score,
            'delay_scores': delay_scores
        }
    
    def _calculate_event_response_score(self, discharge, water_level, event_pos, delay_points):
        """イベント応答スコアの計算（相互相関ベース）"""
        try:
            # イベント前後の窓を広げて相関を計算
            window_before = 6  # 前60分
            window_after = 12  # 後120分
            
            # 分析範囲の設定
            start_discharge = max(0, event_pos - window_before)
            end_discharge = min(len(discharge), event_pos + window_after)
            
            start_water = max(0, event_pos - window_before + delay_points)
            end_water = min(len(water_level), event_pos + window_after + delay_points)
            
            # 有効な範囲を計算
            valid_length = min(end_discharge - start_discharge, end_water - start_water)
            if valid_length < 10:  # 最小100分のデータが必要
                return 0
            
            # 対応する部分を抽出
            discharge_segment = discharge[start_discharge:start_discharge + valid_length]
            water_segment = water_level[start_water:start_water + valid_length]
            
            # 変化量を計算
            discharge_diff = np.diff(discharge_segment)
            water_diff = np.diff(water_segment)
            
            # 相互相関係数を計算
            if len(discharge_diff) > 0 and np.std(discharge_diff) > 0 and np.std(water_diff) > 0:
                correlation = np.corrcoef(discharge_diff, water_diff)[0, 1]
                if np.isnan(correlation):
                    return 0
                return abs(correlation)  # 絶対値を使用
            else:
                return 0
            
        except Exception as e:
            return 0
    
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
    def visualize_improved_stable_analysis(self):
        """改善された安定期間分析結果の視覚化（Figure 5）"""
        print("\n=== 改善された安定期間分析の視覚化 (Figure 5) ===")
        
        # 必要なデータの確認
        if not hasattr(self, 'df_processed') or 'detailed_period_type' not in self.df_processed.columns:
            print("詳細な期間分類が実行されていません。")
            return None
        
        if not hasattr(self, 'detailed_stable_results'):
            print("詳細な安定期間分析が実行されていません。")
            return None
        
        # Figure 5: 改善された安定期間分析
        fig5 = plt.figure(figsize=(20, 16))
        
        # 1. 詳細な期間分類の時系列表示（上部）
        ax1 = plt.subplot(4, 2, (1, 2))
        
        time_hours = np.arange(len(self.df_processed)) / 6
        discharge = self.df_processed['ダム_全放流量']
        water_level = self.df_processed['水位_水位']
        detailed_types = self.df_processed['detailed_period_type']
        
        # 期間タイプごとの色分け
        type_colors = {
            'pre_stable': 'darkgreen',
            'post_stable': 'darkblue',
            'event': 'red',
            'low_activity': 'lightgray',
            'variable': 'lightyellow'
        }
        
        # 背景色を設定
        for period_type, color in type_colors.items():
            mask = detailed_types == period_type
            if mask.any():
                # 連続する領域を見つける
                diff = np.diff(np.concatenate([[False], mask.values, [False]]).astype(int))
                starts = np.where(diff == 1)[0]
                ends = np.where(diff == -1)[0]
                
                for start, end in zip(starts, ends):
                    if start < len(time_hours):
                        ax1.axvspan(time_hours[start], time_hours[min(end, len(time_hours)-1)], 
                                   alpha=0.3, color=color, 
                                   label=period_type if start == starts[0] else '')
        
        # 放流量と水位のプロット
        ax1_twin = ax1.twinx()
        ax1.plot(time_hours, discharge, 'b-', linewidth=0.8, alpha=0.8, label='放流量')
        ax1_twin.plot(time_hours, water_level, 'r-', linewidth=0.8, alpha=0.8, label='水位')
        
        ax1.set_xlabel('時間 (hours)')
        ax1.set_ylabel('放流量 (m³/s)', color='b')
        ax1_twin.set_ylabel('水位 (m)', color='r')
        ax1.set_title('詳細な期間分類（緑:pre_stable, 青:post_stable, 赤:event, 灰:low_activity, 黄:variable）', 
                     fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, min(2000, len(time_hours)))  # 最初の2000時間を表示
        
        # 2. 期間タイプ別の統計（左中）
        ax2 = plt.subplot(4, 2, 3)
        
        type_counts = detailed_types.value_counts()
        colors = [type_colors.get(t, 'gray') for t in type_counts.index]
        
        bars = ax2.bar(range(len(type_counts)), type_counts.values, color=colors, edgecolor='black')
        ax2.set_xticks(range(len(type_counts)))
        ax2.set_xticklabels(type_counts.index, rotation=45, ha='right')
        ax2.set_ylabel('データポイント数')
        ax2.set_title('期間タイプ別のデータ分布', fontsize=12)
        
        # パーセンテージを表示
        for i, (bar, count) in enumerate(zip(bars, type_counts.values)):
            percentage = count / len(detailed_types) * 100
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
                    f'{percentage:.1f}%', ha='center', va='bottom', fontsize=8)
        
        # 3. 高放流量域での期間分布（右中）
        ax3 = plt.subplot(4, 2, 4)
        
        high_discharge_mask = discharge >= 300
        high_types = detailed_types[high_discharge_mask]
        
        if len(high_types) > 0:
            high_counts = high_types.value_counts()
            colors = [type_colors.get(t, 'gray') for t in high_counts.index]
            
            bars = ax3.bar(range(len(high_counts)), high_counts.values, color=colors, edgecolor='black')
            ax3.set_xticks(range(len(high_counts)))
            ax3.set_xticklabels(high_counts.index, rotation=45, ha='right')
            ax3.set_ylabel('データポイント数')
            ax3.set_title('高放流量域（≥300 m³/s）での期間分布', fontsize=12)
            
            for i, (bar, count) in enumerate(zip(bars, high_counts.values)):
                percentage = count / len(high_types) * 100
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                        f'{percentage:.1f}%', ha='center', va='bottom', fontsize=8)
        else:
            ax3.text(0.5, 0.5, '高放流量域のデータなし', ha='center', va='center', transform=ax3.transAxes)
        
        # 4. イベント前後の安定期間の遅延時間（左下）
        ax4 = plt.subplot(4, 2, 5)
        
        delay_data = []
        labels = []
        colors_list = []
        
        for period_type in ['pre_stable', 'post_stable', 'low_activity']:
            if period_type in self.detailed_stable_results:
                df = self.detailed_stable_results[period_type]
                if len(df) > 0:
                    delay_data.append(df['delay_minutes'].values)
                    labels.append(f'{period_type}\n(n={len(df)})')
                    colors_list.append(type_colors[period_type])
        
        if delay_data:
            bp = ax4.boxplot(delay_data, labels=labels, patch_artist=True)
            for patch, color in zip(bp['boxes'], colors_list):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
            
            ax4.set_ylabel('遅延時間 (分)')
            ax4.set_title('期間タイプ別の遅延時間分布', fontsize=12)
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'データなし', ha='center', va='center', transform=ax4.transAxes)
        
        # 5. 相関係数の比較（右下）
        ax5 = plt.subplot(4, 2, 6)
        
        corr_means = []
        corr_stds = []
        labels = []
        colors_list = []
        
        for period_type in ['pre_stable', 'post_stable', 'low_activity']:
            if period_type in self.detailed_stable_results:
                df = self.detailed_stable_results[period_type]
                if len(df) > 0:
                    corr_means.append(df['correlation'].mean())
                    corr_stds.append(df['correlation'].std())
                    labels.append(period_type)
                    colors_list.append(type_colors[period_type])
        
        if corr_means:
            x_pos = np.arange(len(labels))
            bars = ax5.bar(x_pos, corr_means, yerr=corr_stds, color=colors_list, 
                           edgecolor='black', capsize=10, alpha=0.7)
            
            ax5.set_xticks(x_pos)
            ax5.set_xticklabels(labels)
            ax5.set_ylabel('平均相関係数')
            ax5.set_title('期間タイプ別の相関係数', fontsize=12)
            ax5.set_ylim(0, 1)
            ax5.grid(True, alpha=0.3)
            
            # 値を表示
            for bar, mean in zip(bars, corr_means):
                ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'{mean:.3f}', ha='center', va='bottom')
        
        # 6. 放流量範囲別の遅延時間比較（下部）
        ax6 = plt.subplot(4, 2, (7, 8))
        
        # 各期間タイプの放流量範囲別データを集計
        all_ranges = set()
        range_data = {}
        
        for period_type in ['pre_stable', 'post_stable', 'low_activity']:
            if period_type in self.detailed_stable_results:
                df = self.detailed_stable_results[period_type]
                if len(df) > 0:
                    grouped = df.groupby('discharge_range')['delay_minutes'].mean()
                    range_data[period_type] = grouped
                    all_ranges.update(grouped.index)
        
        if all_ranges:
            sorted_ranges = sorted(all_ranges, key=lambda x: int(x.split('-')[0]))
            x_pos = np.arange(len(sorted_ranges))
            width = 0.25
            
            for i, (period_type, data) in enumerate(range_data.items()):
                values = [data.get(r, np.nan) for r in sorted_ranges]
                offset = (i - len(range_data)/2 + 0.5) * width
                bars = ax6.bar(x_pos + offset, values, width, 
                              label=period_type, color=type_colors[period_type], alpha=0.7)
            
            ax6.set_xticks(x_pos)
            ax6.set_xticklabels(sorted_ranges, rotation=45, ha='right')
            ax6.set_xlabel('放流量範囲 (m³/s)')
            ax6.set_ylabel('平均遅延時間 (分)')
            ax6.set_title('放流量範囲別・期間タイプ別の遅延時間', fontsize=12)
            ax6.legend()
            ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存
        output_path = f"figures/figure5_improved_stable_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Figure 5 を保存しました: {output_path}")
        
        plt.show()
        
        return fig5

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
        
        # 5.5. イベント分離分析
        analyzer.event_separation_analysis()
        
        # 6. 非線形モデル構築
        analyzer.fit_nonlinear_model()
        
        # 7. 安定・変動期間の分類と分析
        analyzer.classify_stable_variable_periods(window_hours=2, cv_threshold=0.05)
        analyzer.classify_stable_variable_periods_detailed()
        analyzer.analyze_stable_periods()
        analyzer.analyze_stable_periods_detailed()
        analyzer.analyze_variable_periods()
        
        # 8. 可視化
        print("\n=== 可視化処理 ===")
        
        # Figure 5: 安定期間の分析（水位と遅延時間の関係を含む）
        from create_figure5_with_water_level import UpdatedFigure5Analysis
        figure5_analyzer = UpdatedFigure5Analysis()
        # 既に読み込まれたデータを共有
        figure5_analyzer.df = analyzer.df
        figure5_analyzer.df_processed = analyzer.df_processed
        figure5_analyzer.stable_periods = analyzer.stable_periods
        figure5_analyzer.variable_periods = analyzer.variable_periods
        figure5_analyzer.detailed_stable_results = analyzer.detailed_stable_results
        # Figure 5の生成
        figure5_analyzer.unify_stable_periods()
        figure5_analyzer.visualize_updated_figure5()
        
        # Figure 6: 妥当な遅延時間範囲の分析
        from create_figure6_valid_delay_periods import Figure6ValidDelayAnalysis
        figure6_analyzer = Figure6ValidDelayAnalysis()
        # 既に読み込まれたデータを共有
        figure6_analyzer.df = analyzer.df
        figure6_analyzer.df_processed = analyzer.df_processed
        figure6_analyzer.stable_periods = analyzer.stable_periods
        figure6_analyzer.variable_periods = analyzer.variable_periods
        figure6_analyzer.detailed_stable_results = analyzer.detailed_stable_results
        # Figure 6の生成
        figure6_analyzer.create_figure6_valid_delays()
        
        # Figure 7: 変動期間の分析
        from create_figure7_variable_periods import Figure7VariablePeriodsAnalysis
        figure7_analyzer = Figure7VariablePeriodsAnalysis()
        # 既に読み込まれたデータを共有
        figure7_analyzer.df = analyzer.df
        figure7_analyzer.df_processed = analyzer.df_processed
        figure7_analyzer.stable_periods = analyzer.stable_periods
        figure7_analyzer.variable_periods = analyzer.variable_periods
        figure7_analyzer.variable_analysis_results = analyzer.variable_analysis_results
        # Figure 7の生成
        figure7_analyzer.visualize_figure7_variable_periods()
        
        # Figure 8: 生データ分析
        from create_figure8_raw_data import Figure8RawDataAnalysis
        figure8_analyzer = Figure8RawDataAnalysis()
        # 既に読み込まれたデータを共有
        figure8_analyzer.df = analyzer.df
        figure8_analyzer.df_processed = analyzer.df_processed
        figure8_analyzer.stable_periods = analyzer.stable_periods
        figure8_analyzer.variable_periods = analyzer.variable_periods
        figure8_analyzer.variable_analysis_results = analyzer.variable_analysis_results
        # Figure 8の生成
        figure8_analyzer.visualize_figure8_raw_data()
        
        # Figure 9: 変動期間グリッド表示
        from create_figure9_variable_periods_grid import Figure9VariablePeriodsGrid
        figure9_analyzer = Figure9VariablePeriodsGrid()
        # 既に読み込まれたデータを共有
        figure9_analyzer.df = analyzer.df
        figure9_analyzer.df_processed = analyzer.df_processed
        figure9_analyzer.stable_periods = analyzer.stable_periods
        figure9_analyzer.variable_periods = analyzer.variable_periods
        figure9_analyzer.variable_analysis_results = analyzer.variable_analysis_results
        # Figure 9の生成
        figure9_analyzer.visualize_figure9_variable_periods_grid()
        
        # Figure 10: ピーク・ボトム対応分析
        from create_figure10_peak_bottom_analysis import Figure10PeakBottomAnalysis
        figure10_analyzer = Figure10PeakBottomAnalysis()
        # 既に読み込まれたデータを共有
        figure10_analyzer.df = analyzer.df
        figure10_analyzer.df_processed = analyzer.df_processed
        figure10_analyzer.stable_periods = analyzer.stable_periods
        figure10_analyzer.variable_periods = analyzer.variable_periods
        figure10_analyzer.variable_analysis_results = analyzer.variable_analysis_results
        # Figure 10の生成
        figure10_analyzer.create_figure10_peak_bottom_analysis()
        
        # Figure 11: ヒステリシス分析（固定時間窓版）
        from create_figure11_hysteresis_analysis_fixed_window import Figure11HysteresisAnalysisFixedWindow
        figure11_analyzer = Figure11HysteresisAnalysisFixedWindow()
        # 既に読み込まれたデータを共有
        figure11_analyzer.df = analyzer.df
        figure11_analyzer.df_processed = analyzer.df_processed
        figure11_analyzer.stable_periods = analyzer.stable_periods
        figure11_analyzer.variable_periods = analyzer.variable_periods
        # Figure 11の生成
        figure11_analyzer.create_figure11_hysteresis_analysis_fixed_window()
        
        # Figure 12: 変化量の関係性分析
        from create_figure12_change_amount_analysis_revised import Figure12ChangeAmountAnalysisRevised
        figure12_analyzer = Figure12ChangeAmountAnalysisRevised()
        # 既に読み込まれたデータを共有
        figure12_analyzer.df = analyzer.df
        figure12_analyzer.df_processed = analyzer.df_processed
        figure12_analyzer.stable_periods = analyzer.stable_periods
        figure12_analyzer.variable_periods = analyzer.variable_periods
        # Figure 12の生成
        figure12_analyzer.create_figure12_revised()
        
        # Figure 13: 降雨強度-放流量関係分析
        from create_figure13_rainfall_discharge_analysis import Figure13RainfallDischargeAnalysis
        figure13_analyzer = Figure13RainfallDischargeAnalysis(data_file='統合データ_水位ダム_20250730_205325.csv')
        # Figure 13は独自のデータファイルを使用するため、別途読み込み
        figure13_analyzer.load_and_preprocess_data()
        figure13_analyzer.analyze_cross_correlation()
        figure13_analyzer.analyze_response_patterns()
        figure13_analyzer.build_prediction_models()
        figure13_analyzer.evaluate_predictions()
        figure13_analyzer.create_figure13()
        figure13_analyzer.save_analysis_results()
        
        # 9. 結果保存
        analyzer.save_results()
        
        print("\n分析完了！")
        
    except Exception as e:
        print(f"\nエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()