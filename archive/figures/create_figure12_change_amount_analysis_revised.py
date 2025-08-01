#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 12: 変化量の関係性分析（増加期・減少期における応答特性）- 改訂版
水位予測モデル構築のための変化率分析
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from datetime import datetime
from scipy import stats, signal

from analyze_water_level_delay import RiverDelayAnalysis

class Figure12ChangeAmountAnalysisRevised(RiverDelayAnalysis):
    # 親クラスのload_data()メソッドを使用（ダイアログ機能付き）
    
    def analyze_change_rate_relationships(self):
        """変化率の関係性分析（改訂版）"""
        print("\n=== 変化率の関係性分析（改訂版） ===")
        
        if not hasattr(self, 'df_processed'):
            print("データが処理されていません。")
            return None
        
        # 全データで分析（変動期間に限定しない）
        discharge = self.df_processed['ダム_全放流量'].values
        water_level = self.df_processed['水位_水位'].values
        
        # フィルタリング条件: 水位≥3m かつ 放流量≥150m³/s
        valid_mask = (water_level >= 3.0) & (discharge >= 150.0) & ~np.isnan(discharge) & ~np.isnan(water_level)
        
        change_rate_results = []
        
        # 変化率の計算（10分間隔、30分間隔、60分間隔）
        intervals = {'10min': 1, '30min': 3, '60min': 6}
        
        for interval_name, interval_points in intervals.items():
            for i in range(len(discharge) - interval_points):
                if valid_mask[i] and valid_mask[i+interval_points]:
                    # 変化率計算
                    dQ_dt = (discharge[i+interval_points] - discharge[i]) / (interval_points * 10)  # m³/s/min
                    dH_dt = (water_level[i+interval_points] - water_level[i]) / (interval_points * 10)  # m/min
                    
                    # 変化の方向
                    if abs(dQ_dt) > 0.5:  # 有意な変化のみ（0.5 m³/s/min以上）
                        direction = 'increase' if dQ_dt > 0 else 'decrease'
                        
                        # 水位レベル
                        avg_H = (water_level[i] + water_level[i+interval_points]) / 2
                        if avg_H < 3.5:
                            water_level_category = '3.0-3.5m'
                        elif avg_H < 4.0:
                            water_level_category = '3.5-4.0m'
                        elif avg_H < 4.5:
                            water_level_category = '4.0-4.5m'
                        elif avg_H < 5.0:
                            water_level_category = '4.5-5.0m'
                        else:
                            water_level_category = '5.0m以上'
                        
                        # 放流量レベル
                        avg_Q = (discharge[i] + discharge[i+interval_points]) / 2
                        
                        change_rate_results.append({
                            'time_idx': i,
                            'interval': interval_name,
                            'direction': direction,
                            'dQ_dt': dQ_dt,
                            'dH_dt': dH_dt,
                            'avg_Q': avg_Q,
                            'avg_H': avg_H,
                            'water_level_category': water_level_category,
                            'response_rate': dH_dt / dQ_dt if abs(dQ_dt) > 0 else np.nan
                        })
        
        self.change_rate_results = pd.DataFrame(change_rate_results)
        
        print(f"分析された変化率イベント数: {len(change_rate_results)}件")
        if len(change_rate_results) > 0:
            print(f"増加イベント: {(self.change_rate_results['direction'] == 'increase').sum()}件")
            print(f"減少イベント: {(self.change_rate_results['direction'] == 'decrease').sum()}件")
        
        return self.change_rate_results
    
    def analyze_time_lagged_response(self):
        """時間遅れを考慮した変化率分析"""
        print("\n=== 時間遅れを考慮した変化率分析 ===")
        
        discharge = self.df_processed['ダム_全放流量'].values
        water_level = self.df_processed['水位_水位'].values
        
        # フィルタリング
        valid_mask = (water_level >= 3.0) & (discharge >= 150.0) & ~np.isnan(discharge) & ~np.isnan(water_level)
        
        # 遅延時間の範囲（0～120分、10分刻み）
        lag_results = []
        lags = range(0, 13)  # 0～120分（10分刻み）
        
        # サンプリング（計算量削減のため）
        sample_indices = np.where(valid_mask)[0][::10]  # 10点ごとにサンプリング
        
        for lag in lags:
            correlations_inc = []
            correlations_dec = []
            
            for i in sample_indices:
                if i >= 6 and i < len(discharge) - 6 - lag:
                    # 放流量変化率（現在）
                    dQ_dt = (discharge[i+6] - discharge[i-6]) / 120  # 前後1時間の変化率
                    
                    # 水位変化率（lag後）
                    if i+lag < len(water_level) - 6:
                        dH_dt = (water_level[i+lag+6] - water_level[i+lag-6]) / 120
                        
                        if abs(dQ_dt) > 0.5:  # 有意な変化
                            if dQ_dt > 0:
                                correlations_inc.append((dQ_dt, dH_dt))
                            else:
                                correlations_dec.append((dQ_dt, dH_dt))
            
            # 相関係数計算
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
                'correlation_increase': r_inc,
                'correlation_decrease': r_dec,
                'n_increase': len(correlations_inc),
                'n_decrease': len(correlations_dec)
            })
        
        self.lag_results = pd.DataFrame(lag_results)
        return self.lag_results
    
    def create_figure12_revised(self):
        """Figure 12: 変化率関係性の可視化（改訂版）"""
        print("\n=== Figure 12: 変化率の関係性分析（改訂版） ===")
        
        # 分析実行
        if not hasattr(self, 'change_rate_results'):
            self.analyze_change_rate_relationships()
        
        if not hasattr(self, 'lag_results'):
            self.analyze_time_lagged_response()
        
        if len(self.change_rate_results) == 0:
            print("分析可能なデータがありません。")
            return None
        
        # レイアウトを3行3列に設定
        fig, axes = plt.subplots(3, 3, figsize=(20, 12), num='Figure 12 (Revised)')
        
        # 1. 変化率の直接的な関係（増加期・減少期別）（1行目左）
        ax1 = axes[0, 0]
        
        # 60分間隔のデータのみ使用
        data_60min = self.change_rate_results[self.change_rate_results['interval'] == '60min']
        increase_mask = data_60min['direction'] == 'increase'
        decrease_mask = data_60min['direction'] == 'decrease'
        
        # 増加期
        if increase_mask.sum() > 0:
            ax1.scatter(data_60min.loc[increase_mask, 'dQ_dt'],
                       data_60min.loc[increase_mask, 'dH_dt'],
                       c='red', alpha=0.3, s=10, label=f'増加期 (n={increase_mask.sum()})')
            
            # 回帰直線
            X_inc = data_60min.loc[increase_mask, 'dQ_dt'].values
            y_inc = data_60min.loc[increase_mask, 'dH_dt'].values
            if len(X_inc) > 10:
                coef_inc = np.polyfit(X_inc, y_inc, 1)
                x_plot = np.linspace(0, X_inc.max(), 100)
                ax1.plot(x_plot, np.polyval(coef_inc, x_plot), 'r--', linewidth=2,
                        label=f'増加期: dH/dt = {coef_inc[0]:.4f}·dQ/dt')
        
        # 減少期
        if decrease_mask.sum() > 0:
            ax1.scatter(data_60min.loc[decrease_mask, 'dQ_dt'],
                       data_60min.loc[decrease_mask, 'dH_dt'],
                       c='blue', alpha=0.3, s=10, label=f'減少期 (n={decrease_mask.sum()})')
            
            # 回帰直線
            X_dec = data_60min.loc[decrease_mask, 'dQ_dt'].values
            y_dec = data_60min.loc[decrease_mask, 'dH_dt'].values
            if len(X_dec) > 10:
                coef_dec = np.polyfit(X_dec, y_dec, 1)
                x_plot = np.linspace(X_dec.min(), 0, 100)
                ax1.plot(x_plot, np.polyval(coef_dec, x_plot), 'b--', linewidth=2,
                        label=f'減少期: dH/dt = {coef_dec[0]:.4f}·dQ/dt')
        
        ax1.set_xlabel('放流量変化率 dQ/dt (m³/s/min)')
        ax1.set_ylabel('水位変化率 dH/dt (m/min)')
        ax1.set_title('放流量変化率と水位変化率の関係（60分間隔）')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='k', linewidth=0.5)
        ax1.axvline(x=0, color='k', linewidth=0.5)
        
        # 2. 時間遅れ相関分析（1行目中央）
        ax2 = axes[0, 1]
        
        if len(self.lag_results) > 0:
            ax2.plot(self.lag_results['lag_minutes'], 
                    self.lag_results['correlation_increase'],
                    'ro-', linewidth=2, markersize=8, label='増加期')
            ax2.plot(self.lag_results['lag_minutes'], 
                    self.lag_results['correlation_decrease'],
                    'bo-', linewidth=2, markersize=8, label='減少期')
            
            # 最大相関の位置をマーク
            max_inc_idx = self.lag_results['correlation_increase'].idxmax()
            max_dec_idx = np.abs(self.lag_results['correlation_decrease']).idxmax()
            
            if not pd.isna(max_inc_idx):
                ax2.axvline(self.lag_results.loc[max_inc_idx, 'lag_minutes'], 
                          color='red', linestyle=':', alpha=0.5)
                ax2.text(self.lag_results.loc[max_inc_idx, 'lag_minutes'], 0.9,
                        f"{self.lag_results.loc[max_inc_idx, 'lag_minutes']}分",
                        color='red', ha='center')
            
            if not pd.isna(max_dec_idx):
                ax2.axvline(self.lag_results.loc[max_dec_idx, 'lag_minutes'], 
                          color='blue', linestyle=':', alpha=0.5)
                ax2.text(self.lag_results.loc[max_dec_idx, 'lag_minutes'], -0.9,
                        f"{self.lag_results.loc[max_dec_idx, 'lag_minutes']}分",
                        color='blue', ha='center')
            
            ax2.set_xlabel('遅延時間 (分)')
            ax2.set_ylabel('相関係数')
            ax2.set_title('変化率の時間遅れ相関分析')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim(-1, 1)
        
        # 3. 水位レベル別の変化率応答（1行目右）
        ax3 = axes[0, 2]
        
        # 水位カテゴリ別の応答率
        water_levels = ['3.0-3.5m', '3.5-4.0m', '4.0-4.5m', '4.5-5.0m', '5.0m以上']
        colors = ['lightblue', 'skyblue', 'orange', 'darkorange', 'red']
        
        response_by_level = []
        for level in water_levels:
            level_data = data_60min[data_60min['water_level_category'] == level]
            if len(level_data) > 0:
                inc_data = level_data[level_data['direction'] == 'increase']['response_rate'].dropna()
                dec_data = level_data[level_data['direction'] == 'decrease']['response_rate'].dropna()
                
                response_by_level.append({
                    'level': level,
                    'inc_mean': inc_data.mean() if len(inc_data) > 0 else np.nan,
                    'inc_std': inc_data.std() if len(inc_data) > 0 else np.nan,
                    'dec_mean': dec_data.mean() if len(dec_data) > 0 else np.nan,
                    'dec_std': dec_data.std() if len(dec_data) > 0 else np.nan,
                    'n_inc': len(inc_data),
                    'n_dec': len(dec_data)
                })
        
        if response_by_level:
            df_level = pd.DataFrame(response_by_level)
            x_pos = np.arange(len(water_levels))
            width = 0.35
            
            # 増加期
            inc_means = [r['inc_mean'] for r in response_by_level]
            inc_stds = [r['inc_std'] for r in response_by_level]
            bars1 = ax3.bar(x_pos - width/2, inc_means, width, yerr=inc_stds,
                           label='増加期', color='red', alpha=0.7, capsize=5)
            
            # 減少期（絶対値）
            dec_means = [abs(r['dec_mean']) if not np.isnan(r['dec_mean']) else 0 for r in response_by_level]
            dec_stds = [r['dec_std'] for r in response_by_level]
            bars2 = ax3.bar(x_pos + width/2, dec_means, width, yerr=dec_stds,
                           label='減少期', color='blue', alpha=0.7, capsize=5)
            
            # データ数を表示
            for i, (r, bar1, bar2) in enumerate(zip(response_by_level, bars1, bars2)):
                if r['n_inc'] > 0:
                    ax3.text(bar1.get_x() + bar1.get_width()/2, bar1.get_height() + 0.0001,
                           f"n={r['n_inc']}", ha='center', va='bottom', fontsize=8)
                if r['n_dec'] > 0:
                    ax3.text(bar2.get_x() + bar2.get_width()/2, bar2.get_height() + 0.0001,
                           f"n={r['n_dec']}", ha='center', va='bottom', fontsize=8)
            
            ax3.set_xticks(x_pos)
            ax3.set_xticklabels(water_levels, rotation=45, ha='right')
            ax3.set_xlabel('水位レベル')
            ax3.set_ylabel('応答率の絶対値 |dH/dt / dQ/dt|')
            ax3.set_title('水位レベル別の変化率応答')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 4. 変化規模別の応答（2行目左）
        ax4 = axes[1, 0]
        
        # 放流量変化率の規模で分類
        data_60min['dQ_magnitude'] = pd.cut(data_60min['dQ_dt'].abs(), 
                                           bins=[0, 2, 5, 10, 100],
                                           labels=['小(0-2)', '中(2-5)', '大(5-10)', '特大(>10)'])
        
        magnitude_stats = []
        for mag in ['小(0-2)', '中(2-5)', '大(5-10)', '特大(>10)']:
            mag_data = data_60min[data_60min['dQ_magnitude'] == mag]
            if len(mag_data) > 0:
                inc_data = mag_data[mag_data['direction'] == 'increase']['response_rate'].dropna()
                dec_data = mag_data[mag_data['direction'] == 'decrease']['response_rate'].dropna()
                
                magnitude_stats.append({
                    'magnitude': mag,
                    'inc_mean': inc_data.mean() if len(inc_data) > 0 else np.nan,
                    'dec_mean': abs(dec_data.mean()) if len(dec_data) > 0 else np.nan,
                    'n_total': len(mag_data)
                })
        
        if magnitude_stats:
            df_mag = pd.DataFrame(magnitude_stats)
            x_pos = np.arange(len(df_mag))
            width = 0.35
            
            ax4.bar(x_pos - width/2, df_mag['inc_mean'], width,
                   label='増加期', color='red', alpha=0.7)
            ax4.bar(x_pos + width/2, df_mag['dec_mean'], width,
                   label='減少期', color='blue', alpha=0.7)
            
            ax4.set_xticks(x_pos)
            ax4.set_xticklabels(df_mag['magnitude'])
            ax4.set_xlabel('変化率の大きさ (m³/s/min)')
            ax4.set_ylabel('平均応答率')
            ax4.set_title('変化規模別の応答率')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        # 5. 初期放流量と応答率（2行目中央）
        ax5 = axes[1, 1]
        
        # 放流量範囲で色分け
        q_ranges = [(150, 300), (300, 500), (500, 800), (800, 1200)]
        colors_q = ['lightgreen', 'yellow', 'orange', 'red']
        
        for (q_min, q_max), color in zip(q_ranges, colors_q):
            mask = (data_60min['avg_Q'] >= q_min) & (data_60min['avg_Q'] < q_max)
            if mask.sum() > 0:
                ax5.scatter(data_60min.loc[mask, 'avg_Q'],
                          data_60min.loc[mask, 'response_rate'],
                          c=color, alpha=0.5, s=20, 
                          label=f'{q_min}-{q_max} m³/s')
        
        ax5.set_xlabel('平均放流量 (m³/s)')
        ax5.set_ylabel('応答率 dH/dt / dQ/dt')
        ax5.set_title('放流量レベルと応答率の関係')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        ax5.set_ylim(-0.01, 0.01)
        ax5.axhline(y=0, color='k', linewidth=0.5)
        
        # 6. 非線形性の評価（2行目右）
        ax6 = axes[1, 2]
        
        # 変化率の大きさと応答率の関係
        valid_mask = ~np.isnan(data_60min['response_rate'])
        
        if valid_mask.sum() > 0:
            # 増加期
            inc_mask = (data_60min['direction'] == 'increase') & valid_mask
            if inc_mask.sum() > 0:
                ax6.scatter(data_60min.loc[inc_mask, 'dQ_dt'].abs(),
                          data_60min.loc[inc_mask, 'response_rate'].abs(),
                          c='red', alpha=0.3, s=20, label='増加期')
            
            # 減少期
            dec_mask = (data_60min['direction'] == 'decrease') & valid_mask
            if dec_mask.sum() > 0:
                ax6.scatter(data_60min.loc[dec_mask, 'dQ_dt'].abs(),
                          data_60min.loc[dec_mask, 'response_rate'].abs(),
                          c='blue', alpha=0.3, s=20, label='減少期')
            
            ax6.set_xlabel('放流量変化率の大きさ |dQ/dt| (m³/s/min)')
            ax6.set_ylabel('応答率の絶対値 |dH/dt / dQ/dt|')
            ax6.set_title('変化率の大きさと応答率（非線形性評価）')
            ax6.legend()
            ax6.grid(True, alpha=0.3)
            ax6.set_xlim(0, 20)
            ax6.set_ylim(0, 0.01)
        
        # 7. 予測モデル用応答関数（増加期）（3行目左）
        ax7 = axes[2, 0]
        
        # 増加期の統合モデル
        inc_data = data_60min[data_60min['direction'] == 'increase']
        if len(inc_data) > 0:
            # 放流量ビンごとの平均応答
            q_bins = np.arange(150, 1200, 100)
            mean_responses = []
            std_responses = []
            
            for i in range(len(q_bins)-1):
                bin_mask = (inc_data['avg_Q'] >= q_bins[i]) & (inc_data['avg_Q'] < q_bins[i+1])
                if bin_mask.sum() > 5:
                    responses = inc_data.loc[bin_mask, 'response_rate'].dropna()
                    mean_responses.append(responses.mean())
                    std_responses.append(responses.std())
                else:
                    mean_responses.append(np.nan)
                    std_responses.append(np.nan)
            
            q_centers = (q_bins[:-1] + q_bins[1:]) / 2
            valid = ~np.isnan(mean_responses)
            
            ax7.errorbar(q_centers[valid], np.array(mean_responses)[valid],
                       yerr=np.array(std_responses)[valid],
                       fmt='ro-', capsize=5, label='平均±標準偏差')
            
            # フィッティング曲線
            if valid.sum() > 3:
                # 対数関数でフィット
                popt = np.polyfit(np.log(q_centers[valid]), np.array(mean_responses)[valid], 1)
                q_fit = np.linspace(150, 1200, 100)
                y_fit = popt[0] * np.log(q_fit) + popt[1]
                ax7.plot(q_fit, y_fit, 'r--', linewidth=2,
                        label=f'y = {popt[0]:.5f}·ln(Q) + {popt[1]:.5f}')
            
            ax7.set_xlabel('放流量 (m³/s)')
            ax7.set_ylabel('応答率 dH/dt / dQ/dt')
            ax7.set_title('増加期の応答関数（予測モデル用）')
            ax7.legend()
            ax7.grid(True, alpha=0.3)
        
        # 8. 予測モデル用応答関数（減少期）（3行目中央）
        ax8 = axes[2, 1]
        
        # 減少期の統合モデル
        dec_data = data_60min[data_60min['direction'] == 'decrease']
        if len(dec_data) > 0:
            # 放流量ビンごとの平均応答
            mean_responses = []
            std_responses = []
            
            for i in range(len(q_bins)-1):
                bin_mask = (dec_data['avg_Q'] >= q_bins[i]) & (dec_data['avg_Q'] < q_bins[i+1])
                if bin_mask.sum() > 5:
                    responses = dec_data.loc[bin_mask, 'response_rate'].dropna()
                    mean_responses.append(responses.mean())
                    std_responses.append(responses.std())
                else:
                    mean_responses.append(np.nan)
                    std_responses.append(np.nan)
            
            valid = ~np.isnan(mean_responses)
            
            ax8.errorbar(q_centers[valid], np.array(mean_responses)[valid],
                       yerr=np.array(std_responses)[valid],
                       fmt='bo-', capsize=5, label='平均±標準偏差')
            
            # フィッティング曲線
            if valid.sum() > 3:
                popt = np.polyfit(np.log(q_centers[valid]), np.array(mean_responses)[valid], 1)
                q_fit = np.linspace(150, 1200, 100)
                y_fit = popt[0] * np.log(q_fit) + popt[1]
                ax8.plot(q_fit, y_fit, 'b--', linewidth=2,
                        label=f'y = {popt[0]:.5f}·ln(Q) + {popt[1]:.5f}')
            
            ax8.set_xlabel('放流量 (m³/s)')
            ax8.set_ylabel('応答率 dH/dt / dQ/dt')
            ax8.set_title('減少期の応答関数（予測モデル用）')
            ax8.legend()
            ax8.grid(True, alpha=0.3)
            ax8.axhline(y=0, color='k', linewidth=0.5)
        
        # 9. 累積変化量の関係（3行目右）
        ax9 = axes[2, 2]
        
        # 変動期間での累積変化
        if hasattr(self, 'variable_periods'):
            cumulative_results = []
            
            for period_idx, (start, end) in enumerate(self.variable_periods[:20]):  # 最初の20期間
                if end - start > 12:
                    period_data = self.df_processed.iloc[start:end]
                    
                    # フィルタリング
                    mask = (period_data['水位_水位'] >= 3.0) & (period_data['ダム_全放流量'] >= 150.0)
                    if mask.sum() > 6:
                        Q = period_data.loc[mask, 'ダム_全放流量'].values
                        H = period_data.loc[mask, '水位_水位'].values
                        
                        # 累積変化
                        cum_dQ = np.cumsum(np.diff(Q))
                        cum_dH = np.cumsum(np.diff(H))
                        
                        if len(cum_dQ) > 0:
                            # 全体の方向
                            direction = 'increase' if cum_dQ[-1] > 0 else 'decrease'
                            color = 'red' if direction == 'increase' else 'blue'
                            
                            ax9.plot(cum_dQ, cum_dH, color=color, alpha=0.5, linewidth=1)
                            ax9.scatter(cum_dQ[-1], cum_dH[-1], color=color, s=50, 
                                      edgecolor='black', zorder=5)
        
        ax9.set_xlabel('累積放流量変化 ΣΔQ (m³/s)')
        ax9.set_ylabel('累積水位変化 ΣΔH (m)')
        ax9.set_title('変動期間での累積変化量の関係')
        ax9.grid(True, alpha=0.3)
        ax9.axhline(y=0, color='k', linewidth=0.5)
        ax9.axvline(x=0, color='k', linewidth=0.5)
        
        plt.tight_layout()
        
        # 統計サマリーを表示
        print("\n=== 変化率分析サマリー（改訂版） ===")
        print(f"総イベント数: {len(self.change_rate_results)}")
        
        # 最適遅延時間
        if len(self.lag_results) > 0:
            opt_lag_inc = self.lag_results.loc[self.lag_results['correlation_increase'].idxmax(), 'lag_minutes']
            opt_lag_dec = self.lag_results.loc[np.abs(self.lag_results['correlation_decrease']).idxmax(), 'lag_minutes']
            print(f"\n最適遅延時間:")
            print(f"  増加期: {opt_lag_inc}分")
            print(f"  減少期: {opt_lag_dec}分")
        
        # 水位レベル別の応答
        print(f"\n水位レベル別の平均応答率:")
        for r in response_by_level:
            if not np.isnan(r['inc_mean']):
                print(f"  {r['level']}: 増加期 {r['inc_mean']:.5f}, 減少期 {r['dec_mean']:.5f}")
        
        # 保存
        output_path = f"figures/figure12_change_amount_analysis_revised_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nFigure 12（改訂版）を保存しました: {output_path}")
        
        plt.close()
        
        return fig

def main():
    """単独実行時のみ使用"""
    analyzer = Figure12ChangeAmountAnalysisRevised()
    
    # データ読み込みと分析
    analyzer.load_data()
    analyzer.preprocess_data()
    analyzer.classify_stable_variable_periods()
    
    # Figure 12の作成
    analyzer.create_figure12_revised()
    
    print("\nFigure 12（変化率の関係性分析・改訂版）の作成が完了しました。")

if __name__ == "__main__":
    main()