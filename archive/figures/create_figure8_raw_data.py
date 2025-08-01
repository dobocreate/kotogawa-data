#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 8: 期間平均化なしの生データ分析
安定期間と変動期間を統合し、10分間隔のデータで遅延時間を計算
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from datetime import datetime

from analyze_water_level_delay import RiverDelayAnalysis

class Figure8RawDataAnalysis(RiverDelayAnalysis):
    # 親クラスのload_data()メソッドを使用（ダイアログ機能付き）
    
    def calculate_delay_for_each_point(self):
        """各時点での遅延時間を計算（前後2時間の窓）"""
        print("\n=== 各時点での遅延時間計算 ===")
        
        # 結果を格納するリスト
        delay_results = []
        
        # 窓幅（前後2時間 = 12ポイント）
        window_half = 12
        
        # 各時点で遅延時間を計算（計算量削減のため1時間ごとにサンプリング）
        step = 6  # 1時間ごと
        total_points = len(self.df_processed)
        
        for i in range(window_half, total_points - window_half, step):
            # 進捗表示
            if i % 600 == 0:  # 100時間ごと
                print(f"進捗: {i}/{total_points} ({i/total_points*100:.1f}%)")
            
            # 前後2時間のデータを取得
            start_idx = i - window_half
            end_idx = i + window_half + 1
            
            discharge_window = self.df_processed['ダム_全放流量'].iloc[start_idx:end_idx].values
            water_level_window = self.df_processed['水位_水位'].iloc[start_idx:end_idx].values
            
            # 欠測値処理
            if np.any(np.isnan(discharge_window)) or np.any(np.isnan(water_level_window)):
                continue
            
            # 相関分析で遅延時間を計算
            correlations = []
            for lag in range(0, min(13, len(discharge_window))):  # 最大120分の遅延
                if lag > 0:
                    corr = np.corrcoef(discharge_window[:-lag], water_level_window[lag:])[0, 1]
                else:
                    corr = np.corrcoef(discharge_window, water_level_window)[0, 1]
                correlations.append(corr if not np.isnan(corr) else 0)
            
            best_lag = np.argmax(correlations)
            best_corr = correlations[best_lag]
            
            # 放流量変化率を計算（前後30分 = 3ポイント）
            change_window = 3
            if i >= change_window and i < total_points - change_window:
                discharge_before = self.df_processed['ダム_全放流量'].iloc[i - change_window]
                discharge_after = self.df_processed['ダム_全放流量'].iloc[i + change_window]
                discharge_change = discharge_after - discharge_before
                
                if discharge_change > 0:
                    flow_direction = 'increase'
                elif discharge_change < 0:
                    flow_direction = 'decrease'
                else:
                    flow_direction = 'stable'
            else:
                flow_direction = 'unknown'
            
            # 結果を保存
            delay_results.append({
                'index': i,
                'datetime': self.df_processed.iloc[i]['時刻'],
                'discharge': self.df_processed.iloc[i]['ダム_全放流量'],
                'water_level': self.df_processed.iloc[i]['水位_水位'],
                'delay_minutes': best_lag * 10,
                'correlation': best_corr,
                'period_type': self.df_processed.iloc[i]['period_type'],
                'flow_direction': flow_direction
            })
        
        self.raw_delay_results = pd.DataFrame(delay_results)
        print(f"計算されたデータ点数: {len(self.raw_delay_results)}")
        print(f"平均遅延時間: {self.raw_delay_results['delay_minutes'].mean():.1f}分")
        
        # 方向別の統計
        direction_counts = self.raw_delay_results['flow_direction'].value_counts()
        print("方向別データ数:")
        for direction, count in direction_counts.items():
            print(f"  {direction}: {count}")
        
        return self.raw_delay_results
    
    def visualize_figure8_raw_data(self):
        """Figure 8: 生データでの視覚化"""
        print("\n=== Figure 8: 生データの視覚化 ===")
        
        # 遅延時間が計算されていない場合は計算
        if not hasattr(self, 'raw_delay_results'):
            self.calculate_delay_for_each_point()
        
        # レイアウトを3行3列に設定
        fig, axes = plt.subplots(3, 3, figsize=(20, 12), num='Figure 8')
        
        # 色の設定（全て同じ色）
        data_color = 'darkblue'
        
        # 1. 時系列例（固定期間）
        # Figure 7の期間が変わっても、こちらは常に同じ期間を表示
        example_periods = [
            (28649, 28734),  # 流量増加期（固定）
            (28742, 28795),  # 流量減少期（固定）
            (37584, 37719)   # 複雑な変動期（固定）
        ]
        
        period_names = ['流量増加期', '流量減少期', '複雑な変動期']
        
        for i, (start, end) in enumerate(example_periods):
            ax = axes[0, i]
            
            if end < len(self.df_processed):
                display_start = max(0, start - 18)
                display_end = min(end + 18, len(self.df_processed))
                
                time_hours = np.arange(display_end - display_start) / 6
                discharge = self.df_processed['ダム_全放流量'].iloc[display_start:display_end]
                water_level = self.df_processed['水位_水位'].iloc[display_start:display_end]
                
                # 期間の範囲を薄い色で色づけ
                period_start_rel = (start - display_start) / 6
                period_end_rel = (end - display_start) / 6
                ax.axvspan(period_start_rel, period_end_rel, alpha=0.2, color='gray')
                
                ax2 = ax.twinx()
                line1 = ax.plot(time_hours, discharge, 'b-', linewidth=2, label='放流量')
                line2 = ax2.plot(time_hours, water_level, 'r-', linewidth=2, label='水位')
                
                ax.set_xlabel('時間 (hours)')
                ax.set_ylabel('放流量 (m³/s)', color='b')
                ax2.set_ylabel('水位 (m)', color='r')
                ax.set_title(period_names[i], fontsize=12)
                ax.grid(True, alpha=0.3)
                
                if i == 0:
                    lines = line1 + line2
                    labels = [l.get_label() for l in lines]
                    ax.legend(lines, labels, loc='upper left')
            else:
                ax.text(0.5, 0.5, 'データなし', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(period_names[i], fontsize=12)
        
        # 2. 生データでの放流量-水位関係（2段目左）
        ax3 = axes[1, 0]
        
        # 増加期・減少期別に色分け表示
        increase_mask = self.raw_delay_results['flow_direction'] == 'increase'
        decrease_mask = self.raw_delay_results['flow_direction'] == 'decrease'
        other_mask = ~(increase_mask | decrease_mask)
        
        if increase_mask.any():
            ax3.scatter(self.raw_delay_results[increase_mask]['discharge'], 
                       self.raw_delay_results[increase_mask]['water_level'],
                       c='lightcoral', s=5, alpha=0.6, label=f'増加期 (n={increase_mask.sum()})')
        
        if decrease_mask.any():
            ax3.scatter(self.raw_delay_results[decrease_mask]['discharge'], 
                       self.raw_delay_results[decrease_mask]['water_level'],
                       c='lightblue', s=5, alpha=0.6, label=f'減少期 (n={decrease_mask.sum()})')
        
        if other_mask.any():
            ax3.scatter(self.raw_delay_results[other_mask]['discharge'], 
                       self.raw_delay_results[other_mask]['water_level'],
                       c='lightgray', s=5, alpha=0.3, label=f'その他 (n={other_mask.sum()})')
        
        ax3.set_xlabel('放流量 (m³/s)')
        ax3.set_ylabel('水位 (m)')
        ax3.set_title(f'放流量-水位関係（全データ）\n(n={len(self.raw_delay_results)})', fontsize=12)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 3. 放流量範囲別データ数（2段目中央）
        ax4 = axes[1, 1]
        
        range_labels = ['0-150', '150-200', '200-300', '300-400', '400-500', '500+']
        discharge_ranges = [(0, 150), (150, 200), (200, 300), (300, 400), (400, 500), (500, 1000)]
        
        increase_counts = []
        decrease_counts = []
        other_counts = []
        
        for min_d, max_d in discharge_ranges:
            range_data = self.raw_delay_results[
                (self.raw_delay_results['discharge'] >= min_d) & 
                (self.raw_delay_results['discharge'] < max_d)
            ]
            
            inc_count = len(range_data[range_data['flow_direction'] == 'increase'])
            dec_count = len(range_data[range_data['flow_direction'] == 'decrease'])
            other_count = len(range_data[~range_data['flow_direction'].isin(['increase', 'decrease'])])
            
            increase_counts.append(inc_count)
            decrease_counts.append(dec_count)
            other_counts.append(other_count)
        
        x_pos = np.arange(len(range_labels))
        width = 0.25
        
        bars1 = ax4.bar(x_pos - width, increase_counts, width, 
                       label='増加期', color='lightcoral', alpha=0.7, edgecolor='black')
        bars2 = ax4.bar(x_pos, decrease_counts, width,
                       label='減少期', color='lightblue', alpha=0.7, edgecolor='black')
        bars3 = ax4.bar(x_pos + width, other_counts, width,
                       label='その他', color='lightgray', alpha=0.7, edgecolor='black')
        
        # データ数を表示
        for i, (inc, dec, other) in enumerate(zip(increase_counts, decrease_counts, other_counts)):
            if inc > 0:
                ax4.text(i - width, inc + 5, str(inc), ha='center', va='bottom', fontsize=8)
            if dec > 0:
                ax4.text(i, dec + 5, str(dec), ha='center', va='bottom', fontsize=8)
            if other > 0:
                ax4.text(i + width, other + 5, str(other), ha='center', va='bottom', fontsize=8)
        
        ax4.set_xlabel('放流量範囲 (m³/s)')
        ax4.set_ylabel('データ点数')
        ax4.set_title('放流量範囲別のデータ数', fontsize=12)
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(range_labels)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 4. 遅延時間の分布（2段目右）
        ax5 = axes[1, 2]
        
        # 増加期・減少期別のヒストグラム
        increase_delays = self.raw_delay_results[self.raw_delay_results['flow_direction'] == 'increase']['delay_minutes']
        decrease_delays = self.raw_delay_results[self.raw_delay_results['flow_direction'] == 'decrease']['delay_minutes']
        other_delays = self.raw_delay_results[~self.raw_delay_results['flow_direction'].isin(['increase', 'decrease'])]['delay_minutes']
        
        if len(increase_delays) > 0:
            ax5.hist(increase_delays, bins=7, alpha=0.6, color='lightcoral', 
                    edgecolor='black', label=f'増加期 (n={len(increase_delays)})')
        
        if len(decrease_delays) > 0:
            ax5.hist(decrease_delays, bins=7, alpha=0.6, color='lightblue', 
                    edgecolor='black', label=f'減少期 (n={len(decrease_delays)})')
        
        if len(other_delays) > 0:
            ax5.hist(other_delays, bins=7, alpha=0.3, color='lightgray', 
                    edgecolor='black', label=f'その他 (n={len(other_delays)})')
        
        ax5.axvline(x=self.raw_delay_results['delay_minutes'].mean(), 
                   color='red', linestyle='--', linewidth=2,
                   label=f'全体平均: {self.raw_delay_results["delay_minutes"].mean():.1f}分')
        
        ax5.set_xlabel('遅延時間 (分)')
        ax5.set_ylabel('データ点数')
        ax5.set_title('遅延時間の分布', fontsize=12)
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 5. 水位と遅延時間の関係（3段目左）
        ax6 = axes[2, 0]
        
        # 水位3m以上かつ相関係数0.9以上でフィルタリング
        filtered_wl = self.raw_delay_results[
            (self.raw_delay_results['water_level'] >= 3.0) & 
            (self.raw_delay_results['correlation'] >= 0.9)
        ]
        
        if len(filtered_wl) > 0:
            # 増加期・減少期別に色分け表示
            increase_mask_wl = filtered_wl['flow_direction'] == 'increase'
            decrease_mask_wl = filtered_wl['flow_direction'] == 'decrease'
            other_mask_wl = ~(increase_mask_wl | decrease_mask_wl)
            
            if increase_mask_wl.any():
                ax6.scatter(filtered_wl[increase_mask_wl]['water_level'], 
                           filtered_wl[increase_mask_wl]['delay_minutes'],
                           c='lightcoral', s=5, alpha=0.6, label=f'増加期 (n={increase_mask_wl.sum()})')
            
            if decrease_mask_wl.any():
                ax6.scatter(filtered_wl[decrease_mask_wl]['water_level'], 
                           filtered_wl[decrease_mask_wl]['delay_minutes'],
                           c='lightblue', s=5, alpha=0.6, label=f'減少期 (n={decrease_mask_wl.sum()})')
            
            if other_mask_wl.any():
                ax6.scatter(filtered_wl[other_mask_wl]['water_level'], 
                           filtered_wl[other_mask_wl]['delay_minutes'],
                           c='lightgray', s=5, alpha=0.3, label=f'その他 (n={other_mask_wl.sum()})')
            
            # 増加期・減少期別のトレンドライン（2次関数）
            if increase_mask_wl.any() and increase_mask_wl.sum() >= 4:
                increase_data = filtered_wl[increase_mask_wl]
                z_inc = np.polyfit(increase_data['water_level'], increase_data['delay_minutes'], 2)
                p_inc = np.poly1d(z_inc)
                x_trend_inc = np.linspace(increase_data['water_level'].min(), 
                                        increase_data['water_level'].max(), 100)
                ax6.plot(x_trend_inc, p_inc(x_trend_inc), 'darkred', linewidth=2, 
                        label=f'増加期: y={z_inc[0]:.3f}x²+{z_inc[1]:.2f}x+{z_inc[2]:.1f}')
            
            if decrease_mask_wl.any() and decrease_mask_wl.sum() >= 4:
                decrease_data = filtered_wl[decrease_mask_wl]
                z_dec = np.polyfit(decrease_data['water_level'], decrease_data['delay_minutes'], 2)
                p_dec = np.poly1d(z_dec)
                x_trend_dec = np.linspace(decrease_data['water_level'].min(), 
                                        decrease_data['water_level'].max(), 100)
                ax6.plot(x_trend_dec, p_dec(x_trend_dec), 'darkblue', linewidth=2, 
                        label=f'減少期: y={z_dec[0]:.3f}x²+{z_dec[1]:.2f}x+{z_dec[2]:.1f}')
            
            # 全体のトレンドライン（2次関数）
            if len(filtered_wl) >= 4:
                z_all = np.polyfit(filtered_wl['water_level'], filtered_wl['delay_minutes'], 2)
                p_all = np.poly1d(z_all)
                x_trend_all = np.linspace(filtered_wl['water_level'].min(), 
                                        filtered_wl['water_level'].max(), 100)
                ax6.plot(x_trend_all, p_all(x_trend_all), 'black', linewidth=3, 
                        linestyle='--', alpha=0.8,
                        label=f'全体: y={z_all[0]:.3f}x²+{z_all[1]:.2f}x+{z_all[2]:.1f}')
        
        ax6.set_xlabel('水位 (m)')
        ax6.set_ylabel('遅延時間 (分)')
        ax6.set_title(f'水位と遅延時間の関係（水位≥3m, 相関≥0.9）\n(n={len(filtered_wl)})', fontsize=12)
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # 6. 放流量と遅延時間の関係（3段目中央）
        ax7 = axes[2, 1]
        
        # 放流量150m³/s以上かつ相関係数0.9以上でフィルタリング
        filtered_discharge = self.raw_delay_results[
            (self.raw_delay_results['discharge'] >= 150) &
            (self.raw_delay_results['correlation'] >= 0.9)
        ]
        
        if len(filtered_discharge) > 0:
            # 増加期・減少期別に色分け表示
            increase_mask_dis = filtered_discharge['flow_direction'] == 'increase'
            decrease_mask_dis = filtered_discharge['flow_direction'] == 'decrease'
            other_mask_dis = ~(increase_mask_dis | decrease_mask_dis)
            
            if increase_mask_dis.any():
                ax7.scatter(filtered_discharge[increase_mask_dis]['discharge'], 
                           filtered_discharge[increase_mask_dis]['delay_minutes'],
                           c='lightcoral', s=5, alpha=0.6, label=f'増加期 (n={increase_mask_dis.sum()})')
            
            if decrease_mask_dis.any():
                ax7.scatter(filtered_discharge[decrease_mask_dis]['discharge'], 
                           filtered_discharge[decrease_mask_dis]['delay_minutes'],
                           c='lightblue', s=5, alpha=0.6, label=f'減少期 (n={decrease_mask_dis.sum()})')
            
            if other_mask_dis.any():
                ax7.scatter(filtered_discharge[other_mask_dis]['discharge'], 
                           filtered_discharge[other_mask_dis]['delay_minutes'],
                           c='lightgray', s=5, alpha=0.3, label=f'その他 (n={other_mask_dis.sum()})')
            
            # 増加期・減少期別のトレンドライン（2次関数）
            if increase_mask_dis.any() and increase_mask_dis.sum() >= 4:
                increase_data = filtered_discharge[increase_mask_dis]
                z_inc = np.polyfit(increase_data['discharge'], increase_data['delay_minutes'], 2)
                p_inc = np.poly1d(z_inc)
                x_trend_inc = np.linspace(increase_data['discharge'].min(), 
                                        increase_data['discharge'].max(), 100)
                ax7.plot(x_trend_inc, p_inc(x_trend_inc), 'darkred', linewidth=2, 
                        label=f'増加期: y={z_inc[0]:.5f}x²+{z_inc[1]:.3f}x+{z_inc[2]:.1f}')
            
            if decrease_mask_dis.any() and decrease_mask_dis.sum() >= 4:
                decrease_data = filtered_discharge[decrease_mask_dis]
                z_dec = np.polyfit(decrease_data['discharge'], decrease_data['delay_minutes'], 2)
                p_dec = np.poly1d(z_dec)
                x_trend_dec = np.linspace(decrease_data['discharge'].min(), 
                                        decrease_data['discharge'].max(), 100)
                ax7.plot(x_trend_dec, p_dec(x_trend_dec), 'darkblue', linewidth=2, 
                        label=f'減少期: y={z_dec[0]:.5f}x²+{z_dec[1]:.3f}x+{z_dec[2]:.1f}')
            
            # 全体のトレンドライン（2次関数）
            if len(filtered_discharge) >= 4:
                z_all = np.polyfit(filtered_discharge['discharge'], filtered_discharge['delay_minutes'], 2)
                p_all = np.poly1d(z_all)
                x_trend_all = np.linspace(filtered_discharge['discharge'].min(), 
                                        filtered_discharge['discharge'].max(), 100)
                ax7.plot(x_trend_all, p_all(x_trend_all), 'black', linewidth=3, 
                        linestyle='--', alpha=0.8,
                        label=f'全体: y={z_all[0]:.5f}x²+{z_all[1]:.3f}x+{z_all[2]:.1f}')
        
        ax7.set_xlabel('放流量 (m³/s)')
        ax7.set_ylabel('遅延時間 (分)')
        ax7.set_title(f'放流量と遅延時間の関係（放流量≥150m³/s, 相関≥0.9）\n(n={len(filtered_discharge)})', fontsize=12)
        ax7.legend()
        ax7.grid(True, alpha=0.3)
        
        # 7. 相関係数の分布（3段目右）
        ax8 = axes[2, 2]
        
        ax8.hist(self.raw_delay_results['correlation'], bins=20, 
                alpha=0.7, color=data_color, edgecolor='black')
        
        ax8.axvline(x=self.raw_delay_results['correlation'].mean(), 
                   color='red', linestyle='--', linewidth=2,
                   label=f'平均: {self.raw_delay_results["correlation"].mean():.3f}')
        ax8.set_xlabel('相関係数')
        ax8.set_ylabel('データ点数')
        ax8.set_title('相関係数の分布', fontsize=12)
        ax8.legend()
        ax8.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存
        output_path = f"figures/figure8_raw_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Figure 8を保存しました: {output_path}")
        
        plt.close()
        
        return fig

def main():
    """単独実行時のみ使用"""
    analyzer = Figure8RawDataAnalysis()
    
    # データ読み込みと分析
    analyzer.load_data()
    analyzer.preprocess_data()
    analyzer.classify_stable_variable_periods()
    
    # Figure 8の作成
    analyzer.visualize_figure8_raw_data()
    
    print("\nFigure 8（生データ）の作成が完了しました。")

if __name__ == "__main__":
    main()