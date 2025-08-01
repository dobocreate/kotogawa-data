#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 10: 変動期間におけるピーク・ボトム対応分析
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from datetime import datetime
from scipy import signal

from analyze_water_level_delay import RiverDelayAnalysis

class Figure10PeakBottomAnalysis(RiverDelayAnalysis):
    # 親クラスのload_data()メソッドを使用（ダイアログ機能付き）
    
    def analyze_peak_bottom_correspondence(self):
        """変動期間でのピーク・ボトム対応分析"""
        print("\n=== ピーク・ボトム対応分析 ===")
        
        if not hasattr(self, 'variable_periods'):
            print("変動期間が定義されていません。")
            return None
        
        peak_results = []
        bottom_results = []
        
        for period_idx, (start, end) in enumerate(self.variable_periods):
            if end - start < 12:  # 最低2時間
                continue
            
            # 期間内のデータ
            discharge = self.df_processed['ダム_全放流量'].iloc[start:end].values
            water_level = self.df_processed['水位_水位'].iloc[start:end].values
            time_points = np.arange(len(discharge))
            
            # 欠損値を除去
            mask = ~(np.isnan(discharge) | np.isnan(water_level))
            if mask.sum() < 12:
                continue
            
            discharge_clean = discharge[mask]
            water_level_clean = water_level[mask]
            time_clean = time_points[mask]
            
            # スムージング（ノイズ除去）
            window = 3  # 30分窓
            discharge_smooth = np.convolve(discharge_clean, np.ones(window)/window, mode='same')
            water_level_smooth = np.convolve(water_level_clean, np.ones(window)/window, mode='same')
            
            # ピーク検出
            discharge_peaks, _ = signal.find_peaks(discharge_smooth, distance=6, prominence=10)
            water_peaks, _ = signal.find_peaks(water_level_smooth, distance=6, prominence=0.1)
            
            # ボトム検出
            discharge_bottoms, _ = signal.find_peaks(-discharge_smooth, distance=6, prominence=10)
            water_bottoms, _ = signal.find_peaks(-water_level_smooth, distance=6, prominence=0.1)
            
            # ピーク対応分析
            for dp_idx in discharge_peaks:
                dp_time = time_clean[dp_idx]
                dp_value = discharge_smooth[dp_idx]
                
                # 対応する水位ピークを探す（最大3時間後まで）
                max_delay = 18  # 180分
                candidates = []
                
                for wp_idx in water_peaks:
                    wp_time = time_clean[wp_idx]
                    delay = (wp_time - dp_time) * 10  # 分単位
                    
                    if 0 <= delay <= max_delay * 10:
                        wp_value = water_level_smooth[wp_idx]
                        candidates.append({
                            'delay_minutes': delay,
                            'water_peak_value': wp_value,
                            'water_peak_idx': wp_idx
                        })
                
                if candidates:
                    # 最も近いピークを選択
                    best_candidate = min(candidates, key=lambda x: x['delay_minutes'])
                    
                    # フィルタリング条件: 水位≥3m かつ 放流量≥150m³/s
                    if best_candidate['water_peak_value'] >= 3.0 and dp_value >= 150.0:
                        peak_results.append({
                            'period_idx': period_idx,
                            'discharge_peak_time': start + dp_time,
                            'discharge_peak_value': dp_value,
                            'water_peak_value': best_candidate['water_peak_value'],
                            'delay_minutes': best_candidate['delay_minutes'],
                            'peak_ratio': best_candidate['water_peak_value'] / dp_value if dp_value > 0 else np.nan,
                            'water_discharge_slope': (best_candidate['water_peak_value'] - water_level_smooth[0]) / dp_value if dp_value > 0 else np.nan
                        })
            
            # ボトム対応分析
            for db_idx in discharge_bottoms:
                db_time = time_clean[db_idx]
                db_value = discharge_smooth[db_idx]
                
                # 対応する水位ボトムを探す
                candidates = []
                
                for wb_idx in water_bottoms:
                    wb_time = time_clean[wb_idx]
                    delay = (wb_time - db_time) * 10
                    
                    if 0 <= delay <= max_delay * 10:
                        wb_value = water_level_smooth[wb_idx]
                        candidates.append({
                            'delay_minutes': delay,
                            'water_bottom_value': wb_value,
                            'water_bottom_idx': wb_idx
                        })
                
                if candidates:
                    best_candidate = min(candidates, key=lambda x: x['delay_minutes'])
                    
                    # フィルタリング条件: 水位≥3m かつ 放流量≥150m³/s
                    if best_candidate['water_bottom_value'] >= 3.0 and db_value >= 150.0:
                        bottom_results.append({
                            'period_idx': period_idx,
                            'discharge_bottom_time': start + db_time,
                            'discharge_bottom_value': db_value,
                            'water_bottom_value': best_candidate['water_bottom_value'],
                            'delay_minutes': best_candidate['delay_minutes']
                        })
        
        self.peak_results = pd.DataFrame(peak_results)
        self.bottom_results = pd.DataFrame(bottom_results)
        
        print(f"検出されたピーク対応: {len(peak_results)}件（水位≥3m、放流量≥150m³/s）")
        print(f"検出されたボトム対応: {len(bottom_results)}件（水位≥3m、放流量≥150m³/s）")
        
        if len(peak_results) > 0:
            print(f"\nピーク遅延時間: 平均{self.peak_results['delay_minutes'].mean():.1f}分、"
                  f"中央値{self.peak_results['delay_minutes'].median():.1f}分")
        
        if len(bottom_results) > 0:
            print(f"ボトム遅延時間: 平均{self.bottom_results['delay_minutes'].mean():.1f}分、"
                  f"中央値{self.bottom_results['delay_minutes'].median():.1f}分")
        
        return self.peak_results, self.bottom_results
    
    def create_figure10_peak_bottom_analysis(self):
        """Figure 10: ピーク・ボトム対応分析の可視化"""
        print("\n=== Figure 10: ピーク・ボトム対応分析 ===")
        
        # 分析実行
        if not hasattr(self, 'peak_results'):
            self.analyze_peak_bottom_correspondence()
        
        # レイアウトを3行3列に設定
        fig, axes = plt.subplots(3, 3, figsize=(20, 12), num='Figure 10')
        
        # 1. 代表的な変動期間の例（1行目）
        example_periods = []
        
        # ピーク対応が明確な期間を3つ選択
        if len(self.peak_results) > 0:
            # 遅延時間が短い、中間、長いものを選択
            sorted_peaks = self.peak_results.sort_values('delay_minutes')
            indices = [0, len(sorted_peaks)//2, len(sorted_peaks)-1]
            
            for idx in indices:
                if idx < len(sorted_peaks):
                    period_idx = int(sorted_peaks.iloc[idx]['period_idx'])
                    example_periods.append(period_idx)
        
        # 時系列表示
        for i, period_idx in enumerate(example_periods[:3]):
            ax = axes[0, i]
            
            if period_idx < len(self.variable_periods):
                start, end = self.variable_periods[period_idx]
                
                # 表示範囲を期間の前後に拡張
                display_start = max(0, start - 12)
                display_end = min(end + 12, len(self.df_processed))
                
                time_hours = np.arange(display_end - display_start) / 6
                discharge = self.df_processed['ダム_全放流量'].iloc[display_start:display_end]
                water_level = self.df_processed['水位_水位'].iloc[display_start:display_end]
                
                # 変動期間を色付け
                var_start_rel = (start - display_start) / 6
                var_end_rel = (end - display_start) / 6
                ax.axvspan(var_start_rel, var_end_rel, alpha=0.3, color='lightpink', label='変動期間')
                
                # ピーク・ボトムをマーク
                period_peaks = self.peak_results[self.peak_results['period_idx'] == period_idx]
                period_bottoms = self.bottom_results[self.bottom_results['period_idx'] == period_idx]
                
                ax2 = ax.twinx()
                ax.plot(time_hours, discharge, 'b-', linewidth=2, label='放流量')
                ax2.plot(time_hours, water_level, 'r-', linewidth=2, label='水位')
                
                # ピークをマーク
                for _, peak in period_peaks.iterrows():
                    peak_time = (peak['discharge_peak_time'] - display_start) / 6
                    if 0 <= peak_time <= max(time_hours):
                        ax.scatter(peak_time, peak['discharge_peak_value'], 
                                 color='blue', s=100, marker='^', zorder=5)
                        # 対応する水位ピーク
                        water_peak_time = peak_time + peak['delay_minutes'] / 60
                        if water_peak_time <= max(time_hours):
                            ax2.scatter(water_peak_time, peak['water_peak_value'], 
                                      color='red', s=100, marker='^', zorder=5)
                            # 遅延時間を矢印で表示
                            ax.annotate('', xy=(water_peak_time, 0.1), xytext=(peak_time, 0.1),
                                      xycoords=('data', 'axes fraction'),
                                      arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
                            ax.text((peak_time + water_peak_time)/2, 0.15, 
                                   f"{peak['delay_minutes']:.0f}分",
                                   transform=ax.get_xaxis_transform(), ha='center', fontsize=10)
                
                ax.set_xlabel('時間 (hours)')
                ax.set_ylabel('放流量 (m³/s)', color='b')
                ax2.set_ylabel('水位 (m)', color='r')
                ax.set_title(f'変動期間例 {i+1}', fontsize=12)
                ax.grid(True, alpha=0.3)
                
                if i == 0:
                    ax.legend(loc='upper left')
                    ax2.legend(loc='upper right')
        
        # 2. ピーク遅延時間の分布（2行目左）
        ax3 = axes[1, 0]
        
        if len(self.peak_results) > 0:
            ax3.hist(self.peak_results['delay_minutes'], bins=15, 
                    color='lightcoral', edgecolor='black', alpha=0.7)
            
            mean_delay = self.peak_results['delay_minutes'].mean()
            median_delay = self.peak_results['delay_minutes'].median()
            
            ax3.axvline(mean_delay, color='red', linestyle='--', linewidth=2, 
                       label=f'平均: {mean_delay:.1f}分')
            ax3.axvline(median_delay, color='darkred', linestyle=':', linewidth=2,
                       label=f'中央値: {median_delay:.1f}分')
            
            ax3.set_xlabel('遅延時間 (分)')
            ax3.set_ylabel('頻度')
            ax3.set_title(f'ピーク遅延時間の分布 (n={len(self.peak_results)})', fontsize=12)
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 3. ボトム遅延時間の分布（2行目中央）
        ax4 = axes[1, 1]
        
        if len(self.bottom_results) > 0:
            ax4.hist(self.bottom_results['delay_minutes'], bins=15,
                    color='lightblue', edgecolor='black', alpha=0.7)
            
            mean_delay = self.bottom_results['delay_minutes'].mean()
            median_delay = self.bottom_results['delay_minutes'].median()
            
            ax4.axvline(mean_delay, color='blue', linestyle='--', linewidth=2,
                       label=f'平均: {mean_delay:.1f}分')
            ax4.axvline(median_delay, color='darkblue', linestyle=':', linewidth=2,
                       label=f'中央値: {median_delay:.1f}分')
            
            ax4.set_xlabel('遅延時間 (分)')
            ax4.set_ylabel('頻度')
            ax4.set_title(f'ボトム遅延時間の分布 (n={len(self.bottom_results)})', fontsize=12)
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        # 4. 放流量範囲別の応答特性（2行目右）
        ax5 = axes[1, 2]
        
        if len(self.peak_results) > 0:
            # 放流量を範囲別に分類（150m³/s以上のみ）
            discharge_ranges = [
                (150, 300, '150-300'),
                (300, 500, '300-500'),
                (500, 800, '500-800'),
                (800, 1200, '800-1200'),
                (1200, 2000, '1200+')
            ]
            
            range_stats = []
            
            for min_d, max_d, label in discharge_ranges:
                mask = (self.peak_results['discharge_peak_value'] >= min_d) & \
                       (self.peak_results['discharge_peak_value'] < max_d)
                subset = self.peak_results[mask]
                
                if len(subset) > 0:
                    # 平均遅延時間と水位応答比
                    avg_delay = subset['delay_minutes'].mean()
                    avg_ratio = subset['peak_ratio'].mean()
                    std_delay = subset['delay_minutes'].std()
                    count = len(subset)
                    
                    range_stats.append({
                        'label': label,
                        'avg_delay': avg_delay,
                        'std_delay': std_delay,
                        'avg_ratio': avg_ratio,
                        'count': count,
                        'center': (min_d + max_d) / 2
                    })
            
            if range_stats:
                # 2軸グラフ
                ax5_twin = ax5.twinx()
                
                x_pos = [rs['center'] for rs in range_stats]
                delays = [rs['avg_delay'] for rs in range_stats]
                ratios = [rs['avg_ratio'] for rs in range_stats]
                errors = [rs['std_delay'] for rs in range_stats]
                
                # 遅延時間（左軸）
                ax5.errorbar(x_pos, delays, yerr=errors, fmt='bo-', 
                           capsize=5, linewidth=2, markersize=8,
                           label='平均遅延時間')
                
                # 水位応答比（右軸）
                ax5_twin.plot(x_pos, ratios, 'rs--', linewidth=2, markersize=8,
                            label='水位/放流量比')
                
                # データ数を表示
                for rs in range_stats:
                    ax5.text(rs['center'], rs['avg_delay'] + rs['std_delay'] + 5,
                           f"n={rs['count']}", ha='center', fontsize=8)
                
                ax5.set_xlabel('放流量範囲 (m³/s)')
                ax5.set_ylabel('遅延時間 (分)', color='b')
                ax5_twin.set_ylabel('水位/放流量比', color='r')
                ax5.set_title('放流量範囲別の応答特性', fontsize=12)
                ax5.tick_params(axis='y', labelcolor='b')
                ax5_twin.tick_params(axis='y', labelcolor='r')
                ax5.grid(True, alpha=0.3)
                ax5.legend(loc='upper left')
                ax5_twin.legend(loc='upper right')
        
        # 5. 放流量ピーク値 vs 水位ピーク値（3行目左）
        ax6 = axes[2, 0]
        
        if len(self.peak_results) > 0:
            # 散布図：放流量ピーク vs 水位ピーク
            scatter = ax6.scatter(self.peak_results['discharge_peak_value'], 
                                 self.peak_results['water_peak_value'],
                                 c=self.peak_results['delay_minutes'],
                                 cmap='viridis', alpha=0.6, s=50)
            
            # カラーバー追加
            cbar = plt.colorbar(scatter, ax=ax6)
            cbar.set_label('遅延時間 (分)', rotation=270, labelpad=15)
            
            # トレンドライン（線形回帰）
            if len(self.peak_results) >= 3:
                x = self.peak_results['discharge_peak_value'].values
                y = self.peak_results['water_peak_value'].values
                
                # 線形回帰
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                x_trend = np.linspace(x.min(), x.max(), 100)
                ax6.plot(x_trend, p(x_trend), 'r--', linewidth=2, 
                        label=f'y = {z[0]:.4f}x + {z[1]:.2f}')
                
                # 相関係数計算
                corr_coef = np.corrcoef(x, y)[0, 1]
                ax6.text(0.05, 0.95, f'R = {corr_coef:.3f}', 
                        transform=ax6.transAxes, fontsize=10,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax6.set_xlabel('放流量ピーク値 (m³/s)')
            ax6.set_ylabel('水位ピーク値 (m)')
            ax6.set_title('放流量ピーク vs 水位ピーク（色：遅延時間）', fontsize=12)
            ax6.legend()
            ax6.grid(True, alpha=0.3)
        
        # 6. 水位ピーク値 vs 遅延時間の散布図（3行目中央）
        ax7 = axes[2, 1]
        
        if len(self.peak_results) > 0:
            # 散布図：水位ピーク値 vs 遅延時間
            scatter = ax7.scatter(self.peak_results['water_peak_value'], 
                                 self.peak_results['delay_minutes'],
                                 c=self.peak_results['discharge_peak_value'],
                                 cmap='viridis', alpha=0.6, s=80, edgecolors='black', linewidth=0.5)
            
            # カラーバー追加
            cbar = plt.colorbar(scatter, ax=ax7)
            cbar.set_label('放流量ピーク値 (m³/s)', rotation=270, labelpad=15)
            
            # トレンドライン（線形回帰）
            if len(self.peak_results) >= 3:
                x = self.peak_results['water_peak_value'].values
                y = self.peak_results['delay_minutes'].values
                
                # 線形回帰でトレンドライン
                z = np.polyfit(x, y, 1)
                x_trend = np.linspace(x.min(), x.max(), 100)
                y_trend = z[0] * x_trend + z[1]
                ax7.plot(x_trend, y_trend, 'k--', linewidth=2, 
                        label=f'τ = {z[0]:.1f}·H + {z[1]:.1f}')
                
                # 相関係数計算
                corr_coef = np.corrcoef(x, y)[0, 1]
                ax7.text(0.95, 0.05, f'R = {corr_coef:.3f}', 
                        transform=ax7.transAxes, fontsize=10,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                        ha='right', va='bottom')
            
            ax7.set_xlabel('水位ピーク値 (m)')
            ax7.set_ylabel('遅延時間 (分)')
            ax7.set_title('水位ピーク値 vs 遅延時間', fontsize=12)
            ax7.legend(loc='upper left')
            ax7.grid(True, alpha=0.3)
        
        # 7. 放流量ピーク値 vs 遅延時間の散布図（3行目右）
        ax8 = axes[2, 2]
        
        if len(self.peak_results) > 0:
            # 散布図：放流量ピーク値 vs 遅延時間
            scatter = ax8.scatter(self.peak_results['discharge_peak_value'], 
                                 self.peak_results['delay_minutes'],
                                 c=self.peak_results['water_peak_value'],
                                 cmap='coolwarm', alpha=0.6, s=80, edgecolors='black', linewidth=0.5)
            
            # カラーバー追加
            cbar = plt.colorbar(scatter, ax=ax8)
            cbar.set_label('水位ピーク値 (m)', rotation=270, labelpad=15)
            
            # トレンドライン（対数関数フィッティング）
            if len(self.peak_results) >= 3:
                x = self.peak_results['discharge_peak_value'].values
                y = self.peak_results['delay_minutes'].values
                
                # 対数関数でフィッティング（Q > 0の条件で）
                valid_mask = x > 0
                x_valid = x[valid_mask]
                y_valid = y[valid_mask]
                
                if len(x_valid) >= 3:
                    # 線形回帰でトレンドライン
                    z = np.polyfit(np.log(x_valid), y_valid, 1)
                    x_trend = np.linspace(x_valid.min(), x_valid.max(), 100)
                    y_trend = z[0] * np.log(x_trend) + z[1]
                    ax8.plot(x_trend, y_trend, 'k--', linewidth=2, 
                            label=f'τ = {z[0]:.1f}·ln(Q) + {z[1]:.1f}')
                    
                    # 相関係数計算
                    corr_coef = np.corrcoef(np.log(x_valid), y_valid)[0, 1]
                    ax8.text(0.95, 0.05, f'R = {corr_coef:.3f}', 
                            transform=ax8.transAxes, fontsize=10,
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                            ha='right', va='bottom')
            
            ax8.set_xlabel('放流量ピーク値 (m³/s)')
            ax8.set_ylabel('遅延時間 (分)')
            ax8.set_title('放流量ピーク値 vs 遅延時間', fontsize=12)
            ax8.legend(loc='upper right')
            ax8.grid(True, alpha=0.3)
            ax8.set_xlim(left=0)
        
        plt.tight_layout()
        
        # 統計サマリーを表示
        if len(self.peak_results) > 0 or len(self.bottom_results) > 0:
            print("\n=== ピーク・ボトム対応分析サマリー ===")
            if len(self.peak_results) > 0:
                print(f"ピーク分析:")
                print(f"  サンプル数: {len(self.peak_results)}")
                print(f"  遅延時間: {self.peak_results['delay_minutes'].min():.0f}-"
                      f"{self.peak_results['delay_minutes'].max():.0f}分")
                print(f"  放流量範囲: {self.peak_results['discharge_peak_value'].min():.0f}-"
                      f"{self.peak_results['discharge_peak_value'].max():.0f} m³/s")
            
            if len(self.bottom_results) > 0:
                print(f"\nボトム分析:")
                print(f"  サンプル数: {len(self.bottom_results)}")
                print(f"  遅延時間: {self.bottom_results['delay_minutes'].min():.0f}-"
                      f"{self.bottom_results['delay_minutes'].max():.0f}分")
        
        # 保存
        output_path = f"figures/figure10_peak_bottom_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nFigure 10を保存しました: {output_path}")
        
        plt.close()
        
        return fig

def main():
    """単独実行時のみ使用"""
    analyzer = Figure10PeakBottomAnalysis()
    
    # データ読み込みと分析
    analyzer.load_data()
    analyzer.preprocess_data()
    analyzer.classify_stable_variable_periods()
    analyzer.analyze_variable_periods()
    
    # Figure 10の作成
    analyzer.create_figure10_peak_bottom_analysis()
    
    print("\nFigure 10（ピーク・ボトム対応分析）の作成が完了しました。")

if __name__ == "__main__":
    main()