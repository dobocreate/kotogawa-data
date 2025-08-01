#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 5に水位と遅延時間の関係を追加
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from datetime import datetime

from analyze_water_level_delay import RiverDelayAnalysis

class UpdatedFigure5Analysis(RiverDelayAnalysis):
    # 親クラスのload_data()メソッドを使用（ダイアログ機能付き）
    
    def unify_stable_periods(self):
        """安定期間を統合"""
        if not hasattr(self, 'detailed_stable_results'):
            print("詳細な安定期間分析が実行されていません。")
            return
        
        # 統合された安定期間の結果
        self.unified_stable_results = {}
        
        # 1. イベント前後の安定期間を統合
        event_related_periods = []
        
        if 'pre_stable' in self.detailed_stable_results:
            pre_df = self.detailed_stable_results['pre_stable'].copy()
            pre_df['original_type'] = 'イベント前安定'
            event_related_periods.append(pre_df)
        
        if 'post_stable' in self.detailed_stable_results:
            post_df = self.detailed_stable_results['post_stable'].copy()
            post_df['original_type'] = 'イベント後安定'
            event_related_periods.append(post_df)
        
        if event_related_periods:
            event_related_df = pd.concat(event_related_periods, ignore_index=True)
            self.unified_stable_results['安定期間（イベント前後）'] = event_related_df
        
        # 2. 低活動期間を平常状態として分類
        if 'low_activity' in self.detailed_stable_results:
            normal_df = self.detailed_stable_results['low_activity'].copy()
            normal_df['original_type'] = '低活動'
            self.unified_stable_results['安定期間（平常状態）'] = normal_df
        
        return self.unified_stable_results
    
    def visualize_updated_figure5(self):
        """水位と遅延時間の関係を追加したFigure 5"""
        print("\n=== 更新されたFigure 5の視覚化 ===")
        
        if not hasattr(self, 'unified_stable_results'):
            self.unify_stable_periods()
        
        # レイアウトを3行3列に変更
        fig, axes = plt.subplots(3, 3, figsize=(20, 12), num='Figure 5')
        
        # 色の設定
        unified_colors = {
            '安定期間（イベント前後）': 'darkgreen',
            '安定期間（平常状態）': 'lightgray',
        }
        
        markers = {
            '安定期間（イベント前後）': 'o',
            '安定期間（平常状態）': '.'
        }
        
        # 1. 特定範囲の期間の時系列表示（上部3つ）- 水位3-4.2m、遅延100-120分
        # 全ての安定期間から条件に合う期間を抽出
        target_periods = []
        
        print("\n=== 特定範囲の期間探索（水位: 3-4.2m、遅延: 100-120分） ===")
        for period_type in ['pre_stable', 'post_stable', 'low_activity']:
            if period_type in self.detailed_stable_results:
                df = self.detailed_stable_results[period_type]
                target_count = 0
                
                for idx, row in df.iterrows():
                    delay_minutes = row['delay_minutes']
                    start, end = int(row['start']), int(row['end'])
                    water_level = self.df_processed['水位_水位'].iloc[start:end].mean()
                    discharge = self.df_processed['ダム_全放流量'].iloc[start:end].mean()
                    
                    # 条件: 水位3-4.2m かつ 遅延時間100-120分
                    if 3.0 <= water_level <= 4.2 and 100 <= delay_minutes <= 120:
                        target_periods.append({
                            'start': start,
                            'end': end,
                            'delay_minutes': delay_minutes,
                            'period_type': period_type,
                            'water_level': water_level,
                            'discharge': discharge,
                            'correlation': row['correlation']
                        })
                        target_count += 1
                
                print(f"{period_type}: {target_count}期間（条件適合）")
        
        # 遅延時間の長い順にソート
        target_periods = sorted(target_periods, key=lambda x: x['delay_minutes'], reverse=True)
        
        # 上位3つを表示
        for i in range(3):
            ax = axes[0, i]
            
            if i < len(target_periods):
                period = target_periods[i]
                period_start = period['start']
                display_start = max(0, period_start - 36)
                display_end = min(period['end'] + 36, len(self.df_processed))
                
                time_hours = np.arange(display_end - display_start) / 6
                discharge = self.df_processed['ダム_全放流量'].iloc[display_start:display_end]
                water_level = self.df_processed['水位_水位'].iloc[display_start:display_end]
                
                # 安定期間の範囲を色づけ（高遅延は赤系の色）
                stable_start_rel = (period['start'] - display_start) / 6
                stable_end_rel = (period['end'] - display_start) / 6
                
                ax.axvspan(stable_start_rel, stable_end_rel, alpha=0.3, color='lightcoral', 
                          label='高遅延期間' if i == 0 else '')
                
                ax2 = ax.twinx()
                line1 = ax.plot(time_hours, discharge, 'b-', linewidth=2, label='放流量')
                line2 = ax2.plot(time_hours, water_level, 'r-', linewidth=2, label='水位')
                
                # 遅延時間を表示
                ax.text(0.02, 0.98, f'遅延時間: {period["delay_minutes"]:.1f}分\n相関係数: {period["correlation"]:.3f}', 
                       transform=ax.transAxes, va='top', ha='left',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                ax.set_xlabel('時間 (hours)')
                ax.set_ylabel('放流量 (m³/s)', color='b')
                ax2.set_ylabel('水位 (m)', color='r')
                
                period_type_label = {
                    'pre_stable': 'イベント前安定',
                    'post_stable': 'イベント後安定', 
                    'low_activity': '低活動期間'
                }
                ax.set_title(f'高遅延期間 {i+1} ({period_type_label[period["period_type"]]})', fontsize=12)
                ax.grid(True, alpha=0.3)
                
                if i == 0:
                    lines = line1 + line2
                    labels = [l.get_label() for l in lines]
                    ax.legend(lines, labels, loc='upper left')
            else:
                ax.text(0.5, 0.5, 'データなし', ha='center', va='center', transform=ax.transAxes)
        
        # 2. 修正された散布図（2段目左）
        ax3 = axes[1, 0]
        
        cv_threshold = 0.1
        plot_counts = {'安定期間（イベント前後）': 0, '安定期間（平常状態）': 0}
        
        for period_type in ['安定期間（イベント前後）', '安定期間（平常状態）']:
            if period_type in self.unified_stable_results:
                df = self.unified_stable_results[period_type]
                
                for idx, row in df.iterrows():
                    start, end = int(row['start']), int(row['end'])
                    discharge = self.df_processed['ダム_全放流量'].iloc[start:end]
                    water_level = self.df_processed['水位_水位'].iloc[start:end]
                    
                    if discharge.mean() > 0:
                        cv = discharge.std() / discharge.mean()
                        
                        if cv < cv_threshold:
                            marker_size = 100 if period_type == '安定期間（イベント前後）' else 30
                            alpha = 0.8
                            edge_color = 'black'
                        else:
                            marker_size = 60 if period_type == '安定期間（イベント前後）' else 20
                            alpha = 0.5
                            edge_color = 'gray'
                        
                        ax3.scatter(discharge.mean(), water_level.mean(), 
                                  color=unified_colors[period_type], marker=markers[period_type],
                                  s=marker_size, alpha=alpha, edgecolors=edge_color,
                                  label=f'{period_type} (CV<{cv_threshold*100}%)' if plot_counts[period_type] == 0 and cv < cv_threshold else '')
                        
                        plot_counts[period_type] += 1
        
        ax3.set_xlabel('平均放流量 (m³/s)')
        ax3.set_ylabel('平均水位 (m)')
        ax3.set_title(f'安定期間での放流量-水位関係\\n（濃い色: CV<10%, 薄い色: CV≥10%）', fontsize=12)
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        textstr = f'プロット数:\n安定期間（イベント前後）: {plot_counts["安定期間（イベント前後）"]}\n安定期間（平常状態）: {plot_counts["安定期間（平常状態）"]}'
        ax3.text(0.02, 0.98, textstr, transform=ax3.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # 3. データ数の比較（2段目中央）
        ax4 = axes[1, 1]
        
        data_counts = {}
        range_labels = ['0-100', '100-200', '200-300', '300-400', '400-500']
        discharge_ranges = [(0, 100), (100, 200), (200, 300), (300, 400), (400, 500)]
        
        for period_type in ['安定期間（イベント前後）', '安定期間（平常状態）']:
            if period_type in self.unified_stable_results:
                df = self.unified_stable_results[period_type]
                
                range_counts = []
                for min_d, max_d in discharge_ranges:
                    count = len(df[(df['avg_discharge'] >= min_d) & (df['avg_discharge'] < max_d)])
                    range_counts.append(count)
                
                data_counts[period_type] = range_counts
        
        x_pos = np.arange(len(range_labels))
        width = 0.35
        
        if '安定期間（イベント前後）' in data_counts:
            bars1 = ax4.bar(x_pos - width/2, data_counts['安定期間（イベント前後）'], width,
                           label='安定期間（イベント前後）', color=unified_colors['安定期間（イベント前後）'], 
                           alpha=0.7, edgecolor='black')
        
        if '安定期間（平常状態）' in data_counts:
            bars2 = ax4.bar(x_pos + width/2, data_counts['安定期間（平常状態）'], width,
                           label='安定期間（平常状態）', color=unified_colors['安定期間（平常状態）'], 
                           alpha=0.7, edgecolor='black')
        
        for i, label in enumerate(range_labels):
            if '安定期間（イベント前後）' in data_counts:
                count1 = data_counts['安定期間（イベント前後）'][i]
                if count1 > 0:
                    ax4.text(i - width/2, count1 + 0.5, str(count1), 
                            ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            if '安定期間（平常状態）' in data_counts:
                count2 = data_counts['安定期間（平常状態）'][i]
                if count2 > 0:
                    ax4.text(i + width/2, count2 + 0.5, str(count2), 
                            ha='center', va='bottom', fontsize=10)
        
        ax4.set_xlabel('放流量範囲 (m³/s)')
        ax4.set_ylabel('期間数')
        ax4.set_title('放流量範囲別の安定期間データ数', fontsize=14)
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(range_labels)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 4. CVの分布（2段目右）
        ax5 = axes[1, 2]
        
        cv_data = {'安定期間（イベント前後）': [], '安定期間（平常状態）': []}
        
        for period_type in ['安定期間（イベント前後）', '安定期間（平常状態）']:
            if period_type in self.unified_stable_results:
                df = self.unified_stable_results[period_type]
                
                for idx, row in df.iterrows():
                    start, end = int(row['start']), int(row['end'])
                    discharge = self.df_processed['ダム_全放流量'].iloc[start:end]
                    
                    if discharge.mean() > 0:
                        cv = discharge.std() / discharge.mean()
                        cv_data[period_type].append(cv)
        
        for period_type, cvs in cv_data.items():
            if cvs:
                ax5.hist(cvs, bins=20, alpha=0.7, label=f'{period_type} (n={len(cvs)})',
                        color=unified_colors[period_type], edgecolor='black')
        
        ax5.axvline(x=0.05, color='red', linestyle='--', label='CV=5%（元のフィルタ）')
        ax5.axvline(x=0.1, color='orange', linestyle='--', label='CV=10%（新フィルタ）')
        ax5.set_xlabel('変動係数 (CV)')
        ax5.set_ylabel('期間数')
        ax5.set_title('安定期間の変動係数分布', fontsize=12)
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 5. 水位と遅延時間の関係（3段目左）
        ax6 = axes[2, 0]
        
        water_levels = []
        delay_times = []
        period_types = []
        
        # 全安定期間のデータを収集
        for period_type in ['pre_stable', 'post_stable', 'low_activity']:
            if period_type in self.detailed_stable_results:
                df = self.detailed_stable_results[period_type]
                for idx, row in df.iterrows():
                    start, end = int(row['start']), int(row['end'])
                    water_level = self.df_processed['水位_水位'].iloc[start:end].mean()
                    water_levels.append(water_level)
                    delay_times.append(row['delay_minutes'])
                    
                    if period_type == 'pre_stable':
                        period_types.append('イベント前安定')
                    elif period_type == 'post_stable':
                        period_types.append('イベント後安定')
                    else:
                        period_types.append('平常状態')
        
        if water_levels and delay_times:
            # 平均水位3m以上のデータのみフィルタリング
            filtered_data = [(wl, dt, pt) for wl, dt, pt in zip(water_levels, delay_times, period_types) if wl >= 3.0]
            
            if filtered_data:
                filtered_water_levels, filtered_delay_times, filtered_period_types = zip(*filtered_data)
                
                # 期間タイプ別に色分け
                colors_map = {
                    'イベント前安定': 'lightblue',
                    'イベント後安定': 'lightcoral',
                    '平常状態': 'lightgray'
                }
                
                for ptype in colors_map:
                    mask = [pt == ptype for pt in filtered_period_types]
                    wl_subset = [wl for wl, m in zip(filtered_water_levels, mask) if m]
                    dt_subset = [dt for dt, m in zip(filtered_delay_times, mask) if m]
                    
                    if wl_subset:
                        ax6.scatter(wl_subset, dt_subset, 
                                  c=colors_map[ptype], s=80, alpha=0.7, 
                                  edgecolors='black', label=f'{ptype} (n={len(wl_subset)})')
                
                # 平均水位3m以上のトレンドライン
                if len(filtered_water_levels) >= 3:
                    z = np.polyfit(filtered_water_levels, filtered_delay_times, 2)
                    p = np.poly1d(z)
                    x_trend = np.linspace(min(filtered_water_levels), max(filtered_water_levels), 100)
                    ax6.plot(x_trend, p(x_trend), 'r-', linewidth=2, label='トレンド（2次多項式, 水位≥3m）')
            else:
                ax6.text(0.5, 0.5, '水位3m以上のデータなし', ha='center', va='center', transform=ax6.transAxes)
            
                # 軸の範囲をデータに合わせて調整
                ax6.set_xlim(min(filtered_water_levels) - 0.1, max(filtered_water_levels) + 0.1)
                ax6.set_ylim(min(filtered_delay_times) - 10, max(filtered_delay_times) + 10)
            
        ax6.set_xlabel('平均水位 (m)', fontsize=12)
        ax6.set_ylabel('遅延時間 (分)', fontsize=12)
        ax6.set_title('水位と遅延時間の関係（水位≥3m）', fontsize=12, fontweight='bold')
        ax6.grid(True, alpha=0.3)
        ax6.legend()
        
        # 6. 平均放流量と遅延時間の関係（3段目中央）- 新規追加
        ax7 = axes[2, 1]
        
        discharge_levels = []
        discharge_delay_times = []
        discharge_period_types = []
        
        # 全安定期間のデータを収集
        for period_type in ['pre_stable', 'post_stable', 'low_activity']:
            if period_type in self.detailed_stable_results:
                df = self.detailed_stable_results[period_type]
                for idx, row in df.iterrows():
                    start, end = int(row['start']), int(row['end'])
                    avg_discharge = self.df_processed['ダム_全放流量'].iloc[start:end].mean()
                    discharge_levels.append(avg_discharge)
                    discharge_delay_times.append(row['delay_minutes'])
                    
                    if period_type == 'pre_stable':
                        discharge_period_types.append('イベント前安定')
                    elif period_type == 'post_stable':
                        discharge_period_types.append('イベント後安定')
                    else:
                        discharge_period_types.append('平常状態')
        
        if discharge_levels and discharge_delay_times:
            # 放流量150m³/s以上のデータのみフィルタリング
            filtered_discharge_data = [(dl, dt, pt) for dl, dt, pt in zip(discharge_levels, discharge_delay_times, discharge_period_types) if dl >= 150.0]
            
            if filtered_discharge_data:
                filtered_discharge_levels, filtered_discharge_delay_times, filtered_discharge_period_types = zip(*filtered_discharge_data)
                
                # 期間タイプ別に色分け
                colors_map = {
                    'イベント前安定': 'lightblue',
                    'イベント後安定': 'lightcoral',
                    '平常状態': 'lightgray'
                }
                
                for ptype in colors_map:
                    mask = [pt == ptype for pt in filtered_discharge_period_types]
                    dl_subset = [dl for dl, m in zip(filtered_discharge_levels, mask) if m]
                    dt_subset = [dt for dt, m in zip(filtered_discharge_delay_times, mask) if m]
                    
                    if dl_subset:
                        ax7.scatter(dl_subset, dt_subset, 
                                   c=colors_map[ptype], s=80, alpha=0.7, 
                                   edgecolors='black', label=f'{ptype} (n={len(dl_subset)})')
                
                # 放流量150m³/s以上のトレンドライン
                if len(filtered_discharge_levels) >= 3:
                    z = np.polyfit(filtered_discharge_levels, filtered_discharge_delay_times, 2)
                    p = np.poly1d(z)
                    x_trend = np.linspace(min(filtered_discharge_levels), max(filtered_discharge_levels), 100)
                    ax7.plot(x_trend, p(x_trend), 'g-', linewidth=2, label='トレンド（2次多項式, 放流量≥150）')
                
                # 軸の範囲をデータに合わせて調整
                ax7.set_xlim(min(filtered_discharge_levels) - 20, max(filtered_discharge_levels) + 20)
                ax7.set_ylim(min(filtered_discharge_delay_times) - 10, max(filtered_discharge_delay_times) + 10)
            else:
                ax7.text(0.5, 0.5, '放流量150m³/s以上のデータなし', ha='center', va='center', transform=ax7.transAxes)
        
        ax7.set_xlabel('平均放流量 (m³/s)', fontsize=12)
        ax7.set_ylabel('遅延時間 (分)', fontsize=12)
        ax7.set_title('放流量と遅延時間の関係（放流量≥150m³/s）', fontsize=12, fontweight='bold')
        ax7.grid(True, alpha=0.3)
        ax7.legend()
        
        # 7. 相関係数の比較（3段目右）
        ax8 = axes[2, 2]
        
        corr_means = []
        corr_stds = []
        labels = []
        colors_list = []
        
        for period_type in ['安定期間（イベント前後）', '安定期間（平常状態）']:
            if period_type in self.unified_stable_results:
                df = self.unified_stable_results[period_type]
                if len(df) > 0:
                    corr_means.append(df['correlation'].mean())
                    corr_stds.append(df['correlation'].std())
                    labels.append(f'{period_type}\\n(n={len(df)})')
                    colors_list.append(unified_colors[period_type])
        
        if corr_means:
            x_pos = np.arange(len(labels))
            bars = ax8.bar(x_pos, corr_means, yerr=corr_stds, color=colors_list, 
                           edgecolor='black', capsize=10, alpha=0.7)
            
            ax8.set_xticks(x_pos)
            ax8.set_xticklabels(labels)
            ax8.set_ylabel('平均相関係数')
            ax8.set_title('期間タイプ別の相関係数', fontsize=12)
            ax8.set_ylim(0, 1)
            ax8.grid(True, alpha=0.3)
            
            for bar, mean in zip(bars, corr_means):
                ax8.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'{mean:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        # 保存
        output_path = f"figures/figure5_with_water_level_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"更新されたFigure 5を保存しました: {output_path}")
        
        plt.close()
        
        return fig

def main():
    """単独実行時のみ使用"""
    analyzer = UpdatedFigure5Analysis()
    
    # データ読み込みと分析
    analyzer.load_data()
    analyzer.preprocess_data()
    analyzer.classify_stable_variable_periods()
    analyzer.classify_stable_variable_periods_detailed()
    analyzer.analyze_stable_periods_detailed()
    
    # 安定期間を統合
    analyzer.unify_stable_periods()
    
    # 更新されたFigure 5の作成
    analyzer.visualize_updated_figure5()
    
    print("\n更新されたFigure 5の作成が完了しました。")

if __name__ == "__main__":
    main()