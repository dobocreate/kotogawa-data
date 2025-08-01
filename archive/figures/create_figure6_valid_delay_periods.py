#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 6: 妥当な遅延時間範囲（60分未満）での期間分析
水位3-4m、遅延時間60分未満の条件での時系列表示
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from datetime import datetime

from analyze_water_level_delay import RiverDelayAnalysis

class Figure6ValidDelayAnalysis(RiverDelayAnalysis):
    # 親クラスのload_data()メソッドを使用（ダイアログ機能付き）
    
    def create_figure6_valid_delays(self):
        """Figure 6: 妥当な遅延時間範囲での可視化"""
        print("\n=== Figure 6: 妥当な遅延時間範囲（60分未満）の可視化 ===")
        
        # レイアウトを3行3列に設定
        fig, axes = plt.subplots(3, 3, figsize=(20, 12), num='Figure 6')
        
        # 1. 条件に合う期間の抽出（遅延時間<60分、水位3-4m）
        valid_periods = []
        
        print("\n=== 妥当な遅延時間の期間探索（水位: 3-4m、遅延: <60分） ===")
        for period_type in ['pre_stable', 'post_stable', 'low_activity']:
            if period_type in self.detailed_stable_results:
                df = self.detailed_stable_results[period_type]
                valid_count = 0
                
                for idx, row in df.iterrows():
                    delay_minutes = row['delay_minutes']
                    start, end = int(row['start']), int(row['end'])
                    water_level = self.df_processed['水位_水位'].iloc[start:end].mean()
                    discharge = self.df_processed['ダム_全放流量'].iloc[start:end].mean()
                    
                    # 条件: 水位3-4m かつ 遅延時間60分未満
                    if 3.0 <= water_level <= 4.0 and delay_minutes < 60:
                        valid_periods.append({
                            'start': start,
                            'end': end,
                            'delay_minutes': delay_minutes,
                            'period_type': period_type,
                            'water_level': water_level,
                            'discharge': discharge,
                            'correlation': row['correlation']
                        })
                        valid_count += 1
                
                print(f"{period_type}: {valid_count}期間（条件適合）")
        
        # 遅延時間でソート（短い順）
        valid_periods = sorted(valid_periods, key=lambda x: x['delay_minutes'])
        
        # 2. 時系列表示（1行目：最も遅延が短い3つ）
        for i in range(3):
            ax = axes[0, i]
            
            if i < len(valid_periods):
                period = valid_periods[i]
                period_start = period['start']
                display_start = max(0, period_start - 36)
                display_end = min(period['end'] + 36, len(self.df_processed))
                
                time_hours = np.arange(display_end - display_start) / 6
                discharge = self.df_processed['ダム_全放流量'].iloc[display_start:display_end]
                water_level = self.df_processed['水位_水位'].iloc[display_start:display_end]
                
                # 期間の範囲を色づけ
                stable_start_rel = (period['start'] - display_start) / 6
                stable_end_rel = (period['end'] - display_start) / 6
                
                ax.axvspan(stable_start_rel, stable_end_rel, alpha=0.3, color='lightgreen', 
                          label='分析期間' if i == 0 else '')
                
                ax2 = ax.twinx()
                line1 = ax.plot(time_hours, discharge, 'b-', linewidth=2, label='放流量')
                line2 = ax2.plot(time_hours, water_level, 'r-', linewidth=2, label='水位')
                
                # 遅延時間と水位情報を表示
                info_text = (f'遅延時間: {period["delay_minutes"]:.1f}分\n'
                           f'平均水位: {period["water_level"]:.2f}m\n'
                           f'相関係数: {period["correlation"]:.3f}')
                ax.text(0.02, 0.98, info_text, 
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
                ax.set_title(f'短遅延期間 {i+1} ({period_type_label[period["period_type"]]})', fontsize=12)
                ax.grid(True, alpha=0.3)
                
                if i == 0:
                    lines = line1 + line2
                    labels = [l.get_label() for l in lines]
                    ax.legend(lines, labels, loc='upper left')
            else:
                ax.text(0.5, 0.5, 'データなし', ha='center', va='center', transform=ax.transAxes)
        
        # 3. 時系列表示（2行目：中間的な遅延時間の3つ）
        middle_start = len(valid_periods) // 2 - 1 if len(valid_periods) > 3 else 3
        for i in range(3):
            ax = axes[1, i]
            idx = middle_start + i
            
            if idx < len(valid_periods):
                period = valid_periods[idx]
                period_start = period['start']
                display_start = max(0, period_start - 36)
                display_end = min(period['end'] + 36, len(self.df_processed))
                
                time_hours = np.arange(display_end - display_start) / 6
                discharge = self.df_processed['ダム_全放流量'].iloc[display_start:display_end]
                water_level = self.df_processed['水位_水位'].iloc[display_start:display_end]
                
                # 期間の範囲を色づけ
                stable_start_rel = (period['start'] - display_start) / 6
                stable_end_rel = (period['end'] - display_start) / 6
                
                ax.axvspan(stable_start_rel, stable_end_rel, alpha=0.3, color='lightyellow')
                
                ax2 = ax.twinx()
                line1 = ax.plot(time_hours, discharge, 'b-', linewidth=2, label='放流量')
                line2 = ax2.plot(time_hours, water_level, 'r-', linewidth=2, label='水位')
                
                # 遅延時間と水位情報を表示
                info_text = (f'遅延時間: {period["delay_minutes"]:.1f}分\n'
                           f'平均水位: {period["water_level"]:.2f}m\n'
                           f'相関係数: {period["correlation"]:.3f}')
                ax.text(0.02, 0.98, info_text, 
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
                ax.set_title(f'中遅延期間 {i+1} ({period_type_label[period["period_type"]]})', fontsize=12)
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'データなし', ha='center', va='center', transform=ax.transAxes)
        
        # 4. 時系列表示（3行目：最も遅延が長い3つ、ただし60分未満）
        for i in range(3):
            ax = axes[2, i]
            idx = len(valid_periods) - 3 + i
            
            if idx >= 0 and idx < len(valid_periods):
                period = valid_periods[idx]
                period_start = period['start']
                display_start = max(0, period_start - 36)
                display_end = min(period['end'] + 36, len(self.df_processed))
                
                time_hours = np.arange(display_end - display_start) / 6
                discharge = self.df_processed['ダム_全放流量'].iloc[display_start:display_end]
                water_level = self.df_processed['水位_水位'].iloc[display_start:display_end]
                
                # 期間の範囲を色づけ
                stable_start_rel = (period['start'] - display_start) / 6
                stable_end_rel = (period['end'] - display_start) / 6
                
                ax.axvspan(stable_start_rel, stable_end_rel, alpha=0.3, color='lightcoral')
                
                ax2 = ax.twinx()
                line1 = ax.plot(time_hours, discharge, 'b-', linewidth=2, label='放流量')
                line2 = ax2.plot(time_hours, water_level, 'r-', linewidth=2, label='水位')
                
                # 遅延時間と水位情報を表示
                info_text = (f'遅延時間: {period["delay_minutes"]:.1f}分\n'
                           f'平均水位: {period["water_level"]:.2f}m\n'
                           f'相関係数: {period["correlation"]:.3f}')
                ax.text(0.02, 0.98, info_text, 
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
                ax.set_title(f'長遅延期間 {i+1} ({period_type_label[period["period_type"]]})', fontsize=12)
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'データなし', ha='center', va='center', transform=ax.transAxes)
        
        plt.tight_layout()
        
        # 統計情報を表示
        if valid_periods:
            delays = [p['delay_minutes'] for p in valid_periods]
            water_levels = [p['water_level'] for p in valid_periods]
            correlations = [p['correlation'] for p in valid_periods]
            
            print(f"\n=== 統計情報（総数: {len(valid_periods)}期間） ===")
            print(f"遅延時間: 平均{np.mean(delays):.1f}分、中央値{np.median(delays):.1f}分、範囲{min(delays):.1f}-{max(delays):.1f}分")
            print(f"水位: 平均{np.mean(water_levels):.2f}m、範囲{min(water_levels):.2f}-{max(water_levels):.2f}m")
            print(f"相関係数: 平均{np.mean(correlations):.3f}、範囲{min(correlations):.3f}-{max(correlations):.3f}")
        
        # 保存
        output_path = f"figures/figure6_valid_delay_periods_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nFigure 6を保存しました: {output_path}")
        
        plt.close()
        
        return fig

def main():
    """単独実行時のみ使用"""
    analyzer = Figure6ValidDelayAnalysis()
    
    # データ読み込みと分析
    analyzer.load_data()
    analyzer.preprocess_data()
    analyzer.classify_stable_variable_periods()
    analyzer.classify_stable_variable_periods_detailed()
    analyzer.analyze_stable_periods_detailed()
    
    # Figure 6の作成
    analyzer.create_figure6_valid_delays()
    
    print("\nFigure 6（妥当な遅延時間範囲）の作成が完了しました。")

if __name__ == "__main__":
    main()