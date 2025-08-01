#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 9: 変動期間の9つの例を3x3グリッドで表示
Figure 7の一行目と同様の形式で、変動期間の代表例を表示
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from datetime import datetime

from analyze_water_level_delay import RiverDelayAnalysis

class Figure9VariablePeriodsGrid(RiverDelayAnalysis):
    # 親クラスのload_data()メソッドを使用（ダイアログ機能付き）
    
    def get_nine_variable_periods(self):
        """9つの代表的な変動期間を選択"""
        # 変動期間分析が実行されていない場合は実行
        if not hasattr(self, 'variable_analysis_results') or self.variable_analysis_results is None:
            self.analyze_variable_periods()
        
        selected_periods = []
        
        # 放流量150m³/s以上、2時間以上の期間を候補とする
        candidates = []
        for idx, row in self.variable_analysis_results.iterrows():
            start, end = int(row['start']), int(row['end'])
            discharge = self.df_processed['ダム_全放流量'].iloc[start:end]
            
            if len(discharge) > 12 and discharge.mean() >= 150:  # 2時間以上かつ放流量150m³/s以上
                trend = discharge.iloc[-1] - discharge.iloc[0]
                variation = discharge.std() / discharge.mean() if discharge.mean() > 0 else 0
                
                candidates.append({
                    'start': start,
                    'end': end,
                    'idx': idx,
                    'trend': trend,
                    'variation': variation,
                    'avg_discharge': discharge.mean(),
                    'duration': len(discharge)
                })
        
        # カテゴリ別に選択
        # 1. 流量増加期（trend > 50）から3つ
        increase_periods = sorted([c for c in candidates if c['trend'] > 50], 
                                key=lambda x: x['trend'], reverse=True)[:3]
        
        # 2. 流量減少期（trend < -50）から3つ
        decrease_periods = sorted([c for c in candidates if c['trend'] < -50], 
                                key=lambda x: x['trend'])[:3]
        
        # 3. 複雑な変動期（variation > 0.2）から3つ
        complex_periods = sorted([c for c in candidates if c['variation'] > 0.2], 
                               key=lambda x: x['variation'], reverse=True)[:3]
        
        # 組み合わせて9つにする
        selected_periods = increase_periods + decrease_periods + complex_periods
        
        # 9つに満たない場合は、残りの候補から補充
        if len(selected_periods) < 9:
            used_indices = {p['idx'] for p in selected_periods}
            remaining = [c for c in candidates if c['idx'] not in used_indices]
            # 平均放流量が大きい順に選択
            remaining_sorted = sorted(remaining, key=lambda x: x['avg_discharge'], reverse=True)
            selected_periods.extend(remaining_sorted[:9-len(selected_periods)])
        
        print(f"選択された変動期間数: {len(selected_periods)}")
        
        return selected_periods[:9]  # 最大9つまで
    
    def visualize_figure9_variable_periods_grid(self):
        """Figure 9: 変動期間の9つの例を表示"""
        print("\n=== Figure 9: 変動期間グリッドの視覚化 ===")
        
        # 9つの変動期間を取得
        selected_periods = self.get_nine_variable_periods()
        
        if not selected_periods:
            print("表示可能な変動期間がありません")
            return None
        
        # 3x3のグリッドレイアウト
        fig, axes = plt.subplots(3, 3, figsize=(20, 12), num='Figure 9')
        axes = axes.flatten()
        
        for i in range(9):
            ax = axes[i]
            
            if i < len(selected_periods):
                period = selected_periods[i]
                start, end = period['start'], period['end']
                
                # 表示範囲を前後3時間含めて設定
                display_start = max(0, start - 18)
                display_end = min(end + 18, len(self.df_processed))
                
                time_hours = np.arange(display_end - display_start) / 6
                discharge = self.df_processed['ダム_全放流量'].iloc[display_start:display_end]
                water_level = self.df_processed['水位_水位'].iloc[display_start:display_end]
                
                # 変動期間の範囲をピンク系で色づけ
                var_start_rel = (start - display_start) / 6
                var_end_rel = (end - display_start) / 6
                
                ax.axvspan(var_start_rel, var_end_rel, alpha=0.3, color='lightpink', 
                          label='変動期間' if i == 0 else '')
                
                # 放流量と水位をプロット
                ax2 = ax.twinx()
                line1 = ax.plot(time_hours, discharge, 'b-', linewidth=2, label='放流量')
                line2 = ax2.plot(time_hours, water_level, 'r-', linewidth=2, label='水位')
                
                ax.set_xlabel('時間 (hours)')
                ax.set_ylabel('放流量 (m³/s)', color='b')
                ax2.set_ylabel('水位 (m)', color='r')
                
                # タイトルに期間の特徴を表示
                if period['trend'] > 50:
                    period_type = '流量増加期'
                elif period['trend'] < -50:
                    period_type = '流量減少期'
                else:
                    period_type = '複雑な変動期'
                
                title = f'{period_type} (例{i+1})\n'
                title += f'平均放流量: {period["avg_discharge"]:.0f} m³/s, '
                title += f'継続時間: {period["duration"]/6:.1f}時間'
                ax.set_title(title, fontsize=10)
                
                ax.grid(True, alpha=0.3)
                
                # 最初のグラフにのみ凡例を表示
                if i == 0:
                    lines = line1 + line2
                    labels = [l.get_label() for l in lines]
                    ax.legend(lines, labels, loc='upper left')
                
                # Y軸の範囲を調整
                ax.tick_params(axis='y', colors='b')
                ax2.tick_params(axis='y', colors='r')
                
            else:
                # データがない場合
                ax.text(0.5, 0.5, 'データなし', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=14)
                ax.set_xticks([])
                ax.set_yticks([])
        
        plt.tight_layout()
        
        # 保存
        output_path = f"figures/figure9_variable_periods_grid_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Figure 9を保存しました: {output_path}")
        
        plt.close()
        
        return fig

def main():
    """単独実行時のみ使用"""
    analyzer = Figure9VariablePeriodsGrid()
    
    # データ読み込みと分析
    analyzer.load_data()
    analyzer.preprocess_data()
    analyzer.classify_stable_variable_periods()
    analyzer.analyze_variable_periods()
    
    # Figure 9の作成
    analyzer.visualize_figure9_variable_periods_grid()
    
    print("\nFigure 9（変動期間グリッド）の作成が完了しました。")

if __name__ == "__main__":
    main()