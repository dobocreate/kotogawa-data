#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 7: 変動期間バージョンのFigure 5
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from datetime import datetime

from analyze_water_level_delay import RiverDelayAnalysis

class Figure7VariablePeriodsAnalysis(RiverDelayAnalysis):
    # 親クラスのload_data()メソッドを使用（ダイアログ機能付き）
    
    def get_example_periods(self):
        """Figure 7の期間例を取得 - 特異点を優先的に選択"""
        # 変動期間分析が実行されていない場合は実行
        if not hasattr(self, 'variable_analysis_results') or self.variable_analysis_results is None:
            self.analyze_variable_periods()
        
        example_periods = {
            '高遅延期間(140分付近)': None,
            '低遅延期間(10分付近)': None,
            '標準的な変動期': None
        }
        
        # 特異点を持つ期間を探す
        high_delay_period = None  # 140分付近
        low_delay_period = None   # 10分付近
        normal_period = None      # 標準的な期間
        
        # 水位3m以上かつ放流量150m³/s以上のデータから選択
        for idx, row in self.variable_analysis_results.iterrows():
            start, end = int(row['start']), int(row['end'])
            discharge = self.df_processed['ダム_全放流量'].iloc[start:end]
            water_level = self.df_processed['水位_水位'].iloc[start:end]
            
            if len(discharge) > 12 and discharge.mean() >= 150 and water_level.mean() >= 3.0:
                delay_time = row.get('delay_minutes', row.get('cumsum_delay_minutes', None))
                
                if delay_time is not None:
                    # 高遅延期間（130-150分）
                    if 130 <= delay_time <= 150 and high_delay_period is None:
                        high_delay_period = (start, end, idx, delay_time)
                    # 低遅延期間（0-20分）
                    elif 0 <= delay_time <= 20 and low_delay_period is None:
                        low_delay_period = (start, end, idx, delay_time)
                    # 標準的な期間（40-80分）
                    elif 40 <= delay_time <= 80 and normal_period is None:
                        normal_period = (start, end, idx, delay_time)
        
        # 見つかった期間を割り当て
        if high_delay_period:
            example_periods['高遅延期間(140分付近)'] = high_delay_period[:3]
            print(f"高遅延期間: delay={high_delay_period[3]:.1f}分, start={high_delay_period[0]}, end={high_delay_period[1]}")
        
        if low_delay_period:
            example_periods['低遅延期間(10分付近)'] = low_delay_period[:3]
            print(f"低遅延期間: delay={low_delay_period[3]:.1f}分, start={low_delay_period[0]}, end={low_delay_period[1]}")
        
        if normal_period:
            example_periods['標準的な変動期'] = normal_period[:3]
            print(f"標準的な変動期: delay={normal_period[3]:.1f}分, start={normal_period[0]}, end={normal_period[1]}")
        
        # 見つからなかった場合は従来の方法で選択
        if not all(example_periods.values()):
            print("\n特異点を持つ期間が不足しています。従来の方法で補完します。")
            for idx, row in self.variable_analysis_results.iterrows():
                start, end = int(row['start']), int(row['end'])
                discharge = self.df_processed['ダム_全放流量'].iloc[start:end]
                
                if len(discharge) > 12 and discharge.mean() >= 150:
                    trend = discharge.iloc[-1] - discharge.iloc[0]
                    variation = discharge.std() / discharge.mean() if discharge.mean() > 0 else 0
                    
                    if trend > 50 and example_periods['高遅延期間(140分付近)'] is None:
                        example_periods['高遅延期間(140分付近)'] = (start, end, idx)
                    elif trend < -50 and example_periods['低遅延期間(10分付近)'] is None:
                        example_periods['低遅延期間(10分付近)'] = (start, end, idx)
                    elif variation > 0.3 and example_periods['標準的な変動期'] is None:
                        example_periods['標準的な変動期'] = (start, end, idx)
        
        return example_periods
    
    def visualize_figure7_variable_periods(self):
        """Figure 7: 変動期間の包括的視覚化"""
        print("\n=== Figure 7: 変動期間の視覚化 ===")
        
        # 変動期間分析が実行されていない場合は実行
        if not hasattr(self, 'variable_analysis_results') or self.variable_analysis_results is None:
            self.analyze_variable_periods()
        
        # レイアウトを3行3列に設定
        fig, axes = plt.subplots(3, 3, figsize=(20, 12), num='Figure 7')
        
        # 色の設定
        variable_colors = {
            '流量増加期': 'lightcoral',
            '流量減少期': 'lightblue',
            '複雑な変動期': 'lightyellow'
        }
        
        # 1. 変動期間の時系列例（上部3つ）- 特異点を優先表示
        example_periods = self.get_example_periods()
        
        period_names = ['高遅延期間(140分付近)', '低遅延期間(10分付近)', '標準的な変動期']
        
        for i, period_name in enumerate(period_names):
            ax = axes[0, i]
            
            if example_periods[period_name] is not None:
                start, end, period_idx = example_periods[period_name]
                display_start = max(0, start - 18)
                display_end = min(end + 18, len(self.df_processed))
                
                time_hours = np.arange(display_end - display_start) / 6
                discharge = self.df_processed['ダム_全放流量'].iloc[display_start:display_end]
                water_level = self.df_processed['水位_水位'].iloc[display_start:display_end]
                
                # 変動期間の範囲を色づけ（特異点は赤系、通常はピンク系）
                if '140分' in period_name:
                    color = 'lightcoral'
                elif '10分' in period_name:
                    color = 'lightblue'
                else:
                    color = 'lightpink'
                
                var_start_rel = (start - display_start) / 6
                var_end_rel = (end - display_start) / 6
                
                ax.axvspan(var_start_rel, var_end_rel, alpha=0.3, color=color, 
                          label='変動期間' if i == 0 else '')
                
                ax2 = ax.twinx()
                line1 = ax.plot(time_hours, discharge, 'b-', linewidth=2, label='放流量')
                line2 = ax2.plot(time_hours, water_level, 'r-', linewidth=2, label='水位')
                
                # 遅延時間も表示
                delay_time = self.variable_analysis_results.loc[period_idx, 'delay_minutes']
                ax.text(0.02, 0.98, f'遅延時間: {delay_time:.1f}分', 
                       transform=ax.transAxes, va='top', ha='left',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                ax.set_xlabel('時間 (hours)')
                ax.set_ylabel('放流量 (m³/s)', color='b')
                ax2.set_ylabel('水位 (m)', color='r')
                ax.set_title(period_name, fontsize=12)
                ax.grid(True, alpha=0.3)
                
                if i == 0:
                    lines = line1 + line2
                    labels = [l.get_label() for l in lines]
                    ax.legend(lines, labels, loc='upper left')
            else:
                ax.text(0.5, 0.5, 'データなし', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(period_name, fontsize=12)
        
        # 2. 変動期間での放流量-水位関係（2段目左）
        ax3 = axes[1, 0]
        
        # フィルタリングなし（全データを使用）
        filtered_variable_data = self.variable_analysis_results
        
        discharge_values = []
        water_level_values = []
        period_directions = []
        
        for idx, row in filtered_variable_data.iterrows():
            start, end = int(row['start']), int(row['end'])
            discharge = self.df_processed['ダム_全放流量'].iloc[start:end]
            water_level = self.df_processed['水位_水位'].iloc[start:end]
            
            discharge_values.append(discharge.mean())
            water_level_values.append(water_level.mean())
            
            # 方向を判定
            if row['direction'] == 'increase':
                period_directions.append('増加期')
            elif row['direction'] == 'decrease':
                period_directions.append('減少期')
            else:
                period_directions.append('その他')
        
        if discharge_values:
            colors_map = {'増加期': 'lightcoral', '減少期': 'lightblue', 'その他': 'lightyellow'}
            
            for direction in colors_map:
                mask = [d == direction for d in period_directions]
                d_subset = [d for d, m in zip(discharge_values, mask) if m]
                w_subset = [w for w, m in zip(water_level_values, mask) if m]
                
                if d_subset:
                    ax3.scatter(d_subset, w_subset, c=colors_map[direction], s=60, alpha=0.7,
                              edgecolors='black', label=f'{direction} (n={len(d_subset)})')
        
        ax3.set_xlabel('平均放流量 (m³/s)')
        ax3.set_ylabel('平均水位 (m)')
        ax3.set_title('変動期間での放流量-水位関係', fontsize=12)
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # 3. 変動期間の放流量範囲別データ数（2段目中央）
        ax4 = axes[1, 1]
        
        range_labels = ['0-150', '150-200', '200-300', '300-400', '400-500', '500+']
        discharge_ranges = [(0, 150), (150, 200), (200, 300), (300, 400), (400, 500), (500, 1000)]
        
        increase_counts = []
        decrease_counts = []
        
        for min_d, max_d in discharge_ranges:
            inc_count = len(filtered_variable_data[
                (filtered_variable_data['avg_discharge'] >= min_d) & 
                (filtered_variable_data['avg_discharge'] < max_d) &
                (filtered_variable_data['direction'] == 'increase')
            ])
            dec_count = len(filtered_variable_data[
                (filtered_variable_data['avg_discharge'] >= min_d) & 
                (filtered_variable_data['avg_discharge'] < max_d) &
                (filtered_variable_data['direction'] == 'decrease')
            ])
            
            increase_counts.append(inc_count)
            decrease_counts.append(dec_count)
        
        x_pos = np.arange(len(range_labels))
        width = 0.35
        
        bars1 = ax4.bar(x_pos - width/2, increase_counts, width,
                       label='増加期', color='lightcoral', alpha=0.7, edgecolor='black')
        bars2 = ax4.bar(x_pos + width/2, decrease_counts, width,
                       label='減少期', color='lightblue', alpha=0.7, edgecolor='black')
        
        # データ数を表示
        for i, (inc, dec) in enumerate(zip(increase_counts, decrease_counts)):
            if inc > 0:
                ax4.text(i - width/2, inc + 0.5, str(inc), 
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
            if dec > 0:
                ax4.text(i + width/2, dec + 0.5, str(dec), 
                        ha='center', va='bottom', fontsize=10)
        
        ax4.set_xlabel('放流量範囲 (m³/s)')
        ax4.set_ylabel('期間数')
        ax4.set_title('放流量範囲別の変動期間データ数', fontsize=12)
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(range_labels)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 4. 変動期間の変動係数分布（2段目右）
        ax5 = axes[1, 2]
        
        cv_increase = []
        cv_decrease = []
        
        for idx, row in filtered_variable_data.iterrows():
            start, end = int(row['start']), int(row['end'])
            discharge = self.df_processed['ダム_全放流量'].iloc[start:end]
            
            if discharge.mean() > 0:
                cv = discharge.std() / discharge.mean()
                if row['direction'] == 'increase':
                    cv_increase.append(cv)
                elif row['direction'] == 'decrease':
                    cv_decrease.append(cv)
        
        if cv_increase:
            ax5.hist(cv_increase, bins=15, alpha=0.7, label=f'増加期 (n={len(cv_increase)})',
                    color='lightcoral', edgecolor='black')
        if cv_decrease:
            ax5.hist(cv_decrease, bins=15, alpha=0.7, label=f'減少期 (n={len(cv_decrease)})',
                    color='lightblue', edgecolor='black')
        
        ax5.axvline(x=0.1, color='orange', linestyle='--', label='CV=10%')
        ax5.axvline(x=0.2, color='red', linestyle='--', label='CV=20%')
        ax5.set_xlabel('変動係数 (CV)')
        ax5.set_ylabel('期間数')
        ax5.set_title('変動期間の変動係数分布', fontsize=12)
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 5. 変動期間での水位と遅延時間の関係（3段目左）
        ax6 = axes[2, 0]
        
        water_levels_var = []
        delay_times = []
        directions = []
        
        for idx, row in filtered_variable_data.iterrows():
            if pd.notna(row.get('delay_minutes', row.get('cumsum_delay_minutes'))):
                start, end = int(row['start']), int(row['end'])
                water_level = self.df_processed['水位_水位'].iloc[start:end].mean()
                
                # 水位3m以上のデータのみ収集
                if water_level >= 3.0:
                    water_levels_var.append(water_level)
                    # delay_minutesがあればそれを使用、なければcumsum_delay_minutesを使用
                    delay_time = row.get('delay_minutes', row.get('cumsum_delay_minutes'))
                    delay_times.append(delay_time)
                    directions.append(row['direction'])
        
        if water_levels_var:
            # 遅延時間と水位の関係
            colors_map = {'increase': 'lightcoral', 'decrease': 'lightblue'}
            direction_labels = {'increase': '増加期', 'decrease': '減少期'}
            
            for direction in colors_map:
                mask = [d == direction for d in directions]
                wl_subset = [wl for wl, m in zip(water_levels_var, mask) if m]
                dt_subset = [dt for dt, m in zip(delay_times, mask) if m]
                
                if wl_subset:
                    ax6.scatter(wl_subset, dt_subset, c=colors_map[direction], s=60, alpha=0.7,
                              edgecolors='black', label=f'{direction_labels[direction]} (n={len(wl_subset)})')
            
            # 水位3m以上のトレンドライン
            if len(water_levels_var) >= 3:
                z = np.polyfit(water_levels_var, delay_times, 2)
                p = np.poly1d(z)
                x_trend = np.linspace(min(water_levels_var), max(water_levels_var), 100)
                ax6.plot(x_trend, p(x_trend), 'r-', linewidth=2, label='トレンド（2次多項式, 水位≥3m）')
            
            # 軸の範囲をデータに合わせて調整
            ax6.set_xlim(min(water_levels_var) - 0.1, max(water_levels_var) + 0.1)
            ax6.set_ylim(min(delay_times) - 10, max(delay_times) + 10)
        
        ax6.set_xlabel('平均水位 (m)')
        ax6.set_ylabel('遅延時間 (分)')
        ax6.set_title('変動期間での水位と遅延時間の関係（水位≥3m）', fontsize=12)
        ax6.grid(True, alpha=0.3)
        ax6.legend()
        
        # 6. 変動期間での放流量と遅延時間の関係（3段目中央）
        ax7 = axes[2, 1]
        
        discharge_levels_var = []
        delay_times_2 = []
        directions_2 = []
        
        for idx, row in filtered_variable_data.iterrows():
            if pd.notna(row.get('delay_minutes', row.get('cumsum_delay_minutes'))):
                # 放流量150m³/s以上のデータのみ収集
                if row['avg_discharge'] >= 150:
                    discharge_levels_var.append(row['avg_discharge'])
                    # delay_minutesがあればそれを使用、なければcumsum_delay_minutesを使用
                    delay_time = row.get('delay_minutes', row.get('cumsum_delay_minutes'))
                    delay_times_2.append(delay_time)
                    directions_2.append(row['direction'])
        
        if discharge_levels_var:
            # 遅延時間と放流量の関係
            for direction in colors_map:
                mask = [d == direction for d in directions_2]
                dl_subset = [dl for dl, m in zip(discharge_levels_var, mask) if m]
                dt_subset = [dt for dt, m in zip(delay_times_2, mask) if m]
                
                if dl_subset:
                    ax7.scatter(dl_subset, dt_subset, c=colors_map[direction], s=60, alpha=0.7,
                              edgecolors='black', label=f'{direction_labels[direction]} (n={len(dl_subset)})')
            
            # 放流量150m³/s以上のトレンドライン
            if len(discharge_levels_var) >= 3:
                z = np.polyfit(discharge_levels_var, delay_times_2, 2)
                p = np.poly1d(z)
                x_trend = np.linspace(min(discharge_levels_var), max(discharge_levels_var), 100)
                ax7.plot(x_trend, p(x_trend), 'g-', linewidth=2, label='トレンド（2次多項式, 放流量≥150）')
            
            # 軸の範囲をデータに合わせて調整
            ax7.set_xlim(min(discharge_levels_var) - 20, max(discharge_levels_var) + 20)
            ax7.set_ylim(min(delay_times_2) - 10, max(delay_times_2) + 10)
        
        ax7.set_xlabel('平均放流量 (m³/s)')
        ax7.set_ylabel('遅延時間 (分)')
        ax7.set_title('変動期間での放流量と遅延時間の関係（放流量≥150m³/s）', fontsize=12)
        ax7.grid(True, alpha=0.3)
        ax7.legend()
        
        # 7. 相関係数の分布（3段目右）
        ax8 = axes[2, 2]
        
        # 相関係数を収集
        correlations_increase = []
        correlations_decrease = []
        correlations_other = []
        
        for idx, row in filtered_variable_data.iterrows():
            if pd.notna(row.get('max_correlation', None)):
                if row['direction'] == 'increase':
                    correlations_increase.append(row['max_correlation'])
                elif row['direction'] == 'decrease':
                    correlations_decrease.append(row['max_correlation'])
                else:
                    correlations_other.append(row['max_correlation'])
        
        # ヒストグラムを描画
        bins = np.linspace(-1, 1, 21)  # -1から1まで20区間
        
        if correlations_increase:
            ax8.hist(correlations_increase, bins=bins, alpha=0.7, 
                    label=f'増加期 (n={len(correlations_increase)})',
                    color='lightcoral', edgecolor='black')
        
        if correlations_decrease:
            ax8.hist(correlations_decrease, bins=bins, alpha=0.7, 
                    label=f'減少期 (n={len(correlations_decrease)})',
                    color='lightblue', edgecolor='black')
        
        if correlations_other:
            ax8.hist(correlations_other, bins=bins, alpha=0.7, 
                    label=f'その他 (n={len(correlations_other)})',
                    color='lightyellow', edgecolor='black')
        
        # 全体の平均値線を追加
        all_correlations = correlations_increase + correlations_decrease + correlations_other
        if all_correlations:
            mean_corr = np.mean(all_correlations)
            ax8.axvline(x=mean_corr, color='red', linestyle='--', linewidth=2,
                       label=f'平均: {mean_corr:.3f}')
        
        ax8.set_xlabel('相関係数')
        ax8.set_ylabel('期間数')
        ax8.set_title('相関係数の分布', fontsize=12)
        ax8.set_xlim(-1, 1)
        ax8.legend()
        ax8.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存
        output_path = f"figures/figure7_variable_periods_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Figure 7を保存しました: {output_path}")
        
        plt.close()
        
        return fig

def main():
    """単独実行時のみ使用"""
    analyzer = Figure7VariablePeriodsAnalysis()
    
    # データ読み込みと分析
    analyzer.load_data()
    analyzer.preprocess_data()
    analyzer.classify_stable_variable_periods()
    analyzer.analyze_variable_periods()
    
    # Figure 7の作成
    analyzer.visualize_figure7_variable_periods()
    
    print("\nFigure 7（変動期間）の作成が完了しました。")

if __name__ == "__main__":
    main()