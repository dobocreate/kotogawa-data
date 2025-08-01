#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 12: 変化量の関係性分析（増加期・減少期における応答特性）
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from datetime import datetime
from scipy import stats
# sklearn不要 - NumPyのpolyfitを使用

from analyze_water_level_delay import RiverDelayAnalysis

class Figure12ChangeAmountAnalysis(RiverDelayAnalysis):
    # 親クラスのload_data()メソッドを使用（ダイアログ機能付き）
    
    def analyze_change_amount_relationships(self):
        """変化量の関係性分析"""
        print("\n=== 変化量の関係性分析 ===")
        
        if not hasattr(self, 'variable_periods'):
            print("変動期間が定義されていません。")
            return None
        
        change_analysis_results = []
        
        # 変動期間での分析
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
            
            # フィルタリング条件: 水位≥3m かつ 放流量≥150m³/s
            valid_mask = (water_level_clean >= 3.0) & (discharge_clean >= 150.0)
            if valid_mask.sum() < 6:
                continue
            
            # 変化量を計算（1時間単位）
            for i in range(len(discharge_clean) - 6):
                if valid_mask[i] and valid_mask[i+6]:
                    # 1時間後の変化
                    delta_Q = discharge_clean[i+6] - discharge_clean[i]
                    delta_H = water_level_clean[i+6] - water_level_clean[i]
                    
                    # 初期条件
                    initial_Q = discharge_clean[i]
                    initial_H = water_level_clean[i]
                    
                    # 変化の方向
                    if abs(delta_Q) > 5:  # 有意な変化のみ
                        direction = 'increase' if delta_Q > 0 else 'decrease'
                        
                        # 応答率（ΔH/ΔQ）
                        response_rate = delta_H / delta_Q if abs(delta_Q) > 0 else np.nan
                        
                        # 変化の規模
                        if abs(delta_Q) < 50:
                            magnitude = 'small'
                        elif abs(delta_Q) < 200:
                            magnitude = 'medium'
                        else:
                            magnitude = 'large'
                        
                        # 初期水位レベル
                        if initial_H < 3.5:
                            initial_level = 'low'
                        elif initial_H < 4.5:
                            initial_level = 'medium'
                        else:
                            initial_level = 'high'
                        
                        change_analysis_results.append({
                            'period_idx': period_idx,
                            'time_idx': start + i,
                            'direction': direction,
                            'magnitude': magnitude,
                            'initial_level': initial_level,
                            'initial_Q': initial_Q,
                            'initial_H': initial_H,
                            'delta_Q': delta_Q,
                            'delta_H': delta_H,
                            'response_rate': response_rate,
                            'abs_delta_Q': abs(delta_Q),
                            'abs_delta_H': abs(delta_H)
                        })
        
        self.change_analysis_results = pd.DataFrame(change_analysis_results)
        
        print(f"分析された変化イベント数: {len(change_analysis_results)}件")
        if len(change_analysis_results) > 0:
            print(f"増加イベント: {(self.change_analysis_results['direction'] == 'increase').sum()}件")
            print(f"減少イベント: {(self.change_analysis_results['direction'] == 'decrease').sum()}件")
        
        return self.change_analysis_results
    
    def create_figure12_change_amount_analysis(self):
        """Figure 12: 変化量関係性の可視化"""
        print("\n=== Figure 12: 変化量の関係性分析 ===")
        
        # 分析実行
        if not hasattr(self, 'change_analysis_results'):
            self.analyze_change_amount_relationships()
        
        if len(self.change_analysis_results) == 0:
            print("分析可能なデータがありません。")
            return None
        
        # レイアウトを3行3列に設定
        fig, axes = plt.subplots(3, 3, figsize=(20, 12), num='Figure 12')
        
        # 1. 変化量の直接的な関係（1行目左）
        ax1 = axes[0, 0]
        
        increase_mask = self.change_analysis_results['direction'] == 'increase'
        decrease_mask = self.change_analysis_results['direction'] == 'decrease'
        
        # 増加期
        if increase_mask.sum() > 0:
            ax1.scatter(self.change_analysis_results.loc[increase_mask, 'delta_Q'],
                       self.change_analysis_results.loc[increase_mask, 'delta_H'],
                       c='red', alpha=0.5, s=20, label=f'増加期 (n={increase_mask.sum()})')
            
            # 回帰直線（増加期）
            X_inc = self.change_analysis_results.loc[increase_mask, 'delta_Q'].values
            y_inc = self.change_analysis_results.loc[increase_mask, 'delta_H'].values
            if len(X_inc) > 10:
                # NumPyのpolyfitを使用
                coef_inc = np.polyfit(X_inc, y_inc, 1)
                x_plot = np.linspace(0, X_inc.max(), 100)
                ax1.plot(x_plot, np.polyval(coef_inc, x_plot), 'r--', linewidth=2,
                        label=f'増加期: ΔH = {coef_inc[0]:.4f}ΔQ')
        
        # 減少期
        if decrease_mask.sum() > 0:
            ax1.scatter(self.change_analysis_results.loc[decrease_mask, 'delta_Q'],
                       self.change_analysis_results.loc[decrease_mask, 'delta_H'],
                       c='blue', alpha=0.5, s=20, label=f'減少期 (n={decrease_mask.sum()})')
            
            # 回帰直線（減少期）
            X_dec = self.change_analysis_results.loc[decrease_mask, 'delta_Q'].values
            y_dec = self.change_analysis_results.loc[decrease_mask, 'delta_H'].values
            if len(X_dec) > 10:
                # NumPyのpolyfitを使用
                coef_dec = np.polyfit(X_dec, y_dec, 1)
                x_plot = np.linspace(X_dec.min(), 0, 100)
                ax1.plot(x_plot, np.polyval(coef_dec, x_plot), 'b--', linewidth=2,
                        label=f'減少期: ΔH = {coef_dec[0]:.4f}ΔQ')
        
        ax1.set_xlabel('放流量変化 ΔQ (m³/s)')
        ax1.set_ylabel('水位変化 ΔH (m)')
        ax1.set_title('放流量変化と水位変化の関係')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='k', linewidth=0.5)
        ax1.axvline(x=0, color='k', linewidth=0.5)
        
        # 2. 応答率（ΔH/ΔQ）の分布（1行目中央）
        ax2 = axes[0, 1]
        
        # 応答率の外れ値を除外
        response_rates = self.change_analysis_results['response_rate'].dropna()
        response_rates = response_rates[np.abs(response_rates) < response_rates.abs().quantile(0.95)]
        
        if len(response_rates) > 0:
            # 増加期と減少期で分けて表示
            inc_rates = self.change_analysis_results.loc[increase_mask & (np.abs(self.change_analysis_results['response_rate']) < response_rates.abs().quantile(0.95)), 'response_rate']
            dec_rates = self.change_analysis_results.loc[decrease_mask & (np.abs(self.change_analysis_results['response_rate']) < response_rates.abs().quantile(0.95)), 'response_rate']
            
            bins = np.linspace(-0.01, 0.01, 30)
            
            if len(inc_rates) > 0:
                ax2.hist(inc_rates, bins=bins, alpha=0.5, color='red', label=f'増加期 (平均: {inc_rates.mean():.5f})', density=True)
            if len(dec_rates) > 0:
                ax2.hist(dec_rates, bins=bins, alpha=0.5, color='blue', label=f'減少期 (平均: {dec_rates.mean():.5f})', density=True)
            
            ax2.set_xlabel('応答率 ΔH/ΔQ')
            ax2.set_ylabel('密度')
            ax2.set_title('水位応答率の分布')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # 3. 変化規模別の応答（1行目右）
        ax3 = axes[0, 2]
        
        magnitude_order = ['small', 'medium', 'large']
        magnitude_stats = []
        
        for direction in ['increase', 'decrease']:
            for magnitude in magnitude_order:
                mask = (self.change_analysis_results['direction'] == direction) & \
                       (self.change_analysis_results['magnitude'] == magnitude)
                if mask.sum() > 0:
                    rates = self.change_analysis_results.loc[mask, 'response_rate'].dropna()
                    if len(rates) > 0:
                        magnitude_stats.append({
                            'direction': direction,
                            'magnitude': magnitude,
                            'mean_rate': rates.mean(),
                            'std_rate': rates.std(),
                            'count': len(rates)
                        })
        
        if magnitude_stats:
            df_mag = pd.DataFrame(magnitude_stats)
            
            x_pos = np.arange(len(magnitude_order))
            width = 0.35
            
            # 増加期
            inc_data = df_mag[df_mag['direction'] == 'increase']
            if len(inc_data) > 0:
                inc_means = [inc_data[inc_data['magnitude'] == m]['mean_rate'].values[0] if len(inc_data[inc_data['magnitude'] == m]) > 0 else 0 for m in magnitude_order]
                inc_stds = [inc_data[inc_data['magnitude'] == m]['std_rate'].values[0] if len(inc_data[inc_data['magnitude'] == m]) > 0 else 0 for m in magnitude_order]
                ax3.bar(x_pos - width/2, inc_means, width, yerr=inc_stds, 
                       label='増加期', color='red', alpha=0.7, capsize=5)
            
            # 減少期
            dec_data = df_mag[df_mag['direction'] == 'decrease']
            if len(dec_data) > 0:
                dec_means = [abs(dec_data[dec_data['magnitude'] == m]['mean_rate'].values[0]) if len(dec_data[dec_data['magnitude'] == m]) > 0 else 0 for m in magnitude_order]
                dec_stds = [dec_data[dec_data['magnitude'] == m]['std_rate'].values[0] if len(dec_data[dec_data['magnitude'] == m]) > 0 else 0 for m in magnitude_order]
                ax3.bar(x_pos + width/2, dec_means, width, yerr=dec_stds,
                       label='減少期', color='blue', alpha=0.7, capsize=5)
            
            ax3.set_xticks(x_pos)
            ax3.set_xticklabels(['小\n(<50)', '中\n(50-200)', '大\n(≥200)'])
            ax3.set_xlabel('変化規模 (m³/s)')
            ax3.set_ylabel('平均応答率の絶対値')
            ax3.set_title('変化規模別の水位応答率')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 4. 初期水位レベル別の応答（2行目左）
        ax4 = axes[1, 0]
        
        level_order = ['low', 'medium', 'high']
        level_colors = {'low': 'lightblue', 'medium': 'orange', 'high': 'darkred'}
        
        for direction in ['increase', 'decrease']:
            data_by_level = []
            
            for level in level_order:
                mask = (self.change_analysis_results['direction'] == direction) & \
                       (self.change_analysis_results['initial_level'] == level)
                if mask.sum() > 0:
                    data_by_level.append(self.change_analysis_results.loc[mask, 'response_rate'].dropna().values)
                else:
                    data_by_level.append([])
            
            if any(len(d) > 0 for d in data_by_level):
                positions = np.arange(len(level_order)) + (0 if direction == 'increase' else len(level_order) + 0.5)
                bp = ax4.boxplot([d for d in data_by_level], positions=positions, 
                                widths=0.4, patch_artist=True,
                                labels=level_order if direction == 'increase' else [''] * len(level_order))
                
                # 箱の色設定
                for patch, level in zip(bp['boxes'], level_order):
                    patch.set_facecolor('red' if direction == 'increase' else 'blue')
                    patch.set_alpha(0.5)
        
        ax4.set_xlabel('初期水位レベル')
        ax4.set_ylabel('応答率 ΔH/ΔQ')
        ax4.set_title('初期水位レベル別の応答率分布')
        ax4.grid(True, alpha=0.3)
        
        # カスタム凡例
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='red', alpha=0.5, label='増加期'),
                         Patch(facecolor='blue', alpha=0.5, label='減少期')]
        ax4.legend(handles=legend_elements)
        
        # 5. 非線形性の評価（2行目中央）
        ax5 = axes[1, 1]
        
        # 変化量の大きさと応答率の関係
        valid_mask = ~np.isnan(self.change_analysis_results['response_rate'])
        
        if valid_mask.sum() > 0:
            # 増加期
            inc_mask = increase_mask & valid_mask
            if inc_mask.sum() > 0:
                ax5.scatter(self.change_analysis_results.loc[inc_mask, 'abs_delta_Q'],
                           self.change_analysis_results.loc[inc_mask, 'response_rate'],
                           c='red', alpha=0.3, s=20, label='増加期')
            
            # 減少期（応答率の絶対値）
            dec_mask = decrease_mask & valid_mask
            if dec_mask.sum() > 0:
                ax5.scatter(self.change_analysis_results.loc[dec_mask, 'abs_delta_Q'],
                           np.abs(self.change_analysis_results.loc[dec_mask, 'response_rate']),
                           c='blue', alpha=0.3, s=20, label='減少期')
            
            ax5.set_xlabel('放流量変化の大きさ |ΔQ| (m³/s)')
            ax5.set_ylabel('応答率の絶対値 |ΔH/ΔQ|')
            ax5.set_title('変化量の大きさと応答率の関係（非線形性評価）')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
            ax5.set_xlim(0, 500)
            ax5.set_ylim(0, 0.02)
        
        # 6. 初期放流量と応答率（2行目右）
        ax6 = axes[1, 2]
        
        if valid_mask.sum() > 0:
            # 増加期
            if inc_mask.sum() > 0:
                ax6.scatter(self.change_analysis_results.loc[inc_mask, 'initial_Q'],
                           self.change_analysis_results.loc[inc_mask, 'response_rate'],
                           c='red', alpha=0.3, s=20, label='増加期')
            
            # 減少期
            if dec_mask.sum() > 0:
                ax6.scatter(self.change_analysis_results.loc[dec_mask, 'initial_Q'],
                           self.change_analysis_results.loc[dec_mask, 'response_rate'],
                           c='blue', alpha=0.3, s=20, label='減少期')
            
            ax6.set_xlabel('初期放流量 (m³/s)')
            ax6.set_ylabel('応答率 ΔH/ΔQ')
            ax6.set_title('初期放流量と応答率の関係')
            ax6.legend()
            ax6.grid(True, alpha=0.3)
            ax6.axhline(y=0, color='k', linewidth=0.5)
        
        # 7. 時系列での応答率変化（3行目左）
        ax7 = axes[2, 0]
        
        if len(self.change_analysis_results) > 0:
            # 時系列順にソート
            sorted_data = self.change_analysis_results.sort_values('time_idx')
            
            # 移動平均窓（1日 = 144ポイント）
            window = 144
            
            # 増加期の移動平均
            inc_mask = sorted_data['direction'] == 'increase'
            if inc_mask.sum() > window:
                inc_data = sorted_data[inc_mask]
                inc_ma = inc_data['response_rate'].rolling(window=window, center=True).mean()
                ax7.plot(inc_data.index, inc_ma, 'r-', linewidth=2, label='増加期（移動平均）')
            
            # 減少期の移動平均
            dec_mask = sorted_data['direction'] == 'decrease'
            if dec_mask.sum() > window:
                dec_data = sorted_data[dec_mask]
                dec_ma = dec_data['response_rate'].rolling(window=window, center=True).mean()
                ax7.plot(dec_data.index, dec_ma, 'b-', linewidth=2, label='減少期（移動平均）')
            
            ax7.set_xlabel('データインデックス')
            ax7.set_ylabel('応答率 ΔH/ΔQ')
            ax7.set_title('応答率の時系列変化（1日移動平均）')
            ax7.legend()
            ax7.grid(True, alpha=0.3)
        
        # 8. 累積変化量の関係（3行目中央）
        ax8 = axes[2, 1]
        
        # 各変動期間での累積変化を計算
        cumulative_results = []
        
        for period_idx in self.change_analysis_results['period_idx'].unique():
            period_data = self.change_analysis_results[self.change_analysis_results['period_idx'] == period_idx]
            
            if len(period_data) > 5:
                # 時系列順にソート
                period_data = period_data.sort_values('time_idx')
                
                # 累積変化
                cum_delta_Q = period_data['delta_Q'].cumsum()
                cum_delta_H = period_data['delta_H'].cumsum()
                
                # 全体の方向を判定
                net_direction = 'increase' if cum_delta_Q.iloc[-1] > 0 else 'decrease'
                
                # プロット
                color = 'red' if net_direction == 'increase' else 'blue'
                ax8.plot(cum_delta_Q, cum_delta_H, color=color, alpha=0.5, linewidth=1)
                
                # 最終点をマーク
                ax8.scatter(cum_delta_Q.iloc[-1], cum_delta_H.iloc[-1], 
                          color=color, s=50, edgecolor='black', zorder=5)
        
        ax8.set_xlabel('累積放流量変化 ΣΔQ (m³/s)')
        ax8.set_ylabel('累積水位変化 ΣΔH (m)')
        ax8.set_title('変動期間での累積変化量の関係')
        ax8.grid(True, alpha=0.3)
        ax8.axhline(y=0, color='k', linewidth=0.5)
        ax8.axvline(x=0, color='k', linewidth=0.5)
        
        # 9. 統計サマリー（3行目右）
        ax9 = axes[2, 2]
        
        # サマリーテーブルの作成
        summary_data = []
        
        for direction in ['increase', 'decrease']:
            dir_mask = self.change_analysis_results['direction'] == direction
            if dir_mask.sum() > 0:
                dir_data = self.change_analysis_results[dir_mask]
                
                summary_data.append({
                    '方向': '増加期' if direction == 'increase' else '減少期',
                    'イベント数': dir_mask.sum(),
                    '平均ΔQ': f"{dir_data['delta_Q'].mean():.1f}",
                    '平均ΔH': f"{dir_data['delta_H'].mean():.3f}",
                    '平均応答率': f"{dir_data['response_rate'].mean():.5f}",
                    '応答率標準偏差': f"{dir_data['response_rate'].std():.5f}"
                })
        
        # 変化規模別
        for magnitude in ['small', 'medium', 'large']:
            mag_mask = self.change_analysis_results['magnitude'] == magnitude
            if mag_mask.sum() > 0:
                mag_data = self.change_analysis_results[mag_mask]
                
                summary_data.append({
                    '方向': f'{magnitude}変化',
                    'イベント数': mag_mask.sum(),
                    '平均ΔQ': f"{mag_data['abs_delta_Q'].mean():.1f}",
                    '平均ΔH': f"{mag_data['abs_delta_H'].mean():.3f}",
                    '平均応答率': f"{np.abs(mag_data['response_rate']).mean():.5f}",
                    '応答率標準偏差': '-'
                })
        
        # テーブル表示
        ax9.axis('tight')
        ax9.axis('off')
        
        if summary_data:
            df_summary = pd.DataFrame(summary_data)
            table = ax9.table(cellText=df_summary.values,
                            colLabels=df_summary.columns,
                            cellLoc='center',
                            loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 1.5)
        
        ax9.set_title('統計サマリー', fontsize=12, pad=20)
        
        plt.tight_layout()
        
        # 統計サマリーを表示
        print("\n=== 変化量分析サマリー ===")
        print(f"総イベント数: {len(self.change_analysis_results)}")
        print(f"\n方向別:")
        for direction in ['increase', 'decrease']:
            dir_mask = self.change_analysis_results['direction'] == direction
            if dir_mask.sum() > 0:
                dir_data = self.change_analysis_results[dir_mask]
                print(f"  {direction}:")
                print(f"    イベント数: {dir_mask.sum()}")
                print(f"    平均応答率: {dir_data['response_rate'].mean():.5f}")
                print(f"    応答率範囲: [{dir_data['response_rate'].min():.5f}, {dir_data['response_rate'].max():.5f}]")
        
        # 保存
        output_path = f"figures/figure12_change_amount_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nFigure 12を保存しました: {output_path}")
        
        plt.close()
        
        return fig

def main():
    """単独実行時のみ使用"""
    analyzer = Figure12ChangeAmountAnalysis()
    
    # データ読み込みと分析
    analyzer.load_data()
    analyzer.preprocess_data()
    analyzer.classify_stable_variable_periods()
    
    # Figure 12の作成
    analyzer.create_figure12_change_amount_analysis()
    
    print("\nFigure 12（変化量の関係性分析）の作成が完了しました。")

if __name__ == "__main__":
    main()