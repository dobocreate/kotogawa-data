#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 11: 水位上昇期・下降期の応答特性分析（ヒステリシス分析）
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from datetime import datetime
from scipy import stats

from analyze_water_level_delay import RiverDelayAnalysis

class Figure11HysteresisAnalysis(RiverDelayAnalysis):
    # 親クラスのload_data()メソッドを使用（ダイアログ機能付き）
    
    def analyze_hysteresis_effects(self):
        """上昇期・下降期の応答特性分析"""
        print("\n=== ヒステリシス効果の分析 ===")
        
        if not hasattr(self, 'variable_periods'):
            print("変動期間が定義されていません。")
            return None
        
        hysteresis_results = []
        
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
            
            # 変化率を計算（前後の差分）
            dQ_dt = np.gradient(discharge_clean)
            dH_dt = np.gradient(water_level_clean)
            
            # 上昇期と下降期を分類
            rising_mask = (dQ_dt > 0) & valid_mask
            falling_mask = (dQ_dt < 0) & valid_mask
            
            # 各期の分析
            if rising_mask.sum() >= 3 and falling_mask.sum() >= 3:
                # 上昇期の回帰分析
                rising_Q = discharge_clean[rising_mask]
                rising_H = water_level_clean[rising_mask]
                if len(rising_Q) >= 3:
                    slope_rise, intercept_rise, r_rise, _, _ = stats.linregress(rising_Q, rising_H)
                    
                    # 下降期の回帰分析
                    falling_Q = discharge_clean[falling_mask]
                    falling_H = water_level_clean[falling_mask]
                    slope_fall, intercept_fall, r_fall, _, _ = stats.linregress(falling_Q, falling_H)
                    
                    # ヒステリシスループの面積（簡易計算）
                    # 同じ放流量での水位差の積分として近似
                    Q_common = np.linspace(
                        max(rising_Q.min(), falling_Q.min()),
                        min(rising_Q.max(), falling_Q.max()),
                        50
                    )
                    H_rise_pred = slope_rise * Q_common + intercept_rise
                    H_fall_pred = slope_fall * Q_common + intercept_fall
                    hysteresis_area = np.trapz(np.abs(H_rise_pred - H_fall_pred), Q_common)
                    
                    hysteresis_results.append({
                        'period_idx': period_idx,
                        'start': start,
                        'end': end,
                        'slope_rise': slope_rise,
                        'intercept_rise': intercept_rise,
                        'r_rise': r_rise,
                        'slope_fall': slope_fall,
                        'intercept_fall': intercept_fall,
                        'r_fall': r_fall,
                        'slope_ratio': slope_rise / slope_fall if slope_fall != 0 else np.nan,
                        'hysteresis_area': hysteresis_area,
                        'n_rising': rising_mask.sum(),
                        'n_falling': falling_mask.sum(),
                        'mean_discharge': discharge_clean[valid_mask].mean(),
                        'mean_water_level': water_level_clean[valid_mask].mean(),
                        'discharge_range': discharge_clean[valid_mask].max() - discharge_clean[valid_mask].min()
                    })
        
        self.hysteresis_results = pd.DataFrame(hysteresis_results)
        
        print(f"分析された変動期間数: {len(hysteresis_results)}件")
        if len(hysteresis_results) > 0:
            print(f"平均傾き比（上昇/下降）: {self.hysteresis_results['slope_ratio'].mean():.3f}")
            print(f"平均ヒステリシス面積: {self.hysteresis_results['hysteresis_area'].mean():.2f}")
        
        return self.hysteresis_results
    
    def create_figure11_hysteresis_analysis(self):
        """Figure 11: ヒステリシス効果の可視化"""
        print("\n=== Figure 11: ヒステリシス効果分析 ===")
        
        # 分析実行
        if not hasattr(self, 'hysteresis_results'):
            self.analyze_hysteresis_effects()
        
        # レイアウトを3行3列に設定
        fig, axes = plt.subplots(3, 3, figsize=(20, 12), num='Figure 11')
        
        # 1. 代表的なヒステリシスループ（1行目）
        # ヒステリシスが明確な期間を3つ選択
        if len(self.hysteresis_results) > 0:
            # ヒステリシス面積が大きい順にソート
            sorted_results = self.hysteresis_results.sort_values('hysteresis_area', ascending=False)
            
            for i in range(min(3, len(sorted_results))):
                ax = axes[0, i]
                
                period_data = sorted_results.iloc[i]
                start, end = int(period_data['start']), int(period_data['end'])
                
                # 期間内のデータ取得
                discharge = self.df_processed['ダム_全放流量'].iloc[start:end].values
                water_level = self.df_processed['水位_水位'].iloc[start:end].values
                
                # 欠損値除去とフィルタリング
                mask = ~(np.isnan(discharge) | np.isnan(water_level))
                discharge_clean = discharge[mask]
                water_level_clean = water_level[mask]
                valid_mask = (water_level_clean >= 3.0) & (discharge_clean >= 150.0)
                
                if valid_mask.sum() > 0:
                    # 変化率計算
                    dQ_dt = np.gradient(discharge_clean)
                    
                    # 上昇期と下降期の色分け
                    rising_mask = (dQ_dt > 0) & valid_mask
                    falling_mask = (dQ_dt < 0) & valid_mask
                    
                    # プロット
                    if rising_mask.sum() > 0:
                        ax.scatter(discharge_clean[rising_mask], water_level_clean[rising_mask],
                                 c='red', alpha=0.6, s=30, label='上昇期')
                    if falling_mask.sum() > 0:
                        ax.scatter(discharge_clean[falling_mask], water_level_clean[falling_mask],
                                 c='blue', alpha=0.6, s=30, label='下降期')
                    
                    # 回帰線
                    Q_range = np.linspace(discharge_clean[valid_mask].min(), 
                                        discharge_clean[valid_mask].max(), 100)
                    H_rise = period_data['slope_rise'] * Q_range + period_data['intercept_rise']
                    H_fall = period_data['slope_fall'] * Q_range + period_data['intercept_fall']
                    
                    ax.plot(Q_range, H_rise, 'r--', linewidth=2, 
                           label=f'上昇期: y={period_data["slope_rise"]:.4f}x+{period_data["intercept_rise"]:.2f}')
                    ax.plot(Q_range, H_fall, 'b--', linewidth=2,
                           label=f'下降期: y={period_data["slope_fall"]:.4f}x+{period_data["intercept_fall"]:.2f}')
                    
                    # 時系列順に線で結ぶ
                    ax.plot(discharge_clean[valid_mask], water_level_clean[valid_mask],
                           'gray', alpha=0.3, linewidth=1)
                    
                    ax.set_xlabel('放流量 (m³/s)')
                    ax.set_ylabel('水位 (m)')
                    ax.set_title(f'ヒステリシスループ例 {i+1}\n面積: {period_data["hysteresis_area"]:.1f}')
                    ax.legend(fontsize=8)
                    ax.grid(True, alpha=0.3)
        
        # 2. 傾きの比較（2行目左）
        ax3 = axes[1, 0]
        
        if len(self.hysteresis_results) > 0:
            # 上昇期と下降期の傾きの散布図
            ax3.scatter(self.hysteresis_results['slope_rise'], 
                       self.hysteresis_results['slope_fall'],
                       c=self.hysteresis_results['mean_discharge'],
                       cmap='viridis', s=50, alpha=0.6)
            
            # 1:1ライン
            slope_min = min(self.hysteresis_results['slope_rise'].min(),
                          self.hysteresis_results['slope_fall'].min())
            slope_max = max(self.hysteresis_results['slope_rise'].max(),
                          self.hysteresis_results['slope_fall'].max())
            ax3.plot([slope_min, slope_max], [slope_min, slope_max],
                    'k--', linewidth=2, label='1:1')
            
            # カラーバー
            cbar = plt.colorbar(ax3.collections[0], ax=ax3)
            cbar.set_label('平均放流量 (m³/s)', rotation=270, labelpad=15)
            
            ax3.set_xlabel('上昇期の傾き')
            ax3.set_ylabel('下降期の傾き')
            ax3.set_title('上昇期 vs 下降期の応答傾き')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 3. 傾き比のヒストグラム（2行目中央）
        ax4 = axes[1, 1]
        
        if len(self.hysteresis_results) > 0:
            slope_ratios = self.hysteresis_results['slope_ratio'].dropna()
            if len(slope_ratios) > 0:
                ax4.hist(slope_ratios, bins=15, color='purple', 
                        edgecolor='black', alpha=0.7)
                
                mean_ratio = slope_ratios.mean()
                median_ratio = slope_ratios.median()
                
                ax4.axvline(1.0, color='black', linestyle='--', linewidth=2,
                           label='比率 = 1.0')
                ax4.axvline(mean_ratio, color='red', linestyle='--', linewidth=2,
                           label=f'平均: {mean_ratio:.2f}')
                ax4.axvline(median_ratio, color='orange', linestyle=':', linewidth=2,
                           label=f'中央値: {median_ratio:.2f}')
                
                ax4.set_xlabel('傾き比（上昇期/下降期）')
                ax4.set_ylabel('頻度')
                ax4.set_title('傾き比の分布')
                ax4.legend()
                ax4.grid(True, alpha=0.3)
        
        # 4. ヒステリシス面積の分布（2行目右）
        ax5 = axes[1, 2]
        
        if len(self.hysteresis_results) > 0:
            ax5.hist(self.hysteresis_results['hysteresis_area'], bins=15,
                    color='green', edgecolor='black', alpha=0.7)
            
            mean_area = self.hysteresis_results['hysteresis_area'].mean()
            ax5.axvline(mean_area, color='darkgreen', linestyle='--', linewidth=2,
                       label=f'平均: {mean_area:.1f}')
            
            ax5.set_xlabel('ヒステリシス面積')
            ax5.set_ylabel('頻度')
            ax5.set_title('ヒステリシス面積の分布')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        
        # 5. 放流量規模とヒステリシス（3行目左）
        ax6 = axes[2, 0]
        
        if len(self.hysteresis_results) > 0:
            ax6.scatter(self.hysteresis_results['mean_discharge'],
                       self.hysteresis_results['hysteresis_area'],
                       c='darkblue', alpha=0.6, s=50)
            
            # トレンドライン
            if len(self.hysteresis_results) >= 3:
                z = np.polyfit(self.hysteresis_results['mean_discharge'],
                             self.hysteresis_results['hysteresis_area'], 1)
                p = np.poly1d(z)
                x_trend = np.linspace(self.hysteresis_results['mean_discharge'].min(),
                                    self.hysteresis_results['mean_discharge'].max(), 100)
                ax6.plot(x_trend, p(x_trend), 'r--', linewidth=2)
            
            ax6.set_xlabel('平均放流量 (m³/s)')
            ax6.set_ylabel('ヒステリシス面積')
            ax6.set_title('放流量規模とヒステリシス効果')
            ax6.grid(True, alpha=0.3)
        
        # 6. 変化率による応答特性（3行目中央）
        ax7 = axes[2, 1]
        
        # 全変動期間のデータを統合して変化率分析
        all_dQ_dt = []
        all_dH_dt = []
        all_Q = []
        
        for _, row in self.hysteresis_results.iterrows():
            start, end = int(row['start']), int(row['end'])
            discharge = self.df_processed['ダム_全放流量'].iloc[start:end].values
            water_level = self.df_processed['水位_水位'].iloc[start:end].values
            
            mask = ~(np.isnan(discharge) | np.isnan(water_level))
            if mask.sum() > 2:
                discharge_clean = discharge[mask]
                water_level_clean = water_level[mask]
                valid_mask = (water_level_clean >= 3.0) & (discharge_clean >= 150.0)
                
                if valid_mask.sum() > 2:
                    dQ_dt = np.gradient(discharge_clean[valid_mask])
                    dH_dt = np.gradient(water_level_clean[valid_mask])
                    
                    all_dQ_dt.extend(dQ_dt)
                    all_dH_dt.extend(dH_dt)
                    all_Q.extend(discharge_clean[valid_mask])
        
        if len(all_dQ_dt) > 0:
            all_dQ_dt = np.array(all_dQ_dt)
            all_dH_dt = np.array(all_dH_dt)
            
            # 上昇期と下降期で色分け
            rising = all_dQ_dt > 0
            falling = all_dQ_dt < 0
            
            scatter = ax7.scatter(all_dQ_dt[rising], all_dH_dt[rising],
                                c='red', alpha=0.3, s=10, label='上昇期')
            ax7.scatter(all_dQ_dt[falling], all_dH_dt[falling],
                       c='blue', alpha=0.3, s=10, label='下降期')
            
            ax7.set_xlabel('放流量変化率 dQ/dt (m³/s/10min)')
            ax7.set_ylabel('水位変化率 dH/dt (m/10min)')
            ax7.set_title('変化率による応答特性')
            ax7.legend()
            ax7.grid(True, alpha=0.3)
            ax7.axhline(y=0, color='k', linewidth=0.5)
            ax7.axvline(x=0, color='k', linewidth=0.5)
        
        # 7. 相関係数の比較（3行目右）
        ax8 = axes[2, 2]
        
        if len(self.hysteresis_results) > 0:
            # 上昇期と下降期の相関係数を箱ひげ図で比較
            data = [self.hysteresis_results['r_rise'].values,
                   self.hysteresis_results['r_fall'].values]
            
            bp = ax8.boxplot(data, labels=['上昇期', '下降期'], patch_artist=True)
            bp['boxes'][0].set_facecolor('lightcoral')
            bp['boxes'][1].set_facecolor('lightblue')
            
            # 平均値を追加
            means = [np.mean(d) for d in data]
            ax8.scatter([1, 2], means, color='black', s=100, zorder=5, label='平均値')
            
            # 統計情報を表示
            ax8.text(0.02, 0.98, 
                    f'上昇期: 平均{means[0]:.3f}\n下降期: 平均{means[1]:.3f}',
                    transform=ax8.transAxes, va='top', ha='left',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax8.set_ylabel('相関係数')
            ax8.set_title('上昇期・下降期の相関係数比較')
            ax8.legend()
            ax8.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # 統計サマリーを表示
        if len(self.hysteresis_results) > 0:
            print("\n=== ヒステリシス分析サマリー ===")
            print(f"分析期間数: {len(self.hysteresis_results)}")
            print(f"傾き比（上昇/下降）:")
            print(f"  平均: {self.hysteresis_results['slope_ratio'].mean():.3f}")
            print(f"  中央値: {self.hysteresis_results['slope_ratio'].median():.3f}")
            print(f"ヒステリシス面積:")
            print(f"  平均: {self.hysteresis_results['hysteresis_area'].mean():.1f}")
            print(f"  最大: {self.hysteresis_results['hysteresis_area'].max():.1f}")
            
            # 傾き比が1より大きい期間の割合
            ratio_gt_1 = (self.hysteresis_results['slope_ratio'] > 1).sum()
            print(f"\n上昇期の傾き > 下降期の傾き: {ratio_gt_1}/{len(self.hysteresis_results)} " +
                  f"({ratio_gt_1/len(self.hysteresis_results)*100:.1f}%)")
        
        # 保存
        output_path = f"figures/figure11_hysteresis_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nFigure 11を保存しました: {output_path}")
        
        plt.close()
        
        return fig

def main():
    """単独実行時のみ使用"""
    analyzer = Figure11HysteresisAnalysis()
    
    # データ読み込みと分析
    analyzer.load_data()
    analyzer.preprocess_data()
    analyzer.classify_stable_variable_periods()
    
    # Figure 11の作成
    analyzer.create_figure11_hysteresis_analysis()
    
    print("\nFigure 11（ヒステリシス分析）の作成が完了しました。")

if __name__ == "__main__":
    main()