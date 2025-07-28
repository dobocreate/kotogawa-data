import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy import signal
from scipy.stats import pearsonr
import warnings
from tkinter import Tk, filedialog
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

warnings.filterwarnings('ignore')

# 日本語フォントの設定
plt.rcParams['font.sans-serif'] = ['Yu Gothic', 'Meiryo', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']
plt.rcParams['axes.unicode_minus'] = False

class WaterDischargeAnalyzer:
    """河川水位とダム放流量の関係を分析するクラス"""
    
    def __init__(self):
        self.data = None
        self.water_level_col = '水位_水位'
        self.discharge_col = 'ダム_全放流量'
        self.time_col = '時刻'
        self.rainfall_col = 'ダム_60分雨量'
        
    def load_data(self):
        """統合データを読み込む"""
        root = Tk()
        root.withdraw()
        
        filepath = filedialog.askopenfilename(
            title="統合データファイルを選択",
            filetypes=[("CSVファイル", "*.csv"), ("すべてのファイル", "*.*")]
        )
        
        root.destroy()
        
        if not filepath:
            print("ファイルが選択されませんでした")
            return False
        
        try:
            self.data = pd.read_csv(filepath, encoding='utf-8-sig')
            print(f"データを読み込みました: {filepath}")
            
            # 時刻をdatetime型に変換
            self.data[self.time_col] = pd.to_datetime(self.data[self.time_col])
            
            # データ情報を表示
            print(f"\nデータ期間: {self.data[self.time_col].min()} ～ {self.data[self.time_col].max()}")
            print(f"データ行数: {len(self.data)}")
            
            # 必要なカラムの確認
            required_cols = [self.water_level_col, self.discharge_col]
            missing_cols = [col for col in required_cols if col not in self.data.columns]
            if missing_cols:
                print(f"エラー: 必要なカラムが見つかりません: {missing_cols}")
                return False
            
            return True
            
        except Exception as e:
            print(f"データ読み込みエラー: {e}")
            return False
    
    def preprocess_data(self):
        """データの前処理"""
        print("\n=== データ前処理 ===")
        
        # 欠損値の確認
        print(f"水位データ欠損値: {self.data[self.water_level_col].isnull().sum()}個")
        print(f"放流量データ欠損値: {self.data[self.discharge_col].isnull().sum()}個")
        
        # 線形補間で欠損値を埋める
        self.data[self.water_level_col] = self.data[self.water_level_col].interpolate(method='linear')
        self.data[self.discharge_col] = self.data[self.discharge_col].interpolate(method='linear')
        
        # 外れ値の検出（IQR法）
        def remove_outliers_iqr(series, multiplier=3):
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            return series.clip(lower=lower_bound, upper=upper_bound)
        
        # 外れ値の処理（クリッピング）
        self.data[self.water_level_col] = remove_outliers_iqr(self.data[self.water_level_col])
        self.data[self.discharge_col] = remove_outliers_iqr(self.data[self.discharge_col])
        
        print("前処理完了")
    
    def calculate_water_level_trend(self, window=6):
        """水位の上昇/下降を判定"""
        print(f"\n=== 水位トレンド分析（移動平均窓: {window*10}分） ===")
        
        # 移動平均の計算
        self.data['water_level_ma'] = self.data[self.water_level_col].rolling(window=window, center=True).mean()
        
        # トレンドの判定（閾値: 0.01m = 1cm）
        threshold = 0.01
        self.data['water_trend'] = 'stable'
        self.data.loc[self.data[self.water_level_col] > self.data['water_level_ma'] + threshold, 'water_trend'] = 'rising'
        self.data.loc[self.data[self.water_level_col] < self.data['water_level_ma'] - threshold, 'water_trend'] = 'falling'
        
        # トレンドの統計
        trend_counts = self.data['water_trend'].value_counts()
        print("水位トレンドの分布:")
        for trend, count in trend_counts.items():
            percentage = (count / len(self.data)) * 100
            print(f"  {trend}: {count}個 ({percentage:.1f}%)")
    
    def analyze_discharge_delay(self):
        """放流量範囲別の最適遅延時間を分析"""
        print("\n=== 放流量範囲別遅延時間分析 ===")
        
        # 放流量範囲の設定（0-1200を100刻み）
        discharge_ranges = [(i, i+100) for i in range(0, 1200, 100)]
        
        results = []
        
        for lower, upper in discharge_ranges:
            # 該当範囲のデータを抽出
            mask = (self.data[self.discharge_col] >= lower) & (self.data[self.discharge_col] < upper)
            subset = self.data[mask].copy()
            
            # データ数に応じた閾値の設定
            if lower >= 800:
                min_data_points = 5  # 高放流量域は閾値を緩和
            elif lower >= 400:
                min_data_points = 10
            else:
                min_data_points = 50
            
            if len(subset) < min_data_points:
                print(f"放流量 {lower}-{upper} m³/s: データ不足 ({len(subset)}個)")
                # データ不足でも結果に含める（グラフ表示のため）
                results.append({
                    'discharge_center': (lower + upper) / 2,
                    'discharge_range': f"{lower}-{upper}",
                    'optimal_lag': np.nan,
                    'max_correlation': np.nan,
                    'std_lag': 0,
                    'data_count': len(subset)
                })
                continue
            
            # 相互相関分析
            max_lag = 30  # 最大300分（5時間）
            correlations = []
            
            for lag in range(0, max_lag):
                if lag == 0:
                    continue  # ゼロラグは除外
                
                # 放流量をシフト
                shifted_discharge = subset[self.discharge_col].shift(lag)
                
                # NaNを除外して相関計算
                valid_mask = ~(subset[self.water_level_col].isna() | shifted_discharge.isna())
                if valid_mask.sum() < min_data_points:
                    continue
                
                corr, _ = pearsonr(
                    subset[self.water_level_col][valid_mask],
                    shifted_discharge[valid_mask]
                )
                correlations.append((lag * 10, corr))  # 分単位に変換
            
            if correlations:
                # 最大相関を持つ遅延時間を特定
                correlations.sort(key=lambda x: x[1], reverse=True)
                optimal_lag, max_corr = correlations[0]
                
                # 標準偏差の計算（上位5つの遅延時間から）
                top_lags = [lag for lag, corr in correlations[:5] if corr > 0.5]
                if len(top_lags) > 1:
                    std_lag = np.std(top_lags)
                else:
                    std_lag = 0
                
                results.append({
                    'discharge_center': (lower + upper) / 2,
                    'discharge_range': f"{lower}-{upper}",
                    'optimal_lag': optimal_lag,
                    'max_correlation': max_corr,
                    'std_lag': std_lag,
                    'data_count': len(subset)
                })
                
                print(f"放流量 {lower}-{upper} m³/s: 最適遅延 {optimal_lag}分 (r={max_corr:.3f}, n={len(subset)})")
        
        self.delay_results = pd.DataFrame(results)
        return self.delay_results
    
    def plot_discharge_delay_analysis(self):
        """放流量範囲と遅延時間の関係をプロット（データ数分布含む）"""
        if not hasattr(self, 'delay_results') or self.delay_results.empty:
            print("遅延時間分析結果がありません")
            return
        
        # 3行2列のレイアウトに変更
        fig = plt.figure(figsize=(16, 14))
        
        # 1行1列：遅延時間の棒グラフ
        ax1 = plt.subplot(3, 2, 1)
        x = self.delay_results['discharge_center']
        y = self.delay_results['optimal_lag']
        yerr = self.delay_results['std_lag']
        colors = ['red' if n < 500 else 'blue' for n in self.delay_results['data_count']]
        
        bars = ax1.bar(x, y, width=80, yerr=yerr, capsize=5, 
                       color=colors, alpha=0.7, edgecolor='black')
        
        ax1.set_xlabel('放流量範囲中央値 (m³/s)', fontsize=12)
        ax1.set_ylabel('最適遅延時間 (分)', fontsize=12)
        ax1.set_title('放流量範囲別の最適遅延時間', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, max(y) * 1.2 if len(y) > 0 else 300)
        
        # データ数を表示
        for i, (xi, yi, n) in enumerate(zip(x, y, self.delay_results['data_count'])):
            ax1.text(xi, yi + 5, f'n={n}', ha='center', va='bottom', fontsize=8)
        
        # 凡例
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='blue', alpha=0.7, label='データ数 ≥ 500'),
            Patch(facecolor='red', alpha=0.7, label='データ数 < 500')
        ]
        ax1.legend(handles=legend_elements, loc='upper right')
        
        # 1行2列：相関係数の折れ線グラフ
        ax2 = plt.subplot(3, 2, 2)
        ax2.plot(x, self.delay_results['max_correlation'], 'o-', color='green', 
                linewidth=2, markersize=8)
        ax2.set_xlabel('放流量範囲中央値 (m³/s)', fontsize=12)
        ax2.set_ylabel('最大相関係数', fontsize=12)
        ax2.set_title('放流量範囲別の最大相関係数', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='r=0.5')
        ax2.legend()
        
        # 3行2列：放流量範囲別データ数分布
        ax6 = plt.subplot(3, 2, 6)
        
        # 全データから放流量範囲別のデータ数を計算
        discharge_ranges = [(i, i+100) for i in range(0, 1200, 100)]
        data_counts = []
        
        for min_q, max_q in discharge_ranges:
            mask = (self.data[self.discharge_col] >= min_q) & \
                   (self.data[self.discharge_col] < max_q)
            count = mask.sum()
            data_counts.append(count)
        
        # 色の設定（データ数による）
        colors = ['green' if c >= 1000 else 'yellow' if c >= 100 else 'orange' if c >= 10 else 'red' 
                  for c in data_counts]
        
        # 棒グラフ作成
        bars = ax6.bar(range(len(discharge_ranges)), data_counts, color=colors, alpha=0.7, edgecolor='black')
        
        # 各棒の上にデータ数を表示
        for i, (bar, count) in enumerate(zip(bars, data_counts)):
            height = bar.get_height()
            if height > 0:
                ax6.text(bar.get_x() + bar.get_width()/2, height + 50, 
                         str(count), ha='center', va='bottom', fontsize=8)
        
        # 閾値線（分析に使用した10個）
        ax6.axhline(y=10, color='red', linestyle='--', alpha=0.5, label='分析閾値(n=10)')
        
        # ラベル設定
        ax6.set_xticks(range(len(discharge_ranges)))
        ax6.set_xticklabels([f'{r[0]}-{r[1]}' for r in discharge_ranges], rotation=45)
        ax6.set_xlabel('放流量範囲 (m³/s)', fontsize=12)
        ax6.set_ylabel('データ数', fontsize=12)
        ax6.set_title('放流量範囲別データ数分布', fontsize=14)
        ax6.set_yscale('log')  # 対数スケール
        ax6.grid(True, alpha=0.3, axis='y')
        ax6.legend()
        
        # 凡例（データ数の色分け）
        legend_elements = [
            Patch(facecolor='green', alpha=0.7, label='1000以上'),
            Patch(facecolor='yellow', alpha=0.7, label='100-999'),
            Patch(facecolor='orange', alpha=0.7, label='10-99'),
            Patch(facecolor='red', alpha=0.7, label='10未満')
        ]
        ax6.legend(handles=legend_elements, loc='upper right', title='データ数')
        
        plt.tight_layout()
        plt.show()
    
    def plot_timeseries_with_trend(self, start_date=None, end_date=None):
        """水位と放流量の時系列を水位トレンドで色分けしてプロット"""
        # データの期間を絞る
        plot_data = self.data.copy()
        if start_date:
            plot_data = plot_data[plot_data[self.time_col] >= pd.to_datetime(start_date)]
        if end_date:
            plot_data = plot_data[plot_data[self.time_col] <= pd.to_datetime(end_date)]
        
        if len(plot_data) == 0:
            print("指定期間にデータがありません")
            return
        
        # Plotlyでインタラクティブなグラフを作成
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=('ダム放流量', '河川水位'),
            row_heights=[0.5, 0.5]
        )
        
        # 色の定義
        colors = {
            'rising': {'discharge': 'rgba(255, 200, 200, 0.8)', 'water': 'rgba(255, 0, 0, 0.8)'},
            'falling': {'discharge': 'rgba(200, 200, 255, 0.8)', 'water': 'rgba(0, 0, 255, 0.8)'},
            'stable': {'discharge': 'rgba(200, 200, 200, 0.8)', 'water': 'rgba(100, 100, 100, 0.8)'}
        }
        
        # トレンドごとにデータをプロット
        for trend in ['rising', 'falling', 'stable']:
            trend_data = plot_data[plot_data['water_trend'] == trend]
            
            if len(trend_data) > 0:
                # 放流量
                fig.add_trace(
                    go.Scatter(
                        x=trend_data[self.time_col],
                        y=trend_data[self.discharge_col],
                        mode='markers',
                        name=f'放流量 ({trend})',
                        marker=dict(color=colors[trend]['discharge'], size=3),
                        showlegend=True
                    ),
                    row=1, col=1
                )
                
                # 水位
                fig.add_trace(
                    go.Scatter(
                        x=trend_data[self.time_col],
                        y=trend_data[self.water_level_col],
                        mode='markers',
                        name=f'水位 ({trend})',
                        marker=dict(color=colors[trend]['water'], size=3),
                        showlegend=True
                    ),
                    row=2, col=1
                )
        
        # 降雨量があれば追加（第2軸）
        if self.rainfall_col in plot_data.columns:
            fig.add_trace(
                go.Bar(
                    x=plot_data[self.time_col],
                    y=plot_data[self.rainfall_col],
                    name='降雨量',
                    marker=dict(color='lightblue', opacity=0.5),
                    yaxis='y3'
                ),
                row=1, col=1
            )
        
        # レイアウトの設定
        fig.update_xaxes(title_text="時刻", row=2, col=1)
        fig.update_yaxes(title_text="放流量 (m³/s)", row=1, col=1)
        fig.update_yaxes(title_text="水位 (m)", row=2, col=1)
        
        # 第2軸の設定（降雨量用）
        fig.update_layout(
            yaxis3=dict(
                title="降雨量 (mm)",
                overlaying="y",
                side="right",
                showgrid=False
            )
        )
        
        fig.update_layout(
            height=800,
            title_text=f"水位・放流量時系列（トレンド別色分け）<br>期間: {plot_data[self.time_col].min()} ～ {plot_data[self.time_col].max()}",
            hovermode='x unified'
        )
        
        fig.show()
        
        # 静的なmatplotlibグラフも作成
        self._plot_static_timeseries(plot_data)
    
    def _plot_static_timeseries(self, plot_data):
        """Matplotlibによる静的な時系列グラフ"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
        
        # 色の定義
        colors = {
            'rising': {'discharge': '#ffcccc', 'water': '#ff0000'},
            'falling': {'discharge': '#ccccff', 'water': '#0000ff'},
            'stable': {'discharge': '#cccccc', 'water': '#666666'}
        }
        
        # 背景色で期間を区別（オプション）
        trends = plot_data['water_trend'].values
        times = plot_data[self.time_col].values
        
        # 連続する同じトレンドの期間を特定
        change_points = [0]
        for i in range(1, len(trends)):
            if trends[i] != trends[i-1]:
                change_points.append(i)
        change_points.append(len(trends))
        
        # 背景色を設定
        for i in range(len(change_points)-1):
            start_idx = change_points[i]
            end_idx = change_points[i+1]
            trend = trends[start_idx]
            
            if trend == 'rising':
                bg_color = 'mistyrose'
            elif trend == 'falling':
                bg_color = 'aliceblue'
            else:
                bg_color = 'whitesmoke'
            
            ax1.axvspan(times[start_idx], times[end_idx-1], alpha=0.3, color=bg_color)
            ax2.axvspan(times[start_idx], times[end_idx-1], alpha=0.3, color=bg_color)
        
        # データのプロット
        for trend in ['rising', 'falling', 'stable']:
            trend_data = plot_data[plot_data['water_trend'] == trend]
            
            if len(trend_data) > 0:
                # 放流量
                ax1.scatter(trend_data[self.time_col], trend_data[self.discharge_col],
                          c=colors[trend]['discharge'], s=1, alpha=0.8, label=f'放流量 ({trend})')
                
                # 水位
                ax2.scatter(trend_data[self.time_col], trend_data[self.water_level_col],
                          c=colors[trend]['water'], s=1, alpha=0.8, label=f'水位 ({trend})')
        
        # 降雨量（第2軸）
        if self.rainfall_col in plot_data.columns:
            ax1_rain = ax1.twinx()
            ax1_rain.bar(plot_data[self.time_col], plot_data[self.rainfall_col],
                        alpha=0.3, color='lightblue', label='降雨量', width=0.01)
            ax1_rain.set_ylabel('降雨量 (mm)', fontsize=12)
            ax1_rain.set_ylim(0, plot_data[self.rainfall_col].max() * 3)
            ax1_rain.invert_yaxis()
        
        # グラフの設定
        ax1.set_ylabel('放流量 (m³/s)', fontsize=12)
        ax1.set_title(f'水位・放流量時系列（トレンド別色分け）\n期間: {plot_data[self.time_col].min()} ～ {plot_data[self.time_col].max()}', 
                      fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left', fontsize=8)
        
        ax2.set_xlabel('時刻', fontsize=12)
        ax2.set_ylabel('水位 (m)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper left', fontsize=8)
        
        plt.tight_layout()
        plt.show()
    
    def generate_analysis_report(self):
        """分析レポートの生成"""
        print("\n" + "="*60)
        print("河川水位・ダム放流量分析レポート")
        print("="*60)
        
        if hasattr(self, 'delay_results') and not self.delay_results.empty:
            print("\n【放流量範囲別最適遅延時間】")
            print(self.delay_results.to_string(index=False))
            
            # 全体の傾向
            avg_delay = self.delay_results['optimal_lag'].mean()
            print(f"\n平均遅延時間: {avg_delay:.1f}分")
            
            # 最も相関の高い放流量範囲
            best_corr_idx = self.delay_results['max_correlation'].idxmax()
            best_range = self.delay_results.loc[best_corr_idx]
            print(f"最高相関係数: {best_range['discharge_range']} m³/s (r={best_range['max_correlation']:.3f})")
        
        print("\n【水位トレンド分析結果】")
        if 'water_trend' in self.data.columns:
            trend_summary = self.data.groupby('water_trend').agg({
                self.water_level_col: ['mean', 'std'],
                self.discharge_col: ['mean', 'std']
            }).round(2)
            print(trend_summary)
        
        print("\n" + "="*60)

def main():
    """メイン処理"""
    analyzer = WaterDischargeAnalyzer()
    
    # データの読み込み
    if not analyzer.load_data():
        return
    
    # データの前処理
    analyzer.preprocess_data()
    
    # 水位トレンドの計算
    analyzer.calculate_water_level_trend(window=6)  # 60分移動平均
    
    # 放流量範囲別の遅延時間分析
    analyzer.analyze_discharge_delay()
    
    # グラフの作成
    print("\n=== グラフ作成 ===")
    
    # 1. 放流量範囲と遅延時間の関係
    analyzer.plot_discharge_delay_analysis()
    
    # 2. 時系列グラフ（全期間は重いので、期間を指定）
    print("\n時系列グラフの期間を指定してください")
    print("例: 2024-06-01 から 2024-07-01")
    
    # デモ用に最初の1ヶ月を表示
    start_date = analyzer.data[analyzer.time_col].min()
    end_date = start_date + pd.Timedelta(days=30)
    
    analyzer.plot_timeseries_with_trend(
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d')
    )
    
    # 分析レポートの生成
    analyzer.generate_analysis_report()
    
    print("\n分析が完了しました")

if __name__ == "__main__":
    main()