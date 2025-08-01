#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
水位予測モデルのリアルタイムデモンストレーション
実データを使用して3時間先までの水位予測を可視化
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from datetime import datetime, timedelta
import pickle
import warnings
warnings.filterwarnings('ignore')

# 日本語フォント設定
plt.rcParams['font.sans-serif'] = ['Yu Gothic', 'Hiragino Sans', 'Meirio', 'DejaVu Sans']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.unicode_minus'] = False

# 水位予測モデルのインポート
from water_level_prediction_model import WaterLevelPredictor


class WaterLevelPredictionDemo:
    def __init__(self, model_path, data_path="統合データ_水位ダム_20250730_205325.csv"):
        """
        初期化
        
        Parameters:
        -----------
        model_path : str
            学習済みモデルのパス
        data_path : str
            使用するデータファイルのパス
        """
        self.model_path = model_path
        self.data_path = data_path
        
        # モデルの読み込み
        self.predictor = WaterLevelPredictor()
        self.predictor.load_model(model_path)
        
        # データ読み込み
        self.load_data()
        
        # 結果保存用
        self.history_time = []
        self.history_discharge = []
        self.history_water_level = []
        self.predictions_history = []
        self.actual_future_water_levels = []
    
    def load_data(self):
        """実データの読み込みと前処理"""
        print(f"データファイルを読み込み中: {self.data_path}")
        
        # データ読み込み
        self.df = pd.read_csv(self.data_path, encoding='utf-8')
        self.df['時刻'] = pd.to_datetime(self.df['時刻'])
        
        # 欠損値処理
        self.df['水位_水位'].fillna(method='ffill', inplace=True)
        self.df['ダム_全放流量'].fillna(method='ffill', inplace=True)
        
        # フィルタリング: 放流量≥150m³/s かつ 水位≥3.0m
        mask = (self.df['ダム_全放流量'] >= 150) & (self.df['水位_水位'] >= 3.0)
        self.df_filtered = self.df[mask].reset_index(drop=True)
        
        print(f"読み込み完了: 全{len(self.df)}行中、{len(self.df_filtered)}行が対象")
        print(f"期間: {self.df_filtered['時刻'].min()} ～ {self.df_filtered['時刻'].max()}")
    
    def find_demo_periods(self, n_periods=5):
        """デモに適した期間を自動検出"""
        periods = []
        
        # 変動が大きい期間を探す
        window_size = 36  # 6時間
        
        for i in range(18, len(self.df_filtered) - window_size - 18, 36):  # 6時間ごと
            window = self.df_filtered.iloc[i:i+window_size]
            
            # 変動係数を計算
            cv_discharge = window['ダム_全放流量'].std() / window['ダム_全放流量'].mean()
            
            # 放流量の変化量
            discharge_change = window['ダム_全放流量'].max() - window['ダム_全放流量'].min()
            
            # 変動が大きい期間を選択
            if cv_discharge > 0.1 and discharge_change > 100:
                periods.append({
                    'start_idx': i,
                    'start_time': window.iloc[0]['時刻'],
                    'cv': cv_discharge,
                    'discharge_range': (window['ダム_全放流量'].min(), window['ダム_全放流量'].max()),
                    'type': 'variable',
                    'score': cv_discharge * discharge_change  # スコアリング
                })
        
        # スコアの高い順にソート
        periods.sort(key=lambda x: x['score'], reverse=True)
        
        # 上位n_periods個を返す
        return periods[:n_periods]
    
    def find_period_by_time(self, target_time):
        """指定時刻を含む期間を検索"""
        if isinstance(target_time, str):
            target_time = pd.to_datetime(target_time)
        
        # 指定時刻に最も近いインデックスを検索
        time_diff = abs(self.df_filtered['時刻'] - target_time)
        closest_idx = time_diff.idxmin()
        
        # 最低限の予測データが確保できるか確認
        if closest_idx < 18:
            print(f"警告: 指定時刻 {target_time} は開始位置に近すぎます。最初の有効な時刻を使用します。")
            closest_idx = 18
        elif closest_idx > len(self.df_filtered) - 36:
            print(f"警告: 指定時刻 {target_time} は終了位置に近すぎます。最後の有効な時刻を使用します。")
            closest_idx = len(self.df_filtered) - 36
        
        # 期間情報を作成
        window_size = 36  # 6時間
        window_end = min(closest_idx + window_size, len(self.df_filtered))
        window = self.df_filtered.iloc[closest_idx:window_end]
        
        # 変動係数を計算
        cv_discharge = window['ダム_全放流量'].std() / window['ダム_全放流量'].mean()
        
        period = {
            'start_idx': closest_idx,
            'start_time': self.df_filtered.iloc[closest_idx]['時刻'],
            'cv': cv_discharge,
            'discharge_range': (window['ダム_全放流量'].min(), window['ダム_全放流量'].max()),
            'type': 'user_specified',
            'target_time': target_time
        }
        
        return period
    
    def prepare_demo_data(self, start_idx, demo_length=36):
        """デモ用データの準備"""
        # デモ期間のデータ
        end_idx = min(start_idx + demo_length + 18, len(self.df_filtered))  # 予測分も含める
        demo_data = self.df_filtered.iloc[start_idx:end_idx].copy()
        
        return demo_data
    
    def run_static_demo(self, demo_period, save_csv=True):
        """静的なデモの実行（アニメーションなし）"""
        print(f"\n=== デモ実行: {demo_period['start_time']} ===")
        
        # デモデータの準備
        demo_data = self.prepare_demo_data(demo_period['start_idx'])
        
        # 予測結果を格納
        predictions = []
        actual_values = []
        times = []
        
        # 3時間分のデータで予測
        for i in range(18, min(36, len(demo_data) - 18)):
            # 現在時刻
            current_time = demo_data.iloc[i]['時刻']
            
            # 特徴量作成（demo_dataはすでにフィルタリング済み）
            features = self.predictor.create_features(demo_data, i, already_filtered=True)
            
            # DataFrameに変換
            X = pd.DataFrame([features])
            
            # カテゴリカル変数のエンコーディング
            for col in ['flow_direction', 'change_magnitude', 'water_level_category']:
                if col in self.predictor.label_encoders:
                    X[col] = self.predictor.label_encoders[col].transform(X[col])
            
            # 特徴量の順序を合わせる
            X = X[self.predictor.feature_names]
            
            # デバッグ：特徴量の状態を確認
            if X.shape[1] == 0:
                print(f"警告: 時刻 {current_time} で特徴量が0個になりました")
                print(f"features: {features}")
                print(f"feature_names: {self.predictor.feature_names}")
                continue
            
            # 予測
            y_pred = self.predictor.predict(X.values)
            
            # 実際の値
            actual = []
            for j in range(1, 19):
                if i + j < len(demo_data):
                    actual.append(demo_data.iloc[i + j]['水位_水位'])
                else:
                    actual.append(np.nan)
            
            predictions.append(y_pred[0])
            actual_values.append(actual)
            times.append(current_time)
        
        # CSVに保存
        if save_csv:
            self.save_predictions_to_csv(demo_data, times, predictions, actual_values)
        
        # 結果の可視化
        self.visualize_predictions(demo_data, times, predictions, actual_values)
    
    def save_predictions_to_csv(self, demo_data, times, predictions, actual_values):
        """予測結果をCSVファイルに保存"""
        # 1. グラフ作成用データ（時系列順）
        graph_data = []
        
        # まず実測値データを時系列で追加
        for idx in range(len(demo_data)):
            time = demo_data.iloc[idx]['時刻']
            graph_data.append({
                '時刻': time,
                '実測水位': demo_data.iloc[idx]['水位_水位'],
                '放流量': demo_data.iloc[idx]['ダム_全放流量'],
                '予測実行回': np.nan,  # 実測値なので予測実行回はなし
                '予測ホライズン(分)': 0,  # 実測値は0
                '予測水位': np.nan
            })
        
        # 各予測実行回の予測値を追加
        for exec_idx, (exec_time, pred, actual) in enumerate(zip(times, predictions, actual_values)):
            # 各予測時点での結果
            for j in range(18):
                pred_time = exec_time + timedelta(minutes=10*(j+1))
                graph_data.append({
                    '時刻': pred_time,
                    '実測水位': np.nan,  # 予測値なので実測値はなし（後で埋める）
                    '放流量': np.nan,   # 予測値なので放流量もなし（後で埋める）
                    '予測実行回': exec_idx + 1,
                    '予測ホライズン(分)': (j+1)*10,
                    '予測水位': pred[j]
                })
        
        # DataFrameに変換して時刻順にソート
        df_graph = pd.DataFrame(graph_data)
        df_graph = df_graph.sort_values('時刻').reset_index(drop=True)
        
        # 実測値を前方補完で埋める（同じ時刻の実測値を予測値の行にも設定）
        df_graph['実測水位'] = df_graph.groupby('時刻')['実測水位'].transform(lambda x: x.fillna(method='ffill').fillna(method='bfill'))
        df_graph['放流量'] = df_graph.groupby('時刻')['放流量'].transform(lambda x: x.fillna(method='ffill').fillna(method='bfill'))
        
        # ファイル名の生成
        graph_filename = f'prediction_graph_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        
        # CSV保存
        df_graph.to_csv(graph_filename, index=False, encoding='utf-8-sig')
        print(f"\nグラフ作成用データを保存しました: {graph_filename}")
        
        # 2. 詳細な予測結果データ（元のフォーマット）
        results = []
        
        for i, (time, pred, actual) in enumerate(zip(times, predictions, actual_values)):
            # 基本情報
            base_info = {
                '予測実行時刻': time,
                '放流量': demo_data[demo_data['時刻'] == time]['ダム_全放流量'].values[0],
                '実測水位': demo_data[demo_data['時刻'] == time]['水位_水位'].values[0]
            }
            
            # 各予測時点での結果
            for j in range(18):
                pred_time = time + timedelta(minutes=10*(j+1))
                result = base_info.copy()
                result.update({
                    '予測時刻': pred_time,
                    '予測時間(分)': (j+1)*10,
                    '予測水位': pred[j],
                    '実測水位_予測時刻': actual[j] if j < len(actual) else np.nan,
                    '予測誤差': pred[j] - actual[j] if j < len(actual) and not np.isnan(actual[j]) else np.nan
                })
                results.append(result)
        
        # DataFrameに変換
        df_results = pd.DataFrame(results)
        
        # ファイル名の生成
        filename = f'prediction_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        
        # CSV保存
        df_results.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"予測結果詳細を保存しました: {filename}")
        
        # 3. サマリー統計も計算して保存
        summary = df_results.groupby('予測時間(分)')['予測誤差'].agg(['mean', 'std', 'min', 'max', 'count'])
        summary.columns = ['平均誤差', '標準偏差', '最小誤差', '最大誤差', 'サンプル数']
        summary['MAE'] = df_results.groupby('予測時間(分)')['予測誤差'].apply(lambda x: np.nanmean(np.abs(x)))
        summary['RMSE'] = df_results.groupby('予測時間(分)')['予測誤差'].apply(lambda x: np.sqrt(np.nanmean(x**2)))
        
        summary_filename = f'prediction_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        summary.to_csv(summary_filename, encoding='utf-8-sig')
        print(f"予測精度サマリーを保存しました: {summary_filename}")
    
    def visualize_predictions(self, demo_data, times, predictions, actual_values):
        """予測結果の可視化"""
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
        
        # 1. 放流量の推移
        ax1.plot(demo_data['時刻'], demo_data['ダム_全放流量'], 'b-', linewidth=2, label='放流量')
        ax1.set_ylabel('放流量 (m³/s)')
        ax1.set_title('デモ期間の放流量推移')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 2. 水位の実測値と予測値
        # 実測値
        ax2.plot(demo_data['時刻'], demo_data['水位_水位'], 'ko-', markersize=4, label='実測値')
        
        # 予測値（各時点での3時間先予測）
        colors = plt.cm.rainbow(np.linspace(0, 1, len(times)))
        
        for i, (time, pred, color) in enumerate(zip(times, predictions, colors)):
            # 予測時刻
            pred_times = [time + timedelta(minutes=10*(j+1)) for j in range(18)]
            
            # 最後の予測のみラベル付き
            label = '予測値' if i == len(times) - 1 else None
            ax2.plot(pred_times, pred, '--', color=color, alpha=0.6, linewidth=1, label=label)
            
            # 予測開始点をマーク
            ax2.plot(time, demo_data[demo_data['時刻'] == time]['水位_水位'].values[0], 
                    'o', color=color, markersize=6)
        
        ax2.set_ylabel('水位 (m)')
        ax2.set_title('水位の実測値と予測値（3時間先まで）')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # 3. 予測誤差の推移
        # 最終時点での予測誤差を計算
        last_pred = predictions[-1]
        last_actual = actual_values[-1]
        
        # 予測誤差
        pred_errors = []
        for j in range(18):
            if not np.isnan(last_actual[j]):
                error = last_pred[j] - last_actual[j]
                pred_errors.append(error)
            else:
                pred_errors.append(np.nan)
        
        # 予測時間
        pred_minutes = [(j+1)*10 for j in range(18)]
        
        ax3.plot(pred_minutes, pred_errors, 'ro-', markersize=6, linewidth=2)
        ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax3.set_xlabel('予測時間 (分)')
        ax3.set_ylabel('予測誤差 (m)')
        ax3.set_title('予測誤差の時間変化')
        ax3.grid(True, alpha=0.3)
        
        # 統計情報を追加
        valid_errors = [e for e in pred_errors if not np.isnan(e)]
        if valid_errors:
            mae = np.mean(np.abs(valid_errors))
            rmse = np.sqrt(np.mean(np.array(valid_errors)**2))
            ax3.text(0.02, 0.95, f'MAE: {mae:.3f}m\nRMSE: {rmse:.3f}m', 
                    transform=ax3.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(f'prediction_demo_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png', dpi=150)
        plt.show()
    
    def run_animated_demo(self, demo_period):
        """アニメーション付きデモの実行"""
        print(f"\n=== アニメーションデモ: {demo_period['start_time']} ===")
        
        # デモデータの準備
        self.demo_data = self.prepare_demo_data(demo_period['start_idx'])
        self.current_idx = 18  # 3時間分のデータから開始
        
        # 履歴をリセット
        self.history_time = []
        self.history_discharge = []
        self.history_water_level = []
        self.predictions_history = []
        
        # アニメーション用の図を準備
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # アニメーション実行
        ani = animation.FuncAnimation(
            self.fig, self.update_animation, 
            frames=min(18, len(self.demo_data) - self.current_idx - 18),
            interval=500, repeat=False, blit=False
        )
        
        plt.tight_layout()
        
        # 保存オプション
        save_gif = input("アニメーションをGIFとして保存しますか？ (y/n): ").lower() == 'y'
        if save_gif:
            print("GIFを保存中...")
            ani.save(f'prediction_demo_animation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.gif', 
                    writer='pillow', fps=2)
            print("保存完了！")
        
        plt.show()
    
    def update_animation(self, frame):
        """アニメーションの更新"""
        # クリア
        self.ax1.clear()
        self.ax2.clear()
        
        # 現在のデータ
        current_data = self.demo_data.iloc[self.current_idx]
        current_time = current_data['時刻']
        
        # 履歴に追加
        self.history_time.append(current_time)
        self.history_discharge.append(current_data['ダム_全放流量'])
        self.history_water_level.append(current_data['水位_水位'])
        
        # 特徴量作成と予測（demo_dataはすでにフィルタリング済み）
        features = self.predictor.create_features(self.demo_data, self.current_idx, already_filtered=True)
        X = pd.DataFrame([features])
        
        # エンコーディング
        for col in ['flow_direction', 'change_magnitude', 'water_level_category']:
            if col in self.predictor.label_encoders:
                X[col] = self.predictor.label_encoders[col].transform(X[col])
        
        X = X[self.predictor.feature_names]
        
        # 予測
        y_pred = self.predictor.predict(X.values)[0]
        self.predictions_history.append(y_pred)
        
        # 上部：放流量と遅延時間
        self.ax1.plot(self.history_time, self.history_discharge, 'b-', linewidth=2, label='放流量')
        self.ax1.set_ylabel('放流量 (m³/s)', color='b')
        self.ax1.tick_params(axis='y', labelcolor='b')
        self.ax1.grid(True, alpha=0.3)
        
        # 遅延時間を右軸に表示
        ax1_twin = self.ax1.twinx()
        delay_times = [self.predictor.calculate_delay_time(d) for d in self.history_discharge]
        ax1_twin.plot(self.history_time, delay_times, 'r--', linewidth=1, alpha=0.7, label='遅延時間')
        ax1_twin.set_ylabel('遅延時間 (分)', color='r')
        ax1_twin.tick_params(axis='y', labelcolor='r')
        
        self.ax1.set_title(f'リアルタイム水位予測デモ - {current_time.strftime("%Y-%m-%d %H:%M")}')
        
        # 下部：水位（実測値と予測値）
        self.ax2.plot(self.history_time, self.history_water_level, 'ko-', markersize=4, label='実測値')
        
        # 過去の予測を薄く表示
        for i, (pred_time, pred) in enumerate(zip(self.history_time[:-1], self.predictions_history[:-1])):
            pred_times = [pred_time + timedelta(minutes=10*(j+1)) for j in range(18)]
            self.ax2.plot(pred_times, pred, 'g--', alpha=0.2, linewidth=1)
        
        # 現在の予測を強調表示
        if self.predictions_history:
            current_pred_times = [current_time + timedelta(minutes=10*(j+1)) for j in range(18)]
            self.ax2.plot(current_pred_times, y_pred, 'r-', linewidth=2, label='予測値（3時間先まで）')
            
            # 予測範囲を網掛け
            self.ax2.axvspan(current_time, current_pred_times[-1], alpha=0.2, color='yellow')
        
        self.ax2.set_xlabel('時刻')
        self.ax2.set_ylabel('水位 (m)')
        self.ax2.legend()
        self.ax2.grid(True, alpha=0.3)
        
        # x軸の範囲を調整
        if len(self.history_time) > 6:
            xlim_start = self.history_time[-6]
            xlim_end = current_time + timedelta(hours=3.5)
            self.ax1.set_xlim(xlim_start, xlim_end)
            self.ax2.set_xlim(xlim_start, xlim_end)
        
        # インデックスを進める
        self.current_idx += 1
        
        return self.ax1, self.ax2


def main():
    """メイン処理"""
    print("=== 水位予測モデル デモンストレーション ===")
    
    # モデルファイルの確認
    import glob
    model_files = glob.glob("water_level_predictor_*.pkl")
    
    if not model_files:
        print("エラー: 学習済みモデルが見つかりません")
        print("先に train_water_level_model.py を実行してモデルを学習してください")
        return
    
    # 最新のモデルを選択
    model_files.sort()
    model_path = model_files[-1]
    print(f"使用するモデル: {model_path}")
    
    # デモの初期化
    demo = WaterLevelPredictionDemo(model_path)
    
    # デモ期間の検出
    print("\n適切なデモ期間を検索中...")
    demo_periods = demo.find_demo_periods(n_periods=5)
    
    if not demo_periods:
        print("エラー: 適切なデモ期間が見つかりません")
        return
    
    # デモ期間の選択
    print("\n=== デモ期間の候補 ===")
    for i, period in enumerate(demo_periods):
        print(f"{i+1}. {period['start_time']} - "
              f"放流量: {period['discharge_range'][0]:.0f}-{period['discharge_range'][1]:.0f} m³/s, "
              f"変動係数: {period['cv']:.3f}")
    
    # ユーザー選択
    while True:
        try:
            choice = int(input("\nデモ期間を選択してください (1-5): ")) - 1
            if 0 <= choice < len(demo_periods):
                break
            else:
                print("無効な選択です")
        except ValueError:
            print("数値を入力してください")
    
    selected_period = demo_periods[choice]
    
    # デモタイプの選択
    print("\n=== デモタイプ ===")
    print("1. 静的デモ（予測結果の一括表示）")
    print("2. アニメーションデモ（リアルタイム風の表示）")
    
    demo_type = input("選択してください (1/2): ")
    
    if demo_type == '1':
        demo.run_static_demo(selected_period)
    elif demo_type == '2':
        demo.run_animated_demo(selected_period)
    else:
        print("無効な選択です。静的デモを実行します。")
        demo.run_static_demo(selected_period)
    
    print("\nデモ完了！")


if __name__ == "__main__":
    main()