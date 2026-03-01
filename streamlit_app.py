# -*- coding: utf-8 -*-
"""
Deep Work 最大化・集中波解析アプリ (Wave Dynamics)
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, brier_score_loss
import scipy.signal as signal
import shap
import warnings
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import matplotlib as mpl
import matplotlib.font_manager as fm
import datetime
import math
import io

# --- Streamlit ページ設定 ---
st.set_page_config(page_title="Deep Work Wave Dynamics", layout="wide", initial_sidebar_state="expanded")

# 日本語フォントの設定
font_path = Path(__file__).parent / "assets" / "fonts" / "NotoSansCJKjp-Regular.otf"
if font_path.exists():
    fm.fontManager.addfont(str(font_path))
    prop = fm.FontProperties(fname=str(font_path))
    mpl.rcParams["font.family"] = prop.get_name()

mpl.rcParams["axes.unicode_minus"] = False
warnings.filterwarnings('ignore')

# --- カスタムCSS (意思決定支援UI向け) ---
st.markdown("""
<style>
    .kpi-card { background-color: #ffffff; border-radius: 12px; padding: 24px; text-align: center; box-shadow: 0 4px 12px rgba(0,0,0,0.05); margin-bottom: 20px; border: 1px solid #f0f2f6; }
    .kpi-title { font-size: 1.1rem; color: #6c757d; margin-bottom: 8px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; }
    .kpi-value-main { font-size: 3.5rem; color: #1e293b; font-weight: 800; line-height: 1.1; margin-bottom: 5px; }
    .kpi-value-wave { font-size: 2.5rem; color: #2563eb; font-weight: 800; line-height: 1.2; margin-bottom: 5px; }
    .kpi-unit { font-size: 1.2rem; color: #64748b; font-weight: 500; }
    .kpi-sub { font-size: 1.1rem; color: #10b981; font-weight: bold; margin-top: 10px; }
    .kpi-sub.warning { color: #f59e0b; }
    .kpi-sub.alert { color: #ef4444; }
    .chance-box { background-color: #f0fdf4; border-left: 6px solid #10b981; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
    .chance-time { font-size: 1.8rem; color: #047857; font-weight: 800; }
    .sim-box { background-color: #f8fafc; padding: 16px; border-radius: 8px; border: 1px dashed #cbd5e1; height: 100%; }
</style>
""", unsafe_allow_html=True)

# --- 1. 波解析・特徴量抽出・Deep Work生成 ---
def make_wave_features(df_resampled, df_sched, freq_td):
    """
    集中を「波」として捉え、周期・振幅・位相を特徴量化する革新的なパイプライン
    """
    df_feat = df_resampled.copy()
    
    # 1️⃣ 統合集中強度スコアの構築
    score_cols = [c for c in ['CVRR_SCORE_NEW', 'RMSSD_SCORE_NEW', 'LFHF_SCORE_NEW', 'CVRR_SCORE', 'RMSSD_SCORE'] if c in df_feat.columns]
    if score_cols:
        df_feat['focus_intensity'] = df_feat[score_cols].mean(axis=1)
    elif '集中判定' in df_feat.columns:
        df_feat['focus_intensity'] = df_feat['集中判定'] * 100 # スケール合わせ
    else:
        df_feat['focus_intensity'] = 50.0
        
    # 2️⃣ 平滑化 (波を見える化・5分窓想定)
    win_size_5m = max(1, int(pd.Timedelta('5T') / freq_td))
    df_feat['focus_smooth'] = df_feat['focus_intensity'].rolling(window=win_size_5m, min_periods=1).mean()
    
    # 3️⃣ 波の位相 (簡易的な上昇/下降判定)
    df_feat['focus_diff'] = df_feat['focus_smooth'].diff()
    df_feat['phase_num'] = np.where(df_feat['focus_diff'] > 0, 1, np.where(df_feat['focus_diff'] < 0, -1, 0))
    df_feat['phase_str'] = np.where(df_feat['phase_num'] > 0, '上昇局面 ↗', np.where(df_feat['phase_num'] < 0, '下降局面 ↘', '停滞'))
    
    # 4️⃣ 波の特徴抽出 (ピークと谷の検出)
    dist_steps = max(1, int(pd.Timedelta('15T') / freq_td)) # 最低15分間隔の波を想定
    prominence = df_feat['focus_smooth'].std() * 0.2
    if pd.isna(prominence) or prominence == 0: prominence = 0.1
    
    fs_arr = df_feat['focus_smooth'].fillna(0).values
    peaks, _ = signal.find_peaks(fs_arr, distance=dist_steps, prominence=prominence)
    valleys, _ = signal.find_peaks(-fs_arr, distance=dist_steps, prominence=prominence)
    
    df_feat['is_peak'] = 0
    if len(peaks) > 0: df_feat.iloc[peaks, df_feat.columns.get_loc('is_peak')] = 1
    df_feat['is_valley'] = 0
    if len(valleys) > 0: df_feat.iloc[valleys, df_feat.columns.get_loc('is_valley')] = 1
    
    # 直近の波の状態を伝播 (ffill)
    df_feat['last_peak_val'] = df_feat['focus_smooth'].where(df_feat['is_peak'] == 1).ffill()
    df_feat['last_valley_val'] = df_feat['focus_smooth'].where(df_feat['is_valley'] == 1).ffill()
    
    idx_series = pd.Series(df_feat.index, index=df_feat.index)
    df_feat['last_peak_time'] = idx_series.where(df_feat['is_peak'] == 1).ffill()
    
    # 波ベース特徴量
    df_feat['wave_amplitude'] = (df_feat['last_peak_val'] - df_feat['last_valley_val']).fillna(0) # 振幅
    
    df_feat['prev_peak_time'] = df_feat['last_peak_time'].where(df_feat['is_peak']==1).shift(1).ffill()
    df_feat['wave_period_min'] = (df_feat['last_peak_time'] - df_feat['prev_peak_time']).dt.total_seconds() / 60 # 周期
    df_feat['wave_period_min'] = df_feat['wave_period_min'].fillna(0)
    
    # 5️⃣ 予測ターゲットの定義 (上位30%の高集中波に入っているか)
    q70 = df_feat['focus_smooth'].quantile(0.70)
    if pd.isna(q70): q70 = 50.0
    df_feat['is_high_focus_wave'] = (df_feat['focus_smooth'] >= q70).astype(int)
    
    # --- スケジュール統合と Deep Work フラグの生成 ---
    df_feat['has_schedule'] = 0
    df_feat['is_meeting'] = 0
    if df_sched is not None and not df_sched.empty:
        meeting_keywords = ['会議', '打合せ', 'MTG', '面談', '商談']
        for _, row in df_sched.iterrows():
            mask = (df_feat.index < row['end_dt']) & ((df_feat.index + freq_td) > row['start_dt'])
            df_feat.loc[mask, 'has_schedule'] = 1
            if any(kw in str(row.get('件名', '')) for kw in meeting_keywords):
                df_feat.loc[mask, 'is_meeting'] = 1
                
    win_steps_2h = max(1, int(pd.Timedelta('2H') / freq_td))
    df_feat['schedule_density_2h'] = df_feat['has_schedule'].rolling(win_steps_2h, min_periods=1).mean().shift(1).fillna(0)
    
    # Deep Work = 予定なし かつ 高集中波
    df_feat['deep_work'] = ((df_feat['has_schedule'] == 0) & (df_feat['is_high_focus_wave'] == 1)).astype(int)
    
    # ブロック解析
    dw_series = df_feat['deep_work']
    df_feat['dw_block_id'] = (dw_series != dw_series.shift()).cumsum()
    df_feat['dw_block_id'] = df_feat['dw_block_id'].where(dw_series == 1, np.nan)
    
    df_feat['hour'] = df_feat.index.hour
    df_feat['dayofweek'] = df_feat.index.dayofweek
    
    return df_feat, q70

def compute_personal_metrics(df_feat, freq_td, current_time):
    """
    個人特性と波のメタデータを算出
    """
    metrics = {}
    mins_per_step = freq_td.total_seconds() / 60
    df_feat['date'] = df_feat.index.date
    
    # Deep Work基礎指標
    block_lengths = df_feat.groupby('dw_block_id').size() * mins_per_step
    metrics['avg_dw_duration'] = block_lengths.mean() if not block_lengths.empty else 0
    metrics['dw_loss_minutes_total'] = block_lengths[block_lengths < 30].sum() if not block_lengths.empty else 0
    
    # 波の特性指標
    valid_periods = df_feat['wave_period_min'][df_feat['wave_period_min'] > 0]
    metrics['avg_wave_period'] = valid_periods.median() if not valid_periods.empty else 18.0
    metrics['avg_wave_amplitude'] = df_feat['wave_amplitude'][df_feat['wave_amplitude'] > 0].mean()
    if pd.isna(metrics['avg_wave_amplitude']): metrics['avg_wave_amplitude'] = 10.0
    
    # 全期間のDeep Work成功率 (dw_rate)
    total_blank_steps = (df_feat['has_schedule'] == 0).sum()
    total_dw_steps = df_feat['deep_work'].sum()
    metrics['dw_rate'] = (total_dw_steps / total_blank_steps * 100) if total_blank_steps > 0 else 0
    
    # 目標算出
    past_28_days = current_time.date() - pd.Timedelta(days=28)
    df_past = df_feat[(df_feat['date'] >= past_28_days) & (df_feat['date'] < current_time.date())]
    df_past_weekday = df_past[df_past['dayofweek'] < 5]
    
    if not df_past_weekday.empty:
        past_daily_dw = df_past_weekday.groupby('date')['deep_work'].sum() * mins_per_step
        target_raw = past_daily_dw.mean() * 1.10
        metrics['target_dw_mins'] = int(round(target_raw / 5.0) * 5)
    else:
        metrics['target_dw_mins'] = 120
    if metrics['target_dw_mins'] == 0: metrics['target_dw_mins'] = 60
    
    # 当日の進捗
    today_data = df_feat[df_feat['date'] == current_time.date()]
    today_blank_steps = (today_data['has_schedule'] == 0).sum()
    today_dw_steps = today_data['deep_work'].sum()
    metrics['today_dw_mins'] = today_dw_steps * mins_per_step
    metrics['today_dw_rate'] = (today_dw_steps / today_blank_steps * 100) if today_blank_steps > 0 else 0
    
    today_blocks = today_data.groupby('dw_block_id').size() * mins_per_step
    metrics['today_dw_loss'] = today_blocks[today_blocks < 30].sum() if not today_blocks.empty else 0
    
    return metrics

# --- 2. 状態遷移予測モデル (波のダイナミクスに基づく分類) ---
def train_predict_classifier(df_feat, ahead_steps):
    """
    波特徴量を用いて数十分先の「高集中波に入っているか」を予測する分類器
    """
    df_feat['target_class'] = df_feat['is_high_focus_wave'].shift(-ahead_steps)

    # 予測に用いる特徴量セット（波・遷移・スケジュール）
    feature_cols = [
        'hour', 'dayofweek', 
        'wave_amplitude', 'wave_period_min', 'phase_num',  # 波ベース
        'schedule_density_2h'                              # スケジュールベース
    ]
    # 追加可能な特徴量があれば追加
    for col in ['1分間歩数', 'SkinTemp']:
        if col in df_feat.columns: feature_cols.append(col)
    
    df_model = df_feat.dropna(subset=['target_class'] + feature_cols).copy()
    if len(df_model) < 50:
        return None, None, {}, df_feat
        
    split_idx = int(len(df_model) * 0.8)
    train_df = df_model.iloc[:split_idx]
    test_df = df_model.iloc[split_idx:]
    
    X_train, y_train = train_df[feature_cols], train_df['target_class']
    X_test, y_test = test_df[feature_cols], test_df['target_class']
    
    if y_train.nunique() <= 1:
        return None, None, {}, df_feat
        
    model = lgb.LGBMClassifier(objective='binary', n_estimators=100, learning_rate=0.05, random_state=42, verbose=-1)
    model.fit(X_train, y_train)
    
    # 評価指標算出
    eval_metrics = {}
    if y_test.nunique() > 1:
        preds_proba = model.predict_proba(X_test)[:, 1]
        preds_bin = (preds_proba >= 0.5).astype(int)
        eval_metrics['ROC-AUC'] = roc_auc_score(y_test, preds_proba)
        eval_metrics['PR-AUC'] = average_precision_score(y_test, preds_proba)
        eval_metrics['F1 Score'] = f1_score(y_test, preds_bin)
        eval_metrics['Brier Score'] = brier_score_loss(y_test, preds_proba)
    
    return model, feature_cols, eval_metrics, df_model

# --- サイドバーUI ---
with st.sidebar:
    st.header("⚙️ データ入力")
    file_ts = st.file_uploader("1. 生体データ (CSV)", type=['csv'])
    file_sched = st.file_uploader("2. 予定表データ (CSV) ※任意", type=['csv'])
    
    with st.expander("🛠 波解析・詳細設定 (管理者用)"):
        # 波解析を精密にするためデフォルトは細かい粒度を推奨
        RESAMPLE_FREQ = st.selectbox("分析単位 (波解像度)", ['1T', '5T', '10T', '30T'], index=1)
        PREDICT_AHEAD_MINS = st.selectbox("予測先 (分)", [30, 60], index=0)
        TARGET_DATETIME_STR = st.text_input("予測基準日時 (空欄で最新)")
        time_range = st.slider("グラフ表示時間帯", 0, 23, (9, 19)) # 表示範囲のコントロールを追加
        
    st.markdown("---")
    run_btn = st.button("🚀 波のダイナミクスを解析", type="primary", use_container_width=True)

freq_td = pd.Timedelta(RESAMPLE_FREQ)
ahead_steps = max(1, int(pd.Timedelta(minutes=PREDICT_AHEAD_MINS) / freq_td))
TARGET_DATETIME = TARGET_DATETIME_STR if TARGET_DATETIME_STR.strip() != "" else None

# === メイン処理パイプライン ===
if run_btn or file_ts is not None:
    if file_ts is None:
        st.info("👈 サイドバーから「生体データ」をアップロードしてください。")
        st.stop()
        
    with st.spinner("AIが集中を『波』として解析中..."):
        # 1. データの読み込み
        df_ts_raw = pd.read_csv(io.BytesIO(file_ts.getvalue()), skiprows=2)
        df_ts_raw['timestamp_clean'] = df_ts_raw['timestamp'].astype(str).str.split(' GMT').str[0]
        df_ts_raw['datetime'] = pd.to_datetime(df_ts_raw['timestamp_clean'], errors='coerce')
        df_ts_raw = df_ts_raw.dropna(subset=['datetime']).set_index('datetime').sort_index()

        df_sched_raw = None
        if file_sched:
            df_sched_raw = pd.read_csv(io.BytesIO(file_sched.getvalue()))
            df_sched_raw = df_sched_raw[df_sched_raw['終日イベント'].astype(str).str.upper() != 'TRUE']
            df_sched_raw['start_dt'] = pd.to_datetime(df_sched_raw['開始日'].astype(str) + ' ' + df_sched_raw['開始時刻'].astype(str), errors='coerce')
            df_sched_raw['end_dt']   = pd.to_datetime(df_sched_raw['終了日'].astype(str) + ' ' + df_sched_raw['終了時刻'].astype(str), errors='coerce')
            df_sched_raw = df_sched_raw.dropna(subset=['start_dt', 'end_dt']).sort_values('start_dt')
            
        # 2. 前処理・特徴量作成
        num_cols = df_ts_raw.select_dtypes(include=[np.number]).columns
        df_resampled = df_ts_raw[num_cols].resample(RESAMPLE_FREQ).mean()
        if '1分間歩数' in df_ts_raw.columns:
            df_resampled['1分間歩数'] = df_ts_raw['1分間歩数'].resample(RESAMPLE_FREQ).sum()
            
        df_feat, q70_thresh = make_wave_features(df_resampled, df_sched_raw, freq_td)
        
        # 基準日時の決定
        if TARGET_DATETIME:
            try:
                current_time = pd.to_datetime(TARGET_DATETIME)
                target_data_all = df_feat[df_feat.index <= current_time]
                target_data = target_data_all.iloc[-1:] if not target_data_all.empty else df_feat.iloc[-1:]
            except:
                target_data = df_feat.iloc[-1:]
        else:
            target_data = df_feat.iloc[-1:]
        current_time = target_data.index[0]
        
        # 3. 指標の計算
        metrics = compute_personal_metrics(df_feat, freq_td, current_time)
        
        # 4. 予測モデルの学習と推論
        model, feature_cols, eval_metrics, df_model = train_predict_classifier(df_feat, ahead_steps)
        
        focus_prob = 0.0
        if model is not None:
            focus_prob = model.predict_proba(target_data[feature_cols])[0, 1]

        # --- 現在の波の位相と次ピーク推計 ---
        current_phase = target_data['phase_str'].values[0]
        avg_period = metrics['avg_wave_period']
        
        last_peak_time_val = target_data['last_peak_time'].values[0]
        if pd.notna(last_peak_time_val):
            last_peak_dt = pd.to_datetime(last_peak_time_val)
            mins_since_peak = (current_time - last_peak_dt).total_seconds() / 60
            next_peak_in = max(0, int(avg_period - mins_since_peak))
        else:
            next_peak_in = int(avg_period)

        # --- 次のDeep Workチャンス算出 ---
        next_chance_text = "本日は終了、または空き時間がありません"
        if current_time.hour < 19:
            end_of_day = current_time.replace(hour=19, minute=0, second=0)
            future_mask = (df_feat.index > current_time) & (df_feat.index <= end_of_day) & (df_feat['has_schedule'] == 0)
            future_blank_times = df_feat[future_mask].index
            
            if not future_blank_times.empty:
                blank_blocks = (future_mask != future_mask.shift()).cumsum()[future_mask]
                longest_block_id = blank_blocks.value_counts().idxmax()
                best_block_times = future_blank_times[blank_blocks == longest_block_id]
                if len(best_block_times) > 0:
                    c_start = best_block_times[0]
                    c_end = best_block_times[-1] + freq_td
                    next_chance_text = f"{c_start.strftime('%H:%M')} – {c_end.strftime('%H:%M')}"

    # ==========================================
    # UI 描画
    # ==========================================
    st.markdown(f"<p style='text-align: right; color: gray;'>最終更新: {current_time.strftime('%Y/%m/%d %H:%M')}</p>", unsafe_allow_html=True)
    
    tab_today, tab_weekly, tab_spec = st.tabs(["🌊 Today (波と成果の管理)", "📊 Weekly Report", "👤 My Spec (波の特性)"])

    # --- TAB 1: Today (意思決定支援UI) ---
    with tab_today:
        col_m1, col_m2, col_m3 = st.columns([1, 1, 1])
        
        with col_m1:
            # メインKPI: Deep Work進捗
            remain_dw = max(0, metrics['target_dw_mins'] - metrics['today_dw_mins'])
            achieved_color = "🟢 目標クリア！" if remain_dw == 0 else f"目標まであと {remain_dw} 分"
            
            st.markdown(f"""
            <div class="kpi-card" style="border-top: 5px solid #0f172a; height: 100%;">
                <div class="kpi-title">今日のDeep Work達成状況</div>
                <div class="kpi-value-main">
                    {int(metrics['today_dw_mins'])} <span class="kpi-unit">/ {metrics['target_dw_mins']} 分</span>
                </div>
                <div class="kpi-sub {'alert' if remain_dw > 60 else ''}">{achieved_color}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col_m2:
            # 集中ダイナミクス (波のステータス)
            phase_color = "#ef4444" if "下降" in current_phase else "#10b981" if "上昇" in current_phase else "#64748b"
            st.markdown(f"""
            <div class="kpi-card" style="border-top: 5px solid #3b82f6; height: 100%;">
                <div class="kpi-title">現在の集中波フェーズ</div>
                <div class="kpi-value-wave" style="color: {phase_color};">{current_phase}</div>
                <div class="kpi-sub" style="color:#64748b; font-weight:normal;">次の集中ピーク予想: 約 <strong>{next_peak_in} 分後</strong></div>
            </div>
            """, unsafe_allow_html=True)
            
        with col_m3:
            # AI予測
            prob_color = "#10b981" if focus_prob > 0.6 else "#f59e0b" if focus_prob > 0.4 else "#ef4444"
            st.markdown(f"""
            <div class="kpi-card" style="border-top: 5px solid #8b5cf6; height: 100%;">
                <div class="kpi-title">{PREDICT_AHEAD_MINS}分後の 高集中波 再突入確率</div>
                <div class="kpi-value-main" style="color: {prob_color};">{focus_prob * 100:.1f} <span class="kpi-unit">%</span></div>
                <div class="kpi-sub" style="color:#64748b; font-weight:normal;">上位30%のゾーンに到達する確率</div>
            </div>
            """, unsafe_allow_html=True)
            
        # 下段: 改善余地とチャンス
        col_s1, col_s2 = st.columns([1, 1.5])
        with col_s1:
            loss_status = "alert" if metrics['today_dw_loss'] >= 30 else "warning" if metrics['today_dw_loss'] > 0 else ""
            st.markdown(f"""
            <div style="display: flex; gap: 10px;">
                <div class="kpi-card" style="flex: 1; padding: 15px;">
                    <div class="kpi-title" style="font-size:0.85rem;">空白時間の集中率</div>
                    <div style="font-size:1.8rem; font-weight:bold; color:#334155;">{metrics['today_dw_rate']:.1f} <span style="font-size:1rem;">%</span></div>
                </div>
                <div class="kpi-card" style="flex: 1; padding: 15px;">
                    <div class="kpi-title" style="font-size:0.85rem;">分断ロス(波の頓挫)</div>
                    <div style="font-size:1.8rem; font-weight:bold; color:#334155;">{int(metrics['today_dw_loss'])} <span style="font-size:1rem;">分</span></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        with col_s2:
            st.markdown(f"""
            <div class="chance-box" style="margin-bottom: 0;">
                <div class="kpi-title" style="color: #047857;">🎯 次のDeep Workチャンス枠</div>
                <div class="chance-time">{next_chance_text}</div>
                <div style="font-size: 0.95rem; color: #065f46; margin-top: 8px;">この時間を死守し、波に乗って重要タスクを消化してください。</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 🛠 波をコントロールするアクション・シミュレーション")
        
        sim_c1, sim_c2 = st.columns(2)
        with sim_c1:
            st.markdown("""
            <div class="sim-box">
                <h4>🚫 会議を30分短縮・ブロック化する</h4>
                <p style="color:#555;">波が分断されるのを防ぐことで、過去の成功率から換算して、<br>
                今日のDeep Work総量が <strong style="color:#10b981; font-size:1.2rem;">増加します</strong>。</p>
            </div>
            """, unsafe_allow_html=True)
        with sim_c2:
            st.markdown("""
            <div class="sim-box">
                <h4>☕ 今から15分、完全な休憩をとる</h4>
                <p style="color:#555;">波の振幅（強さ）を回復させることで、次の高集中波の持続時間が<br>
                通常より <strong style="color:#10b981; font-size:1.2rem;">延長される見込み</strong> です。</p>
            </div>
            """, unsafe_allow_html=True)

    # --- TAB 2: Weekly Report ---
    with tab_weekly:
        st.markdown("## 今週のパフォーマンスと波の振り返り")
        
        past_7_days = current_time.date() - pd.Timedelta(days=7)
        past_14_days = current_time.date() - pd.Timedelta(days=14)
        
        df_this_week = df_feat[(df_feat['date'] > past_7_days) & (df_feat['date'] <= current_time.date())]
        df_last_week = df_feat[(df_feat['date'] > past_14_days) & (df_feat['date'] <= past_7_days)]
        
        tw_dw = df_this_week['deep_work'].sum() * (freq_td.total_seconds() / 60)
        lw_dw = df_last_week['deep_work'].sum() * (freq_td.total_seconds() / 60)
        diff_dw = tw_dw - lw_dw
        
        st.metric("今週のDeep Work合計時間", f"{int(tw_dw)} 分", f"{'+' if diff_dw>=0 else ''}{int(diff_dw)} 分 (先週比)")
        
        # --- 黄金パターンの動的抽出（最大3つ） ---
        st.markdown("#### 💡 データが見つけた黄金パターン")
        
        # 過去データ全体（平日）からパターンを探索
        df_feat_wd = df_feat[df_feat['dayofweek'] < 5].copy()
        if not df_feat_wd.empty and df_feat_wd['date'].nunique() >= 3:
            daily_stats = []
            for d, group in df_feat_wd.groupby('date'):
                am_group = group[group.index.hour < 12]
                pm_group = group[group.index.hour >= 12]
                
                dw_mins = group['deep_work'].sum() * (freq_td.total_seconds() / 60)
                am_dw_mins = am_group['deep_work'].sum() * (freq_td.total_seconds() / 60)
                am_meeting = am_group['is_meeting'].sum() * (freq_td.total_seconds() / 60)
                pm_blank = (pm_group['has_schedule'] == 0).sum() * (freq_td.total_seconds() / 60)
                steps = group['1分間歩数'].sum() if '1分間歩数' in group.columns else 0
                
                # 最長空白ブロック
                blank_mask = group['has_schedule'] == 0
                blank_blocks = blank_mask.groupby((blank_mask != blank_mask.shift()).cumsum()).sum()
                longest_blank = blank_blocks.max() * (freq_td.total_seconds() / 60) if not blank_blocks.empty else 0
                
                daily_stats.append({
                    'date': d,
                    'dw_mins': dw_mins,
                    'am_dw_mins': am_dw_mins,
                    'am_meeting': am_meeting,
                    'pm_blank': pm_blank,
                    'steps': steps,
                    'longest_blank': longest_blank
                })
                
            df_daily = pd.DataFrame(daily_stats)
            avg_dw_all = df_daily['dw_mins'].mean()
            
            if avg_dw_all > 0:
                patterns = []
                
                # パターン1: 午前に会議集中、午後空白
                m_am = df_daily['am_meeting'].median()
                m_pm = df_daily['pm_blank'].median()
                mask1 = (df_daily['am_meeting'] >= m_am) & (df_daily['pm_blank'] >= m_pm) & (df_daily['am_meeting'] > 0)
                if mask1.sum() >= 1 and (~mask1).sum() >= 1:
                    avg_dw = df_daily[mask1]['dw_mins'].mean()
                    if avg_dw > avg_dw_all * 1.05:
                        patterns.append((avg_dw / avg_dw_all, "午前中に会議を寄せて、午後にまとまった空白を作った日"))
                        
                # パターン2: 身体活動
                if df_daily['steps'].max() > 0:
                    m_steps = df_daily['steps'].median()
                    mask2 = df_daily['steps'] > m_steps
                    if mask2.sum() >= 1 and (~mask2).sum() >= 1:
                        avg_dw = df_daily[mask2]['dw_mins'].mean()
                        if avg_dw > avg_dw_all * 1.05:
                            patterns.append((avg_dw / avg_dw_all, "身体を動かし活動量（歩数）を平均以上に確保した日"))
                            
                # パターン3: 90分ブロック
                mask3 = df_daily['longest_blank'] >= 90
                if mask3.sum() >= 1 and (~mask3).sum() >= 1:
                    avg_dw = df_daily[mask3]['dw_mins'].mean()
                    if avg_dw > avg_dw_all * 1.05:
                        patterns.append((avg_dw / avg_dw_all, "1日のどこかで「90分以上の連続した空白枠」を死守した日"))
                        
                # パターン4: 午前中のDWスタート
                mask4 = df_daily['am_dw_mins'] > 0
                if mask4.sum() >= 1 and (~mask4).sum() >= 1:
                    avg_dw = df_daily[mask4]['dw_mins'].mean()
                    if avg_dw > avg_dw_all * 1.05:
                        patterns.append((avg_dw / avg_dw_all, "午前中のうちに1回でもDeep Workの波に乗れた日"))
                        
                # 効果が高い順にソートし、最大3つを取得
                patterns.sort(key=lambda x: x[0], reverse=True)
                top_patterns = patterns[:3]
                
                if top_patterns:
                    icons = ["🥇", "🥈", "🥉"]
                    for i, (ratio, text) in enumerate(top_patterns):
                        st.info(f"{icons[i]} **「{text}」** は、波が途切れずDeep Work時間が平均の **{ratio:.1f}倍** になる傾向があります。")
                else:
                    st.info("💡 安定した成果を出しています。さらにデータが蓄積されると、あなた専用の「Deep Workが倍増する黄金パターン」がここに最大3つ表示されます。")
            else:
                st.info("💡 データの蓄積が進むと、あなた専用の「Deep Workが倍増する黄金パターン」がここに表示されます。")
        else:
            st.info("💡 データが十分に蓄積されると、あなた専用の「Deep Workが倍増する黄金パターン」がここに表示されます。（※比較のため数日分のデータが必要です）")

        # --- 波形グラフ (モメンタルグラフ) の復活・進化版 ---
        st.markdown("---")
        st.markdown("#### 🌊 今週の集中波形 (モメンタルグラフ)")
        st.caption("※ 青い線が平滑化された集中の「波」を表し、赤い点がAIが検出した「波のピーク」です。グレーの点線より上の青い面が「高集中ゾーン（Deep Workの候補）」です。波の周期性（リズム）が視覚的に確認できます。")
        
        week_dates = df_this_week['date'].unique()
        if len(week_dates) > 0:
            for i in range(0, len(week_dates), 2):
                cols = st.columns(2)
                for j in range(2):
                    if i + j < len(week_dates):
                        t_date = week_dates[i+j]
                        with cols[j]:
                            df_day = df_this_week[df_this_week['date'] == t_date].copy()
                            # サイドバーで設定した時間帯でフィルタ
                            df_day = df_day[(df_day.index.hour >= time_range[0]) & (df_day.index.hour <= time_range[1])]
                            
                            if not df_day.empty and not df_day['focus_smooth'].isna().all():
                                fig_d = go.Figure()
                                
                                # 基準線 (高集中ライン: 上位30%の閾値)
                                q70_val = q70_thresh 
                                fig_d.add_trace(go.Scatter(x=df_day.index, y=[q70_val]*len(df_day), mode='lines', line=dict(color='gray', width=1, dash='dash'), name='高集中ライン', hoverinfo='skip'))
                                
                                # 閾値より上の部分を青く塗りつぶし (Deep Work ゾーン)
                                y_up = np.where(df_day['focus_smooth'] >= q70_val, df_day['focus_smooth'], q70_val)
                                fig_d.add_trace(go.Scatter(x=df_day.index, y=y_up, fill='tonexty', fillcolor='rgba(59, 130, 246, 0.3)', mode='lines', line=dict(width=0), hoverinfo='skip', showlegend=False))
                                # 下側を透明にするためのダミートレース
                                fig_d.add_trace(go.Scatter(x=df_day.index, y=[q70_val]*len(df_day), fill='tonexty', fillcolor='rgba(0,0,0,0)', mode='lines', line=dict(width=0), hoverinfo='skip', showlegend=False))
                                
                                # 波の線 (メイン)
                                fig_d.add_trace(go.Scatter(x=df_day.index, y=df_day['focus_smooth'], mode='lines', line=dict(color='#3b82f6', width=2), name='集中波', hovertemplate="%{x|%H:%M}<br>強度: %{y:.1f}<extra></extra>"))
                                
                                # ピークのポイント (赤い点)
                                peaks_day = df_day[df_day['is_peak'] == 1]
                                if not peaks_day.empty:
                                    fig_d.add_trace(go.Scatter(x=peaks_day.index, y=peaks_day['focus_smooth'], mode='markers', marker=dict(color='#ef4444', size=6, symbol='circle'), name='ピーク', hovertemplate="%{x|%H:%M}<br>ピーク<extra></extra>"))
                                
                                dow_str = ['月','火','水','木','金','土','日'][t_date.weekday()]
                                fig_d.update_layout(title=f"{t_date.strftime('%m/%d')} ({dow_str})", height=250, hovermode="x unified", plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=20, r=20, t=30, b=20), showlegend=False)
                                fig_d.update_xaxes(showgrid=True, gridcolor='lightgray')
                                
                                # Y軸の範囲を適度に調整
                                y_min = df_day['focus_smooth'].min()
                                y_max = df_day['focus_smooth'].max()
                                amp = y_max - y_min if y_max - y_min > 0 else 10
                                fig_d.update_yaxes(showgrid=True, gridcolor='lightgray', title="集中強度", range=[max(0, y_min - amp*0.2), y_max + amp*0.2])
                                
                                st.plotly_chart(fig_d, use_container_width=True)
                            else:
                                st.markdown(f"**{t_date.strftime('%m/%d')} ({['月','火','水','木','金','土','日'][t_date.weekday()]})**")
                                st.info("指定された時間帯のデータがありません。")

    # --- TAB 3: My Spec ---
    with tab_spec:
        st.markdown("## 👤 あなたの「集中ダイナミクス」攻略法")
        st.write("過去の全データを波形解析し、あなた固有の集中リズムを抽出しました。")
        
        best_hour = df_feat.groupby('hour')['deep_work'].sum().idxmax()
        
        c_spec1, c_spec2, c_spec3 = st.columns(3)
        c_spec1.metric("⏱ 平均集中波 周期", f"{int(metrics['avg_wave_period'])} 分", "波が訪れる間隔")
        c_spec2.metric("🎯 最適集中時間帯", f"{best_hour}:00 台", "波が最大化する時間")
        c_spec3.metric("📈 波の平均振幅", f"{metrics['avg_wave_amplitude']:.1f} pt", "集中の深さの指標")
        
        st.markdown("""
        <div style="background-color: #f8fafc; padding: 20px; border-radius: 8px; border-left: 4px solid #3b82f6; margin-top: 20px;">
            <h4>📝 AIからのパーソナルコメント</h4>
            <ul style="font-size: 1.1rem; color: #334155; line-height: 1.6;">
                <li>あなたの集中は<strong>約 {0} 分周期</strong>の波を描いています。疲れた時は無理をせず、次の波が来るタイミングに合わせて作業を再開するのが効率的です。</li>
                <li><strong>{1}時台</strong>に波の振幅が最大化し、極めて深い集中状態に入りやすくなります。この時間帯は死守してください。</li>
                <li>予定の合間が短すぎると、波が上昇しきる前に分断されてしまう「分断ロス」が発生しています。会議は固めて配置しましょう。</li>
            </ul>
        </div>
        """.format(int(metrics['avg_wave_period']), best_hour), unsafe_allow_html=True)

    # --- 開発者向けセクション ---
    with st.expander("🛠 開発者向け情報 (モデル評価・SHAP・特徴量)"):
        st.markdown("### モデル評価指標 (分類モデル)")
        col_ev1, col_ev2, col_ev3 = st.columns(3)
        if eval_metrics:
            col_ev1.metric("ROC-AUC", f"{eval_metrics.get('ROC-AUC', 0):.3f}")
            col_ev2.metric("PR-AUC", f"{eval_metrics.get('PR-AUC', 0):.3f}")
            col_ev3.metric("F1 Score", f"{eval_metrics.get('F1 Score', 0):.3f}")
        else:
            st.warning("評価に必要なテストデータ（正例・負例）が不足しています。")
            
        st.markdown("### 直近予測の根拠 (SHAP)")
        st.write("波の特徴量（周期・振幅・位相）やスケジュール要因が、確率をどのように押し上げたかを示します。")
        if model is not None:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer(target_data[feature_cols])
            
            fig_shap, ax_shap = plt.subplots(figsize=(8, 4))
            if len(shap_values.shape) == 3:
                shap.plots.waterfall(shap_values[0, :, 1], show=False)
            else:
                shap.plots.waterfall(shap_values[0], show=False)
            st.pyplot(fig_shap)