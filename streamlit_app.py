# -*- coding: utf-8 -*-
"""
Deep Work 最大化・集中波解析アプリ (Wave Dynamics + Fatigue & Arousal Layer)
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

# --- カスタムCSS ---
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

# ==========================================
# 🛑 A. 疲労・回復レイヤー (1分粒度)
# ==========================================
def compute_fatigue_features(df_1min, steps_col=None):
    df = df_1min.copy()
    
    # 1) 疲労スコアの定義
    has_rmssd = 'RMSSD_SCORE_NEW' in df.columns
    has_tp = 'TP_SCORE_NEW' in df.columns
    
    if has_rmssd and has_tp:
        df['fatigue_score'] = 0.6 * df['RMSSD_SCORE_NEW'] + 0.4 * df['TP_SCORE_NEW']
    elif has_rmssd:
        df['fatigue_score'] = df['RMSSD_SCORE_NEW']
    elif has_tp:
        df['fatigue_score'] = df['TP_SCORE_NEW']
    else:
        df['fatigue_score'] = 50.0  # fallback
        
    # 平滑化 (EWMA span=10)
    df['fatigue_smooth'] = df['fatigue_score'].ewm(span=10, min_periods=1).mean()
    
    # 2) 疲労のドリフト (直近60分の傾き近似: FIRフィルタ)
    # 線形回帰の傾きを高速化するため重みベクトルを使用
    w60 = np.arange(60) - 29.5
    var_x = np.sum(w60**2)
    w60 = w60 / var_x if var_x > 0 else w60
    df['fatigue_drift_60m'] = df['fatigue_smooth'].rolling(60, min_periods=60).apply(lambda y: np.dot(w60, y), raw=True).fillna(0)
    
    # バンド分類
    q33 = df['fatigue_smooth'].quantile(0.33) if not df['fatigue_smooth'].isna().all() else 33.0
    q66 = df['fatigue_smooth'].quantile(0.66) if not df['fatigue_smooth'].isna().all() else 66.0
    df['fatigue_level_band'] = np.where(df['fatigue_smooth'] >= q66, '高', np.where(df['fatigue_smooth'] <= q33, '低', '中'))
    
    # 3) 回復の推定 (歩数がある場合)
    if steps_col and steps_col in df.columns:
        df['rest_flag'] = np.where(df[steps_col] <= 5, 1, np.where(df[steps_col] >= 20, 0, np.nan))
        df['rest_flag'] = df['rest_flag'].ffill().fillna(0)
        
        # 安静ブロックID
        rest_blocks = (df['rest_flag'] != df['rest_flag'].shift()).cumsum()
        df['rest_block_id'] = rest_blocks.where(df['rest_flag'] == 1, np.nan)
    else:
        df['rest_flag'] = np.nan
        df['rest_block_id'] = np.nan
        
    return df

def compute_morning_residual(df_1min, date_col='date', tp_col="TP_SCORE_NEW", rest_flag_col="rest_flag"):
    results = []
    df = df_1min.copy()
    if 'date' not in df.columns: df['date'] = df.index.date
    
    for d, group in df.groupby('date'):
        morning = group[(group.index.hour >= 0) & (group.index.hour < 12)]
        if morning.empty or rest_flag_col not in morning.columns:
            continue
            
        rest_blocks = morning[rest_flag_col] == 1
        blocks = rest_blocks.groupby((rest_blocks != rest_blocks.shift()).cumsum())
        
        target_block = None
        for _, b in blocks:
            if b.iloc[0] == True:
                if len(b) >= 60:
                    target_block = b
                    break
                elif len(b) >= 30 and target_block is None:
                    target_block = b # 60分がない場合は30分で妥協
                    
        if target_block is not None and tp_col in morning.columns:
            tp_median = morning.loc[target_block.index, tp_col].median()
            results.append({'date': d, 'morning_tp_median': tp_median, 'rest_duration': len(target_block)})
            
    res_df = pd.DataFrame(results)
    if not res_df.empty and 'morning_tp_median' in res_df.columns:
        median_all = res_df['morning_tp_median'].median()
        mad = (res_df['morning_tp_median'] - median_all).abs().median()
        if mad == 0: mad = 1.0
        res_df['morning_residual_z'] = 0.6745 * (res_df['morning_tp_median'] - median_all) / mad
        res_df['morning_residual_band'] = np.where(res_df['morning_residual_z'] >= 1.0, '高',
                                          np.where(res_df['morning_residual_z'] <= -1.0, '低', '中'))
    return res_df

# ==========================================
# 🛑 B. 低覚醒レイヤー (1分粒度)
# ==========================================
def compute_low_arousal(df_1min, pr_col="PR_SCORE_NEW", steps_col=None):
    df = df_1min.copy()
    if pr_col not in df.columns:
        df['low_arousal'] = 0.0
        df['low_arousal_band'] = '低'
        df['low_arousal_rise_15m'] = 0.0
        return df
        
    # 5分窓の回帰傾き
    w5 = np.array([-2, -1, 0, 1, 2]) / 10.0
    slope = df[pr_col].rolling(5, min_periods=5).apply(lambda y: np.dot(w5, y), raw=True).fillna(0)
    
    eps = 0.02
    delta = np.maximum(0, -(slope + eps))
    
    alpha = 0.95
    k = 1.0
    
    low_arousal = np.zeros(len(df))
    dates = df.index.date
    steps = df[steps_col].values if steps_col and steps_col in df.columns else np.zeros(len(df))
    
    # 減衰積分
    for i in range(1, len(df)):
        if dates[i] != dates[i-1]:
            low_arousal[i] = 0
        else:
            current_alpha = alpha
            if steps[i] >= 20:
                current_alpha = 0.80 # 活動中は減衰を早める
            low_arousal[i] = current_alpha * low_arousal[i-1] + k * delta.iloc[i]
            
    df['low_arousal'] = low_arousal
    
    q33 = df['low_arousal'].quantile(0.33) if df['low_arousal'].max() > 0 else 0
    q66 = df['low_arousal'].quantile(0.66) if df['low_arousal'].max() > 0 else 0
    df['low_arousal_band'] = np.where(df['low_arousal'] >= q66, '高', np.where(df['low_arousal'] <= q33, '低', '中'))
    
    df['low_arousal_rise_15m'] = df['low_arousal'] - df['low_arousal'].shift(15).fillna(0)
    
    return df

def summarize_daily_condition(df_1min):
    m_res = compute_morning_residual(df_1min)
    
    daily = []
    df = df_1min.copy()
    if 'date' not in df.columns: df['date'] = df.index.date
    
    for d, group in df.groupby('date'):
        daytime = group[(group.index.hour >= 9) & (group.index.hour <= 19)]
        fatigue_load = daytime['fatigue_score'].sum() if 'fatigue_score' in daytime.columns else 0
        
        recovery = 0.0
        if 'rest_block_id' in group.columns and 'fatigue_smooth' in group.columns:
            for _, b in group.groupby('rest_block_id'):
                diff = b['fatigue_smooth'].iloc[0] - b['fatigue_smooth'].iloc[-1]
                if diff > 0: recovery += diff
                
        la_peak_time = "なし"
        if 'low_arousal' in group.columns and group['low_arousal'].max() > 0:
            peak_idx = group['low_arousal'].idxmax()
            la_peak_time = peak_idx.strftime('%H:%M')
            
        daily.append({
            '日付': d,
            '日中疲労負荷': int(fatigue_load),
            '安静回復量': round(recovery, 1),
            '低覚醒ピーク': la_peak_time
        })
        
    df_daily = pd.DataFrame(daily)
    if not m_res.empty:
        df_daily = pd.merge(df_daily, m_res[['date', 'morning_residual_band']], left_on='日付', right_on='date', how='left')
        df_daily = df_daily.rename(columns={'morning_residual_band': '朝の残疲労 (可能性)'}).drop(columns=['date'])
    else:
        df_daily['朝の残疲労 (可能性)'] = '不明'
        
    # 列順調整
    cols = ['日付', '朝の残疲労 (可能性)', '日中疲労負荷', '安静回復量', '低覚醒ピーク']
    return df_daily[[c for c in cols if c in df_daily.columns]]

# ==========================================
# 🌊 1. 波解析・特徴量抽出・Deep Work生成
# ==========================================
def make_wave_features(df_resampled, df_sched, freq_td):
    df_feat = df_resampled.copy()
    
    # 1️⃣ 統合集中強度スコアの構築
    focus_components = []
    if 'CVRR_SCORE_NEW' in df_feat.columns:
        focus_components.append(df_feat['CVRR_SCORE_NEW'])
    if 'RMSSD_SCORE_NEW' in df_feat.columns:
        focus_components.append(100 - df_feat['RMSSD_SCORE_NEW']) # 疲労指標を反転
    if 'LFHF_SCORE_NEW' in df_feat.columns:
        focus_components.append(df_feat['LFHF_SCORE_NEW'])
        
    if focus_components:
        df_feat['focus_intensity'] = pd.concat(focus_components, axis=1).mean(axis=1)
    elif '集中判定' in df_feat.columns:
        df_feat['focus_intensity'] = df_feat['集中判定'] * 100 
    else:
        df_feat['focus_intensity'] = 50.0
        
    win_size_5m = max(1, int(pd.Timedelta('5T') / freq_td))
    df_feat['focus_smooth'] = df_feat['focus_intensity'].rolling(window=win_size_5m, min_periods=1).mean()
    
    df_feat['focus_diff'] = df_feat['focus_smooth'].diff()
    df_feat['phase_num'] = np.where(df_feat['focus_diff'] > 0, 1, np.where(df_feat['focus_diff'] < 0, -1, 0))
    df_feat['phase_str'] = np.where(df_feat['phase_num'] > 0, '上昇局面 ↗', np.where(df_feat['phase_num'] < 0, '下降局面 ↘', '停滞'))
    
    dist_steps = max(1, int(pd.Timedelta('15T') / freq_td))
    prominence = df_feat['focus_smooth'].std() * 0.2
    if pd.isna(prominence) or prominence == 0: prominence = 0.1
    
    fs_arr = df_feat['focus_smooth'].fillna(0).values
    peaks, _ = signal.find_peaks(fs_arr, distance=dist_steps, prominence=prominence)
    valleys, _ = signal.find_peaks(-fs_arr, distance=dist_steps, prominence=prominence)
    
    df_feat['is_peak'] = 0
    if len(peaks) > 0: df_feat.iloc[peaks, df_feat.columns.get_loc('is_peak')] = 1
    df_feat['is_valley'] = 0
    if len(valleys) > 0: df_feat.iloc[valleys, df_feat.columns.get_loc('is_valley')] = 1
    
    df_feat['last_peak_val'] = df_feat['focus_smooth'].where(df_feat['is_peak'] == 1).ffill()
    df_feat['last_valley_val'] = df_feat['focus_smooth'].where(df_feat['is_valley'] == 1).ffill()
    
    idx_series = pd.Series(df_feat.index, index=df_feat.index)
    df_feat['last_peak_time'] = idx_series.where(df_feat['is_peak'] == 1).ffill()
    
    df_feat['wave_amplitude'] = (df_feat['last_peak_val'] - df_feat['last_valley_val']).fillna(0)
    
    df_feat['prev_peak_time'] = df_feat['last_peak_time'].where(df_feat['is_peak']==1).shift(1).ffill()
    df_feat['wave_period_min'] = (df_feat['last_peak_time'] - df_feat['prev_peak_time']).dt.total_seconds() / 60
    df_feat['wave_period_min'] = df_feat['wave_period_min'].fillna(0)
    
    q70 = df_feat['focus_smooth'].quantile(0.70)
    if pd.isna(q70): q70 = 50.0
    df_feat['is_high_focus_wave'] = (df_feat['focus_smooth'] >= q70).astype(int)
    
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
    
    df_feat['deep_work'] = ((df_feat['has_schedule'] == 0) & (df_feat['is_high_focus_wave'] == 1)).astype(int)
    
    dw_series = df_feat['deep_work']
    df_feat['dw_block_id'] = (dw_series != dw_series.shift()).cumsum()
    df_feat['dw_block_id'] = df_feat['dw_block_id'].where(dw_series == 1, np.nan)
    
    df_feat['hour'] = df_feat.index.hour
    df_feat['dayofweek'] = df_feat.index.dayofweek
    
    return df_feat, q70

def compute_personal_metrics(df_feat, freq_td, current_time):
    metrics = {}
    mins_per_step = freq_td.total_seconds() / 60
    df_feat['date'] = df_feat.index.date
    
    block_lengths = df_feat.groupby('dw_block_id').size() * mins_per_step
    metrics['avg_dw_duration'] = block_lengths.mean() if not block_lengths.empty else 0
    metrics['dw_loss_minutes_total'] = block_lengths[block_lengths < 30].sum() if not block_lengths.empty else 0
    
    valid_periods = df_feat['wave_period_min'][df_feat['wave_period_min'] > 0]
    metrics['avg_wave_period'] = valid_periods.median() if not valid_periods.empty else 18.0
    metrics['avg_wave_amplitude'] = df_feat['wave_amplitude'][df_feat['wave_amplitude'] > 0].mean()
    if pd.isna(metrics['avg_wave_amplitude']): metrics['avg_wave_amplitude'] = 10.0
    
    total_blank_steps = (df_feat['has_schedule'] == 0).sum()
    total_dw_steps = df_feat['deep_work'].sum()
    metrics['dw_rate'] = (total_dw_steps / total_blank_steps * 100) if total_blank_steps > 0 else 0
    
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
    
    today_data = df_feat[df_feat['date'] == current_time.date()]
    today_blank_steps = (today_data['has_schedule'] == 0).sum()
    today_dw_steps = today_data['deep_work'].sum()
    metrics['today_dw_mins'] = today_dw_steps * mins_per_step
    metrics['today_dw_rate'] = (today_dw_steps / today_blank_steps * 100) if today_blank_steps > 0 else 0
    
    today_blocks = today_data.groupby('dw_block_id').size() * mins_per_step
    metrics['today_dw_loss'] = today_blocks[today_blocks < 30].sum() if not today_blocks.empty else 0
    
    return metrics

def train_predict_classifier(df_feat, ahead_steps):
    df_feat['target_class'] = df_feat['is_high_focus_wave'].shift(-ahead_steps)

    feature_cols = ['hour', 'dayofweek', 'wave_amplitude', 'wave_period_min', 'phase_num', 'schedule_density_2h']
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
        RESAMPLE_FREQ = st.selectbox("分析単位 (波解像度)", ['1T', '5T', '10T', '30T'], index=1)
        PREDICT_AHEAD_MINS = st.selectbox("予測先 (分)", [30, 60], index=0)
        TARGET_DATETIME_STR = st.text_input("予測基準日時 (空欄で最新)")
        time_range = st.slider("グラフ表示時間帯", 0, 23, (9, 19))
        
    st.markdown("---")
    run_btn = st.button("🚀 コンディションを解析", type="primary", use_container_width=True)

freq_td = pd.Timedelta(RESAMPLE_FREQ)
ahead_steps = max(1, int(pd.Timedelta(minutes=PREDICT_AHEAD_MINS) / freq_td))
TARGET_DATETIME = TARGET_DATETIME_STR if TARGET_DATETIME_STR.strip() != "" else None

# === メイン処理パイプライン ===
if run_btn or file_ts is not None:
    if file_ts is None:
        st.info("👈 サイドバーから「生体データ」をアップロードしてください。")
        st.stop()
        
    with st.spinner("AIが集中・疲労・覚醒のダイナミクスを解析中..."):
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
            
        # 2. 1分粒度データと疲労・低覚醒レイヤーの作成
        num_cols = df_ts_raw.select_dtypes(include=[np.number]).columns
        df_1min = df_ts_raw[num_cols].resample('1T').mean()
        if '1分間歩数' in df_ts_raw.columns:
            df_1min['1分間歩数'] = df_ts_raw['1分間歩数'].resample('1T').sum()
        df_1min = df_1min.ffill(limit=5)
        
        steps_col_name = '1分間歩数' if '1分間歩数' in df_1min.columns else None
        df_1min = compute_fatigue_features(df_1min, steps_col=steps_col_name)
        df_1min = compute_low_arousal(df_1min, pr_col='PR_SCORE_NEW' if 'PR_SCORE_NEW' in df_1min.columns else None, steps_col=steps_col_name)

        # 3. 集中波解析用のリサンプルデータ作成
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
        
        # 4. 指標計算と推論
        metrics = compute_personal_metrics(df_feat, freq_td, current_time)
        model, feature_cols, eval_metrics, df_model = train_predict_classifier(df_feat, ahead_steps)
        
        focus_prob = 0.0
        if model is not None:
            focus_prob = model.predict_proba(target_data[feature_cols])[0, 1]

        # 現在の1分粒度コンディション取得
        current_1min = df_1min[df_1min.index <= current_time]
        cur_1m = current_1min.iloc[-1] if not current_1min.empty else df_1min.iloc[-1]
        
        fatigue_band = cur_1m.get('fatigue_level_band', '不明')
        fatigue_drift = cur_1m.get('fatigue_drift_60m', 0.0)
        drift_str = "蓄積中 ↗" if fatigue_drift > 0.05 else "回復傾向 ↘" if fatigue_drift < -0.05 else "横ばい →"
        
        la_band = cur_1m.get('low_arousal_band', '不明')
        la_rise = cur_1m.get('low_arousal_rise_15m', 0.0)
        la_str = "上昇中 ⚠️" if la_rise > 0.5 else "安定"

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

        # --- 生産性アクションの提案判定 ---
        is_focus_low = focus_prob < 0.4
        action_text = "現在のコンディションは安定しています。このまま波に乗ってDeep Workを進めましょう。"
        if la_band == '高' and is_focus_low:
            action_text = "集中力が低下し、眠気（低覚醒）が高まっています。短い歩行や軽いストレッチで脳をリフレッシュしましょう。"
        elif la_band == '高' and fatigue_band == '高':
            action_text = "疲労と眠気がピークに達しています。無理な作業は控え、完全な休息を取ることを強く推奨します。"
        elif la_band == '高' and fatigue_band == '低':
            action_text = "疲労は少ないですが、単調さから眠気が生じています。少し立ち上がって歩くなど、姿勢を変えてみましょう。"

    # ==========================================
    # UI 描画
    # ==========================================
    st.markdown(f"<p style='text-align: right; color: gray;'>最終更新: {current_time.strftime('%Y/%m/%d %H:%M')}</p>", unsafe_allow_html=True)
    
    tab_today, tab_weekly, tab_spec = st.tabs(["🌊 Today (波と成果の管理)", "📊 Weekly Report", "👤 My Spec (波の特性)"])

    # --- TAB 1: Today (意思決定支援UI) ---
    with tab_today:
        col_m1, col_m2, col_m3 = st.columns([1, 1, 1])
        with col_m1:
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
            phase_color = "#ef4444" if "下降" in current_phase else "#10b981" if "上昇" in current_phase else "#64748b"
            st.markdown(f"""
            <div class="kpi-card" style="border-top: 5px solid #3b82f6; height: 100%;">
                <div class="kpi-title">現在の集中波フェーズ</div>
                <div class="kpi-value-wave" style="color: {phase_color};">{current_phase}</div>
                <div class="kpi-sub" style="color:#64748b; font-weight:normal;">次の集中ピーク予想: 約 <strong>{next_peak_in} 分後</strong></div>
            </div>
            """, unsafe_allow_html=True)
            
        with col_m3:
            prob_color = "#10b981" if focus_prob > 0.6 else "#f59e0b" if focus_prob > 0.4 else "#ef4444"
            st.markdown(f"""
            <div class="kpi-card" style="border-top: 5px solid #8b5cf6; height: 100%;">
                <div class="kpi-title">{PREDICT_AHEAD_MINS}分後の 高集中波 再突入確率</div>
                <div class="kpi-value-main" style="color: {prob_color};">{focus_prob * 100:.1f} <span class="kpi-unit">%</span></div>
                <div class="kpi-sub" style="color:#64748b; font-weight:normal;">上位30%のゾーンに到達する確率</div>
            </div>
            """, unsafe_allow_html=True)
            
        # 🔋 追加: リアルタイム コンディション (疲労・覚醒)
        st.markdown("### 🔋 リアルタイム コンディション (疲労・覚醒)")
        col_c1, col_c2, col_c3 = st.columns([1, 1, 1.5])
        with col_c1:
            f_color = "#ef4444" if fatigue_band == '高' else "#10b981" if fatigue_band == '低' else "#f59e0b"
            st.markdown(f"""
            <div class="kpi-card" style="border-top: 5px solid {f_color}; padding: 15px;">
                <div class="kpi-title" style="font-size:0.9rem;">現在の疲労レベル</div>
                <div style="font-size:2rem; font-weight:bold; color:{f_color};">{fatigue_band}</div>
                <div style="font-size:0.9rem; color:#64748b;">トレンド: {drift_str}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col_c2:
            la_color = "#ef4444" if la_band == '高' else "#10b981" if la_band == '低' else "#f59e0b"
            st.markdown(f"""
            <div class="kpi-card" style="border-top: 5px solid {la_color}; padding: 15px;">
                <div class="kpi-title" style="font-size:0.9rem;">現在の低覚醒 (眠気)</div>
                <div style="font-size:2rem; font-weight:bold; color:{la_color};">{la_band}</div>
                <div style="font-size:0.9rem; color:#64748b;">状態: {la_str}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col_c3:
            st.markdown(f"""
            <div class="chance-box" style="margin-bottom: 0; background-color: #f8fafc; border-left: 6px solid #3b82f6;">
                <div class="kpi-title" style="color: #1e293b; font-size:0.9rem;">🤖 AIアクション提案</div>
                <div style="font-size: 1.1rem; color: #334155; margin-top: 10px; font-weight: 500;">{action_text}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        col_s1, col_s2 = st.columns([1, 1.5])
        with col_s1:
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

    # --- TAB 2: Weekly Report ---
    with tab_weekly:
        st.markdown("## 今週のパフォーマンスとコンディション振り返り")
        
        past_7_days = current_time.date() - pd.Timedelta(days=7)
        past_14_days = current_time.date() - pd.Timedelta(days=14)
        
        df_this_week = df_feat[(df_feat['date'] > past_7_days) & (df_feat['date'] <= current_time.date())]
        df_last_week = df_feat[(df_feat['date'] > past_14_days) & (df_feat['date'] <= past_7_days)]
        
        tw_dw = df_this_week['deep_work'].sum() * (freq_td.total_seconds() / 60)
        lw_dw = df_last_week['deep_work'].sum() * (freq_td.total_seconds() / 60)
        diff_dw = tw_dw - lw_dw
        
        st.metric("今週のDeep Work合計時間", f"{int(tw_dw)} 分", f"{'+' if diff_dw>=0 else ''}{int(diff_dw)} 分 (先週比)")
        
        # 追加: 日別コンディション サマリー
        st.markdown("#### 📅 日別コンディション・サマリー (疲労と回復)")
        df_daily_cond = summarize_daily_condition(df_1min)
        if not df_daily_cond.empty:
            st.dataframe(df_daily_cond, use_container_width=True)
        else:
            st.info("コンディションサマリーを計算するためのデータが不足しています。")

        st.markdown("#### 💡 データが見つけた黄金パターン")
        
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
                
                blank_mask = group['has_schedule'] == 0
                blank_blocks = blank_mask.groupby((blank_mask != blank_mask.shift()).cumsum()).sum()
                longest_blank = blank_blocks.max() * (freq_td.total_seconds() / 60) if not blank_blocks.empty else 0
                
                daily_stats.append({
                    'date': d, 'dw_mins': dw_mins, 'am_dw_mins': am_dw_mins,
                    'am_meeting': am_meeting, 'pm_blank': pm_blank,
                    'steps': steps, 'longest_blank': longest_blank
                })
                
            df_daily = pd.DataFrame(daily_stats)
            avg_dw_all = df_daily['dw_mins'].mean()
            
            if avg_dw_all > 0:
                patterns = []
                m_am = df_daily['am_meeting'].median()
                m_pm = df_daily['pm_blank'].median()
                mask1 = (df_daily['am_meeting'] >= m_am) & (df_daily['pm_blank'] >= m_pm) & (df_daily['am_meeting'] > 0)
                if mask1.sum() >= 1 and (~mask1).sum() >= 1:
                    avg_dw = df_daily[mask1]['dw_mins'].mean()
                    if avg_dw > avg_dw_all * 1.05:
                        patterns.append((avg_dw / avg_dw_all, "午前中に会議を寄せて、午後にまとまった空白を作った日"))
                        
                if df_daily['steps'].max() > 0:
                    m_steps = df_daily['steps'].median()
                    mask2 = df_daily['steps'] > m_steps
                    if mask2.sum() >= 1 and (~mask2).sum() >= 1:
                        avg_dw = df_daily[mask2]['dw_mins'].mean()
                        if avg_dw > avg_dw_all * 1.05:
                            patterns.append((avg_dw / avg_dw_all, "身体を動かし活動量（歩数）を平均以上に確保した日"))
                            
                mask3 = df_daily['longest_blank'] >= 90
                if mask3.sum() >= 1 and (~mask3).sum() >= 1:
                    avg_dw = df_daily[mask3]['dw_mins'].mean()
                    if avg_dw > avg_dw_all * 1.05:
                        patterns.append((avg_dw / avg_dw_all, "1日のどこかで「90分以上の連続した空白枠」を死守した日"))
                        
                mask4 = df_daily['am_dw_mins'] > 0
                if mask4.sum() >= 1 and (~mask4).sum() >= 1:
                    avg_dw = df_daily[mask4]['dw_mins'].mean()
                    if avg_dw > avg_dw_all * 1.05:
                        patterns.append((avg_dw / avg_dw_all, "午前中のうちに1回でもDeep Workの波に乗れた日"))
                        
                patterns.sort(key=lambda x: x[0], reverse=True)
                top_patterns = patterns[:3]
                
                if top_patterns:
                    icons = ["🥇", "🥈", "🥉"]
                    for i, (ratio, text) in enumerate(top_patterns):
                        st.info(f"{icons[i]} **「{text}」** は、波が途切れずDeep Work時間が平均の **{ratio:.1f}倍** になる傾向があります。")
                else:
                    st.info("💡 安定した成果を出しています。さらにデータが蓄積されると、あなた専用の「黄金パターン」がここに表示されます。")
        
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
                            df_day = df_day[(df_day.index.hour >= time_range[0]) & (df_day.index.hour <= time_range[1])]
                            
                            if not df_day.empty and not df_day['focus_smooth'].isna().all():
                                fig_d = go.Figure()
                                q70_val = q70_thresh 
                                fig_d.add_trace(go.Scatter(x=df_day.index, y=[q70_val]*len(df_day), mode='lines', line=dict(color='gray', width=1, dash='dash'), name='高集中ライン', hoverinfo='skip'))
                                y_up = np.where(df_day['focus_smooth'] >= q70_val, df_day['focus_smooth'], q70_val)
                                fig_d.add_trace(go.Scatter(x=df_day.index, y=y_up, fill='tonexty', fillcolor='rgba(59, 130, 246, 0.3)', mode='lines', line=dict(width=0), hoverinfo='skip', showlegend=False))
                                fig_d.add_trace(go.Scatter(x=df_day.index, y=[q70_val]*len(df_day), fill='tonexty', fillcolor='rgba(0,0,0,0)', mode='lines', line=dict(width=0), hoverinfo='skip', showlegend=False))
                                fig_d.add_trace(go.Scatter(x=df_day.index, y=df_day['focus_smooth'], mode='lines', line=dict(color='#3b82f6', width=2), name='集中波', hovertemplate="%{x|%H:%M}<br>強度: %{y:.1f}<extra></extra>"))
                                peaks_day = df_day[df_day['is_peak'] == 1]
                                if not peaks_day.empty:
                                    fig_d.add_trace(go.Scatter(x=peaks_day.index, y=peaks_day['focus_smooth'], mode='markers', marker=dict(color='#ef4444', size=6, symbol='circle'), name='ピーク', hovertemplate="%{x|%H:%M}<br>ピーク<extra></extra>"))
                                dow_str = ['月','火','水','木','金','土','日'][t_date.weekday()]
                                fig_d.update_layout(title=f"{t_date.strftime('%m/%d')} ({dow_str})", height=250, hovermode="x unified", plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=20, r=20, t=30, b=20), showlegend=False)
                                fig_d.update_xaxes(showgrid=True, gridcolor='lightgray')
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
    with st.expander("🛠 開発者向け情報 (モデル評価・パラメータ・特徴量)"):
        st.markdown("### 🔬 新規レイヤー パラメータ定義")
        st.write("- **Fatigue Score (疲労)**: 0.6 * RMSSD_SCORE_NEW + 0.4 * TP_SCORE_NEW (10分 EWMA)")
        st.write("- **Low Arousal (低覚醒)**: PR_SCORE_NEW の5分窓傾き（低下方向のみ）を減衰積分 (alpha=0.95, eps=0.02)")
        
        # 疲労と低覚醒のプロット (当日)
        today_1min = df_1min[df_1min.index.date == current_time.date()]
        if not today_1min.empty:
            fig_cond = go.Figure()
            fig_cond.add_trace(go.Scatter(x=today_1min.index, y=today_1min['fatigue_smooth'], name='疲労(Smooth)', line=dict(color='#ef4444')))
            if 'low_arousal' in today_1min.columns:
                # 視認性向上のためスケールを調整してプロット
                scaled_la = today_1min['low_arousal'] * (today_1min['fatigue_smooth'].max() / (today_1min['low_arousal'].max() + 0.1))
                fig_cond.add_trace(go.Scatter(x=today_1min.index, y=scaled_la, name='低覚醒(スケーリング済)', line=dict(color='#8b5cf6')))
            fig_cond.update_layout(title="本日の疲労・低覚醒 推移", height=300, margin=dict(l=20, r=20, t=30, b=20), hovermode="x unified")
            st.plotly_chart(fig_cond, use_container_width=True)

        st.markdown("### モデル評価指標 (分類モデル)")
        col_ev1, col_ev2, col_ev3 = st.columns(3)
        if eval_metrics:
            col_ev1.metric("ROC-AUC", f"{eval_metrics.get('ROC-AUC', 0):.3f}")
            col_ev2.metric("PR-AUC", f"{eval_metrics.get('PR-AUC', 0):.3f}")
            col_ev3.metric("F1 Score", f"{eval_metrics.get('F1 Score', 0):.3f}")
        else:
            st.warning("評価に必要なテストデータ（正例・負例）が不足しています。")
            
        st.markdown("### 直近予測の根拠 (SHAP)")
        if model is not None:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer(target_data[feature_cols])
            fig_shap, ax_shap = plt.subplots(figsize=(8, 4))
            if len(shap_values.shape) == 3:
                shap.plots.waterfall(shap_values[0, :, 1], show=False)
            else:
                shap.plots.waterfall(shap_values[0], show=False)
            st.pyplot(fig_shap)