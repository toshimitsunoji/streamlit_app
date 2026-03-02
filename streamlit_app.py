# -*- coding: utf-8 -*-
"""
NeuroDesign　- 個人の深思考マネジメント -
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
st.set_page_config(page_title="NeuroDesign - 深思考マネジメント -", layout="wide", initial_sidebar_state="expanded")

# 日本語フォントの設定
font_path = Path(__file__).parent / "assets" / "fonts" / "NotoSansCJKjp-Regular.otf"
if font_path.exists():
    fm.fontManager.addfont(str(font_path))
    prop = fm.FontProperties(fname=str(font_path))
    mpl.rcParams["font.family"] = prop.get_name()

mpl.rcParams["axes.unicode_minus"] = False
warnings.filterwarnings('ignore')

# ==========================================
# 🛑 パラメータ設定 (深思考保全エンジン)
# ==========================================
MIN_DEEP_DURATION = 60      # 深思考とみなす最低ブロック長(分)
FOCUS_STREAK_MIN = 5        # 高集中が連続すべき最低時間(分)
GAP_TOLERANCE = 5           # 許容する中断時間(分)
MAX_DAILY_DEEP_BLOCKS = 1   # 1日に提示する深思考枠の最大数
DISPLAY_DEEP_DURATION = 90  # 画面に提示する深思考枠の上限時間(分)

# --- カスタムCSS ---
st.markdown("""
<style>
    .kpi-card { background-color: #ffffff; border-radius: 12px; padding: 24px; text-align: center; box-shadow: 0 4px 12px rgba(0,0,0,0.05); margin-bottom: 20px; border: 1px solid #f0f2f6; }
    .kpi-title { font-size: 1.1rem; color: #6c757d; margin-bottom: 8px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; }
    .kpi-value-main { font-size: 3.5rem; color: #1e293b; font-weight: 800; line-height: 1.1; margin-bottom: 5px; }
    .kpi-value-wave { font-size: 2.5rem; color: #2563eb; font-weight: 800; line-height: 1.2; margin-bottom: 5px; }
    .kpi-unit { font-size: 1.2rem; color: #64748b; font-weight: 500; }
    .chance-box { background-color: #f0fdf4; border-left: 6px solid #10b981; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 🛑 A. 疲労・回復レイヤー (1分粒度)
# ==========================================
def compute_fatigue_features(df_1min, steps_col=None):
    df = df_1min.copy()
    has_rmssd = 'RMSSD_SCORE_NEW' in df.columns
    has_tp = 'TP_SCORE_NEW' in df.columns
    
    if has_rmssd and has_tp:
        df['fatigue_score'] = 0.6 * df['RMSSD_SCORE_NEW'] + 0.4 * df['TP_SCORE_NEW']
    elif has_rmssd:
        df['fatigue_score'] = df['RMSSD_SCORE_NEW']
    elif has_tp:
        df['fatigue_score'] = df['TP_SCORE_NEW']
    else:
        df['fatigue_score'] = 50.0 
        
    df['fatigue_smooth'] = df['fatigue_score'].ewm(span=10, min_periods=1).mean()
    
    w60 = np.arange(60) - 29.5
    var_x = np.sum(w60**2)
    w60 = w60 / var_x if var_x > 0 else w60
    df['fatigue_drift_60m'] = df['fatigue_smooth'].rolling(60, min_periods=60).apply(lambda y: np.dot(w60, y), raw=True).fillna(0)
    
    q33 = df['fatigue_smooth'].quantile(0.33) if not df['fatigue_smooth'].isna().all() else 33.0
    q66 = df['fatigue_smooth'].quantile(0.66) if not df['fatigue_smooth'].isna().all() else 66.0
    df['fatigue_level_band'] = np.where(df['fatigue_smooth'] >= q66, '高', np.where(df['fatigue_smooth'] <= q33, '低', '中'))
    
    if steps_col and steps_col in df.columns:
        df['rest_flag'] = np.where(df[steps_col] <= 5, 1, np.where(df[steps_col] >= 20, 0, np.nan))
        df['rest_flag'] = df['rest_flag'].ffill().fillna(0)
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
        if morning.empty or rest_flag_col not in morning.columns: continue
            
        rest_blocks = morning[rest_flag_col] == 1
        blocks = rest_blocks.groupby((rest_blocks != rest_blocks.shift()).cumsum())
        
        target_block = None
        for _, b in blocks:
            if b.iloc[0] == True:
                if len(b) >= 60:
                    target_block = b
                    break
                elif len(b) >= 30 and target_block is None:
                    target_block = b 
                    
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
        
    w5 = np.array([-2, -1, 0, 1, 2]) / 10.0
    slope = df[pr_col].rolling(5, min_periods=5).apply(lambda y: np.dot(w5, y), raw=True).fillna(0)
    eps = 0.02
    delta = np.maximum(0, -(slope + eps))
    alpha = 0.95
    k = 1.0
    
    low_arousal = np.zeros(len(df))
    dates = df.index.date
    steps = df[steps_col].values if steps_col and steps_col in df.columns else np.zeros(len(df))
    
    for i in range(1, len(df)):
        if dates[i] != dates[i-1]:
            low_arousal[i] = 0
        else:
            current_alpha = 0.80 if steps[i] >= 20 else alpha
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
            '日付': d, '日中疲労負荷': int(fatigue_load), '安静回復量': round(recovery, 1), '低覚醒ピーク': la_peak_time
        })
        
    df_daily = pd.DataFrame(daily)
    if not m_res.empty:
        df_daily = pd.merge(df_daily, m_res[['date', 'morning_residual_band']], left_on='日付', right_on='date', how='left')
        df_daily = df_daily.rename(columns={'morning_residual_band': '朝の残疲労 (可能性)'}).drop(columns=['date'])
    else:
        df_daily['朝の残疲労 (可能性)'] = '不明'
        
    cols = ['日付', '朝の残疲労 (可能性)', '日中疲労負荷', '安静回復量', '低覚醒ピーク']
    return df_daily[[c for c in cols if c in df_daily.columns]]

# ==========================================
# 🌊 1. 波解析・特徴量抽出
# ==========================================
def make_wave_features(df_resampled, df_sched, freq_td):
    df_feat = df_resampled.copy()
    focus_components = []
    if 'CVRR_SCORE_NEW' in df_feat.columns: focus_components.append(df_feat['CVRR_SCORE_NEW'])
    if 'RMSSD_SCORE_NEW' in df_feat.columns: focus_components.append(100 - df_feat['RMSSD_SCORE_NEW']) 
    if 'LFHF_SCORE_NEW' in df_feat.columns: focus_components.append(df_feat['LFHF_SCORE_NEW'])
        
    if focus_components: df_feat['focus_intensity'] = pd.concat(focus_components, axis=1).mean(axis=1)
    elif '集中判定' in df_feat.columns: df_feat['focus_intensity'] = df_feat['集中判定'] * 100 
    else: df_feat['focus_intensity'] = 50.0
        
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
    df_feat['hour'] = df_feat.index.hour
    df_feat['dayofweek'] = df_feat.index.dayofweek
    return df_feat, q70

def compute_personal_metrics(df_feat, freq_td):
    metrics = {}
    valid_periods = df_feat['wave_period_min'][df_feat['wave_period_min'] > 0]
    metrics['avg_wave_period'] = valid_periods.median() if not valid_periods.empty else 18.0
    metrics['avg_wave_amplitude'] = df_feat['wave_amplitude'][df_feat['wave_amplitude'] > 0].mean()
    if pd.isna(metrics['avg_wave_amplitude']): metrics['avg_wave_amplitude'] = 10.0
    return metrics

def train_predict_classifier(df_feat, ahead_steps):
    df_feat['target_class'] = df_feat['is_high_focus_wave'].shift(-ahead_steps)
    feature_cols = ['hour', 'dayofweek', 'wave_amplitude', 'wave_period_min', 'phase_num', 'schedule_density_2h']
    for col in ['1分間歩数', 'SkinTemp']:
        if col in df_feat.columns: feature_cols.append(col)
    
    df_model = df_feat.dropna(subset=['target_class'] + feature_cols).copy()
    if len(df_model) < 50: return None, None, {}, df_feat
        
    split_idx = int(len(df_model) * 0.8)
    train_df = df_model.iloc[:split_idx]
    test_df = df_model.iloc[split_idx:]
    
    X_train, y_train = train_df[feature_cols], train_df['target_class']
    X_test, y_test = test_df[feature_cols], test_df['target_class']
    if y_train.nunique() <= 1: return None, None, {}, df_feat
        
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

# ==========================================
# 🎯 C. 深思考保全エンジン (NeuroDesign Core)
# ==========================================
def add_1min_focus_wave(df_1min):
    df = df_1min.copy()
    focus_components = []
    if 'CVRR_SCORE_NEW' in df.columns: focus_components.append(df['CVRR_SCORE_NEW'])
    if 'RMSSD_SCORE_NEW' in df.columns: focus_components.append(100 - df['RMSSD_SCORE_NEW'])
    if 'LFHF_SCORE_NEW' in df.columns: focus_components.append(df['LFHF_SCORE_NEW'])
    
    if focus_components: df['focus_intensity'] = pd.concat(focus_components, axis=1).mean(axis=1)
    elif '集中判定' in df.columns: df['focus_intensity'] = df['集中判定'] * 100
    else: df['focus_intensity'] = 50.0
        
    df['focus_smooth'] = df['focus_intensity'].rolling(window=5, min_periods=1).mean()
    q70 = df['focus_smooth'].quantile(0.70) if not df['focus_smooth'].isna().all() else 50.0
    df['is_high_focus_wave'] = (df['focus_smooth'] >= q70).astype(int)
    return df

def extract_free_blocks(df_sched_raw, start_dt, end_dt):
    idx = pd.date_range(start=start_dt.ceil('1T'), end=end_dt.floor('1T'), freq='1T')
    df_dummy = pd.DataFrame(index=idx)
    df_dummy = df_dummy[(df_dummy.index.hour >= 9) & (df_dummy.index.hour < 19)]
    df_dummy = df_dummy[df_dummy.index.dayofweek < 5]
    
    df_dummy['has_schedule'] = 0
    if df_sched_raw is not None and not df_sched_raw.empty:
        for _, row in df_sched_raw.iterrows():
            mask = (df_dummy.index >= row['start_dt']) & (df_dummy.index < row['end_dt'])
            df_dummy.loc[mask, 'has_schedule'] = 1
            
    sched_blocks = (df_dummy['has_schedule'] != df_dummy['has_schedule'].shift()).cumsum()
    for b_id, b_df in df_dummy[df_dummy['has_schedule'] == 1].groupby(sched_blocks):
        if len(b_df) <= GAP_TOLERANCE:
            df_dummy.loc[b_df.index, 'has_schedule'] = 0
            
    free_blocks_id = (df_dummy['has_schedule'] != df_dummy['has_schedule'].shift()).cumsum()
    blocks = []
    
    for b_id, b_df in df_dummy[df_dummy['has_schedule'] == 0].groupby(free_blocks_id):
        for d, d_df in b_df.groupby(b_df.index.date):
            duration = len(d_df)
            if duration >= MIN_DEEP_DURATION:
                blocks.append({
                    'date': d, 'start_dt': d_df.index[0], 'end_dt': d_df.index[-1] + pd.Timedelta(minutes=1),
                    'duration': duration, 'hour': d_df.index[0].hour
                })
    return blocks

def evaluate_deep_success(df_1min, block, fatigue_drift_th):
    b_df = df_1min[(df_1min.index >= block['start_dt']) & (df_1min.index < block['end_dt'])]
    if len(b_df) < MIN_DEEP_DURATION: return 0
        
    focus = b_df.get('is_high_focus_wave', pd.Series(0, index=b_df.index))
    if focus.sum() == 0: cond_A = False
    else:
        focus_streaks = focus.groupby((focus != focus.shift()).cumsum()).sum()
        cond_A = focus_streaks.max() >= FOCUS_STREAK_MIN
        
    cond_B = True 
    
    fatigue = b_df.get('fatigue_smooth', pd.Series(0, index=b_df.index)).dropna()
    if len(fatigue) > 10:
        x = np.arange(len(fatigue))
        slope = np.polyfit(x, fatigue.values, 1)[0]
    else: slope = 0
    cond_C = slope <= fatigue_drift_th
    
    return 1 if (cond_A and cond_B and cond_C) else 0

def compute_hourly_profile(df_1min, df_sched_raw, current_time):
    past_start = current_time - pd.Timedelta(weeks=4)
    past_blocks = extract_free_blocks(df_sched_raw, past_start, current_time)
    df_1m = add_1min_focus_wave(df_1min)
    
    slopes = []
    for b in past_blocks:
        b_df = df_1m[(df_1m.index >= b['start_dt']) & (df_1m.index < b['end_dt'])]
        fatigue = b_df.get('fatigue_smooth', pd.Series(dtype=float)).dropna()
        if len(fatigue) > 10:
            slopes.append(np.polyfit(np.arange(len(fatigue)), fatigue.values, 1)[0])
    fatigue_drift_th = np.percentile(slopes, 80) if slopes else 0.5
    
    for b in past_blocks:
        b['deep_success'] = evaluate_deep_success(df_1m, b, fatigue_drift_th)
        
    df_blocks = pd.DataFrame(past_blocks)
    if not df_blocks.empty and 'hour' in df_blocks.columns:
        success_rate = df_blocks.groupby('hour')['deep_success'].mean()
    else:
        success_rate = pd.Series(0, index=np.arange(9, 19))
        
    df_1m['hour'] = df_1m.index.hour
    daytime_df = df_1m[(df_1m['hour'] >= 9) & (df_1m['hour'] <= 18)]
    
    mean_fat = daytime_df.groupby('hour')['fatigue_smooth'].mean() if 'fatigue_smooth' in daytime_df else pd.Series(0)
    mean_aro = daytime_df.groupby('hour')['low_arousal'].mean() if 'low_arousal' in daytime_df else pd.Series(0)
        
    profile = pd.DataFrame({
        'success_rate': success_rate, 'mean_fat': mean_fat, 'mean_aro': mean_aro
    }).reindex(np.arange(9, 19)).fillna(0)
    
    def z_score(s):
        if s.std() == 0: return s - s.mean()
        return (s - s.mean()) / s.std()
        
    profile['suitability'] = z_score(profile['success_rate']) - z_score(profile['mean_fat']) - z_score(profile['mean_aro'])
    return profile

def optimize_today_deep_block(df_sched_raw, hourly_profile, current_time, cur_1m):
    """
    当日の空き時間から、深思考に最も適した1枠だけを抽出する
    """
    today_start = current_time.replace(hour=9, minute=0, second=0, microsecond=0)
    today_end = current_time.replace(hour=19, minute=0, second=0, microsecond=0)
    
    if current_time >= today_end:
        return None, []
        
    search_start = max(current_time, today_start)
    future_blocks = extract_free_blocks(df_sched_raw, search_start, today_end)
    
    if not future_blocks:
        return None, []
        
    best_block = None
    best_score = -float('inf')
    reasons = []
    
    cur_fatigue = cur_1m.get('fatigue_smooth', 50.0)
    cur_arousal = cur_1m.get('low_arousal', 0.0)
    
    for b in future_blocks:
        h = b['hour']
        suitability = hourly_profile.loc[h, 'suitability'] if h in hourly_profile.index else 0
        
        penalty = 0
        has_prev_meeting = False
        has_next_meeting = False
        if df_sched_raw is not None and not df_sched_raw.empty:
            prev_s = df_sched_raw[(df_sched_raw['end_dt'] > b['start_dt'] - pd.Timedelta(minutes=30)) & (df_sched_raw['end_dt'] <= b['start_dt'])]
            next_s = df_sched_raw[(df_sched_raw['start_dt'] >= b['end_dt']) & (df_sched_raw['start_dt'] < b['end_dt'] + pd.Timedelta(minutes=30))]
            if not prev_s.empty:
                penalty -= 0.5
                has_prev_meeting = True
            if not next_s.empty:
                penalty -= 0.5
                has_next_meeting = True
                
        mins_to_start = (b['start_dt'] - current_time).total_seconds() / 60
        recovery_bonus = (mins_to_start / 60.0) * 0.5 if (cur_fatigue > 60 or cur_arousal > 2.0) else 0
        
        score = suitability + penalty + recovery_bonus
        
        if score > best_score:
            best_score = score
            b_reasons = []
            if suitability > 0:
                b_reasons.append("この時間帯は過去の深思考成功率が高い傾向にあります。")
            if not has_prev_meeting and not has_next_meeting:
                b_reasons.append("前後に会議がなく、集中を分断されるリスクが低いです。")
            else:
                b_reasons.append("限られた空き時間の中で最も条件が良い枠です。")
                
            if recovery_bonus > 0:
                b_reasons.append("現在の疲労・低覚醒状態から回復するまでの猶予が確保されています。")
            else:
                fatigue_h = hourly_profile.loc[h, 'mean_fat'] if h in hourly_profile.index else 50
                if fatigue_h < 50:
                    b_reasons.append("この時間帯は疲労・低覚醒リスクが低い傾向にあります。")
                    
            reasons = b_reasons[:3]
            best_block = b

    if best_block:
        display_duration = min(best_block['duration'], DISPLAY_DEEP_DURATION)
        best_block['display_end_dt'] = best_block['start_dt'] + pd.Timedelta(minutes=display_duration)
        return best_block, reasons
    
    return None, []

# --- サイドバーUI ---
with st.sidebar:
    st.header("⚙️ データ入力")
    file_ts = st.file_uploader("1. 生体データ (CSV)", type=['csv'])
    file_sched = st.file_uploader("2. 予定表データ (CSV) ※必須", type=['csv'])
    
    with st.expander("🛠 波解析・詳細設定 (管理者用)"):
        RESAMPLE_FREQ = st.selectbox("分析単位 (波解像度)", ['1T', '5T', '10T', '30T'], index=1)
        PREDICT_AHEAD_MINS = st.selectbox("予測先 (分)", [30, 60], index=0)
        TARGET_DATETIME_STR = st.text_input("予測基準日時 (空欄で最新)")
        
        st.markdown("**📅 分析対象フィルタ**")
        dow_options = ["月", "火", "水", "木", "金", "土", "日"]
        selected_dows = st.multiselect("分析対象とする曜日", dow_options, default=dow_options[0:5])
        time_range = st.slider("グラフ表示時間帯", 0, 23, (9, 19))
        selected_dow_indices = [dow_options.index(d) for d in selected_dows]
        
    st.markdown("---")
    run_btn = st.button("🚀 NeuroDesign 実行", type="primary", use_container_width=True)

freq_td = pd.Timedelta(RESAMPLE_FREQ)
ahead_steps = max(1, int(pd.Timedelta(minutes=PREDICT_AHEAD_MINS) / freq_td))
TARGET_DATETIME = TARGET_DATETIME_STR if TARGET_DATETIME_STR.strip() != "" else None

# === メイン処理パイプライン ===
if run_btn or file_ts is not None:
    if file_ts is None:
        st.info("👈 サイドバーから「生体データ」をアップロードしてください。")
        st.stop()
        
    with st.spinner("深思考マネジメントエンジンを起動中..."):
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
            
        num_cols = df_ts_raw.select_dtypes(include=[np.number]).columns
        df_1min = df_ts_raw[num_cols].resample('1T').mean()
        if '1分間歩数' in df_ts_raw.columns:
            df_1min['1分間歩数'] = df_ts_raw['1分間歩数'].resample('1T').sum()
        df_1min = df_1min.ffill(limit=5)
        
        steps_col_name = '1分間歩数' if '1分間歩数' in df_1min.columns else None
        df_1min = compute_fatigue_features(df_1min, steps_col=steps_col_name)
        df_1min = compute_low_arousal(df_1min, pr_col='PR_SCORE_NEW' if 'PR_SCORE_NEW' in df_1min.columns else None, steps_col=steps_col_name)

        df_resampled = df_ts_raw[num_cols].resample(RESAMPLE_FREQ).mean()
        if '1分間歩数' in df_ts_raw.columns:
            df_resampled['1分間歩数'] = df_ts_raw['1分間歩数'].resample(RESAMPLE_FREQ).sum()
            
        df_feat, q70_thresh = make_wave_features(df_resampled, df_sched_raw, freq_td)
        
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
        
        metrics = compute_personal_metrics(df_feat, freq_td)
        model, feature_cols, eval_metrics, df_model = train_predict_classifier(df_feat, ahead_steps)
        focus_prob = model.predict_proba(target_data[feature_cols])[0, 1] if model is not None else 0.0

        current_1min = df_1min[df_1min.index <= current_time]
        cur_1m = current_1min.iloc[-1] if not current_1min.empty else df_1min.iloc[-1]
        
        fatigue_band = cur_1m.get('fatigue_level_band', '不明')
        fatigue_drift = cur_1m.get('fatigue_drift_60m', 0.0)
        drift_str = "蓄積中 ↗" if fatigue_drift > 0.05 else "回復傾向 ↘" if fatigue_drift < -0.05 else "横ばい →"
        la_band = cur_1m.get('low_arousal_band', '不明')
        la_rise = cur_1m.get('low_arousal_rise_15m', 0.0)
        la_str = "上昇中 ⚠️" if la_rise > 0.5 else "安定"

        current_phase = target_data['phase_str'].values[0]
        avg_period = metrics['avg_wave_period']
        last_peak_time_val = target_data['last_peak_time'].values[0]
        if pd.notna(last_peak_time_val):
            mins_since_peak = (current_time - pd.to_datetime(last_peak_time_val)).total_seconds() / 60
            next_peak_in = max(0, int(avg_period - mins_since_peak))
        else:
            next_peak_in = int(avg_period)

        is_focus_low = focus_prob < 0.4
        action_text = "現在のコンディションは安定しています。このまま波に乗ってDeep Workを進めましょう。"
        if la_band == '高' and is_focus_low: action_text = "集中力が低下し、眠気（低覚醒）が高まっています。短い歩行や軽いストレッチで脳をリフレッシュしましょう。"
        elif la_band == '高' and fatigue_band == '高': action_text = "疲労と眠気がピークに達しています。無理な作業は控え、完全な休息を取ることを強く推奨します。"
        elif la_band == '高' and fatigue_band == '低': action_text = "疲労は少ないですが、単調さから眠気が生じています。少し立ち上がって歩くなど、姿勢を変えてみましょう。"

        # --- 最適な深思考枠の算出 ---
        best_deep_block = None
        deep_reasons = []
        if file_sched is not None:
            hourly_profile = compute_hourly_profile(df_1min, df_sched_raw, current_time)
            best_deep_block, deep_reasons = optimize_today_deep_block(df_sched_raw, hourly_profile, current_time, cur_1m)

    # ==========================================
    # UI 描画
    # ==========================================
    st.markdown(f"<p style='text-align: right; color: gray;'>最終更新: {current_time.strftime('%Y/%m/%d %H:%M')}</p>", unsafe_allow_html=True)
    
    tab_today, tab_weekly, tab_spec = st.tabs(["🌊 Today", "📊 Weekly Report", "👤 My Spec"])

    # --- TAB 1: Today ---
    with tab_today:
        st.markdown("### 今日の深思考")
        
        if best_deep_block:
            start_str = best_deep_block['start_dt'].strftime('%H:%M')
            end_str = best_deep_block['display_end_dt'].strftime('%H:%M')
            
            reasons_html = '<br>'.join(['・' + r for r in deep_reasons])
            
            st.markdown(f"""
            <div style="background-color: #f8fafc; border-left: 6px solid #8b5cf6; padding: 20px; border-radius: 8px; margin-bottom: 30px;">
                <div style="font-size: 2.5rem; font-weight: 800; color: #1e293b; margin-bottom: 10px;">
                    🔥 {start_str} – {end_str}
                </div>
                <div style="font-size: 1.1rem; color: #475569; line-height: 1.6;">
                    {reasons_html}
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background-color: #fffbeb; border-left: 6px solid #f59e0b; padding: 20px; border-radius: 8px; margin-bottom: 30px;">
                <div style="font-size: 1.8rem; font-weight: 700; color: #92400e; margin-bottom: 10px;">
                    ⚠ 本日は60分以上の深思考枠がありません
                </div>
                <div style="font-size: 1rem; color: #92400e;">
                    無理にタスクを詰め込まず、軽作業や情報収集、あるいは戦略的な余白（休憩）としてお過ごしください。
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("### 🔋 リアルタイム コンディション")
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
        col_m1, col_m2 = st.columns([1, 1])
        with col_m1:
            phase_color = "#ef4444" if "下降" in current_phase else "#10b981" if "上昇" in current_phase else "#64748b"
            st.markdown(f"""
            <div class="kpi-card" style="border-top: 5px solid #3b82f6; height: 100%;">
                <div class="kpi-title">現在の集中波フェーズ</div>
                <div class="kpi-value-wave" style="color: {phase_color};">{current_phase}</div>
                <div class="kpi-sub" style="color:#64748b; font-weight:normal;">次の集中ピーク予想: 約 <strong>{next_peak_in} 分後</strong></div>
            </div>
            """, unsafe_allow_html=True)
            
        with col_m2:
            prob_color = "#10b981" if focus_prob > 0.6 else "#f59e0b" if focus_prob > 0.4 else "#ef4444"
            st.markdown(f"""
            <div class="kpi-card" style="border-top: 5px solid #8b5cf6; height: 100%;">
                <div class="kpi-title">{PREDICT_AHEAD_MINS}分後の 高集中波 再突入確率</div>
                <div class="kpi-value-main" style="color: {prob_color};">{focus_prob * 100:.1f} <span class="kpi-unit">%</span></div>
                <div class="kpi-sub" style="color:#64748b; font-weight:normal;">上位30%のゾーンに到達する確率</div>
            </div>
            """, unsafe_allow_html=True)

    # --- TAB 2: Weekly Report ---
    with tab_weekly:
        st.markdown("## 今週のパフォーマンスとコンディション振り返り")
        
        st.markdown("#### 📅 日別コンディション・サマリー (疲労と回復)")
        df_daily_cond = summarize_daily_condition(df_1min)
        if not df_daily_cond.empty: st.dataframe(df_daily_cond, use_container_width=True)

        st.markdown("#### 🌊 今週の集中波形 (モメンタルグラフ)")
        st.caption("※ 青い線が平滑化された集中の「波」を表し、赤い点がAIが検出した「波のピーク」です。グレーの点線より上の青い面が「高集中ゾーン」です。波の周期性が確認できます。")
        
        df_this_week = df_feat[(df_feat['date'] > (current_time.date() - pd.Timedelta(days=7))) & (df_feat['date'] <= current_time.date())]
        week_dates = df_this_week['date'].unique()
        week_dates = [d for d in week_dates if d.weekday() in selected_dow_indices]
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
        
        df_feat_spec = df_feat[df_feat.index.dayofweek.isin(selected_dow_indices)].copy()
        df_feat_spec = df_feat_spec[(df_feat_spec.index.hour >= time_range[0]) & (df_feat_spec.index.hour <= time_range[1])]
        
        df_1min_spec = df_1min[df_1min.index.dayofweek.isin(selected_dow_indices)].copy()
        df_1min_spec = df_1min_spec[(df_1min_spec.index.hour >= time_range[0]) & (df_1min_spec.index.hour <= time_range[1])]

        c_spec1, c_spec2, c_spec3 = st.columns(3)
        c_spec1.metric("⏱ 平均集中波 周期", f"{int(metrics['avg_wave_period'])} 分", "波が訪れる間隔")
        c_spec3.metric("📈 波の平均振幅", f"{metrics['avg_wave_amplitude']:.1f} pt", "集中の深さの指標")

        st.markdown("---")
        st.markdown("### 📊 曜日・時間帯別のコンディション特性")
        st.write("設定した曜日・時間帯における「集中」「疲労」「低覚醒」の傾向を可視化しています。")

        st.markdown("#### 🕒 時間帯別 平均ステータス")
        col_g1, col_g2, col_g3 = st.columns(3)
        
        if not df_feat_spec.empty:
            hour_focus = df_feat_spec.groupby(df_feat_spec.index.hour)['is_high_focus_wave'].mean() * 100
            with col_g1:
                fig1 = px.bar(x=[f"{h}:00" for h in hour_focus.index], y=hour_focus.values, title="高集中波 発生確率 (%)", labels={'x': '時間帯', 'y': '確率 (%)'})
                fig1.update_traces(marker_color='#3b82f6')
                st.plotly_chart(fig1, use_container_width=True)
                
        if not df_1min_spec.empty:
            hour_fatigue = df_1min_spec.groupby(df_1min_spec.index.hour)['fatigue_smooth'].mean()
            with col_g2:
                fig2 = px.bar(x=[f"{h}:00" for h in hour_fatigue.index], y=hour_fatigue.values, title="平均疲労スコア", labels={'x': '時間帯', 'y': 'スコア'})
                fig2.update_traces(marker_color='#ef4444')
                f_min, f_max = hour_fatigue.min(), hour_fatigue.max()
                if pd.notna(f_min) and pd.notna(f_max):
                    fig2.update_yaxes(range=[math.floor(f_min) - 2, math.ceil(f_max) + 2])
                st.plotly_chart(fig2, use_container_width=True)
                
            hour_arousal = df_1min_spec.groupby(df_1min_spec.index.hour)['low_arousal'].mean()
            with col_g3:
                fig3 = px.bar(x=[f"{h}:00" for h in hour_arousal.index], y=hour_arousal.values, title="平均低覚醒スコア", labels={'x': '時間帯', 'y': 'スコア'})
                fig3.update_traces(marker_color='#8b5cf6')
                st.plotly_chart(fig3, use_container_width=True)

        st.markdown("#### 📍 曜日×時間帯 ヒートマップ")
        
        def plot_heatmap(df, val_col, title, colorscale, is_prob=False):
            if df.empty or val_col not in df.columns: return None
            df_hm = df.copy()
            df_hm['hour'] = df_hm.index.hour
            df_hm['dow'] = df_hm.index.dayofweek
            pivot = df_hm.pivot_table(values=val_col, index='hour', columns='dow', aggfunc='mean')
            
            if is_prob:
                pivot = pivot * 100
                
            valid_dows = [d for d in selected_dow_indices if d in pivot.columns]
            valid_hours = list(range(time_range[0], time_range[1]+1))
            
            if not valid_dows: return None
            
            heatmap_data = np.full((len(valid_hours), len(valid_dows)), np.nan)
            for i, h in enumerate(valid_hours):
                for j, d in enumerate(valid_dows):
                    if h in pivot.index and d in pivot.columns:
                        heatmap_data[i, j] = pivot.loc[h, d]
                        
            x_labels = [dow_options[d] for d in valid_dows]
            y_labels = [f"{h}:00" for h in valid_hours]
            
            fig = go.Figure(data=go.Heatmap(z=heatmap_data, x=x_labels, y=y_labels, colorscale=colorscale, hoverongaps=False))
            fig.update_layout(title=title, yaxis_autorange='reversed', height=350, margin=dict(l=20, r=20, t=40, b=20))
            return fig
            
        col_hm1, col_hm2, col_hm3 = st.columns(3)
        with col_hm1:
            fig_hm1 = plot_heatmap(df_feat_spec, 'is_high_focus_wave', '高集中 確率 (%)', 'Blues', is_prob=True)
            if fig_hm1: st.plotly_chart(fig_hm1, use_container_width=True)
        with col_hm2:
            fig_hm2 = plot_heatmap(df_1min_spec, 'fatigue_smooth', '疲労スコア', 'Reds')
            if fig_hm2: st.plotly_chart(fig_hm2, use_container_width=True)
        with col_hm3:
            fig_hm3 = plot_heatmap(df_1min_spec, 'low_arousal', '低覚醒スコア', 'Purples')
            if fig_hm3: st.plotly_chart(fig_hm3, use_container_width=True)