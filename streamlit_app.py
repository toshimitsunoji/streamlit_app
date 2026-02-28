# -*- coding: utf-8 -*-
"""
ウェアラブル + Outlookスケジュール 集中・疲労予測アプリ (V2: アクション特化型)
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.tree import DecisionTreeRegressor, _tree, plot_tree
import google.generativeai as genai
import shap
import warnings
import plotly.graph_objects as go
from pathlib import Path
import matplotlib as mpl
import matplotlib.font_manager as fm
import datetime

# --- Streamlit ページ設定 ---
st.set_page_config(page_title="Focus Battery | 集中予測", layout="wide", initial_sidebar_state="expanded")

# 日本語フォントの設定
font_path = Path(__file__).parent / "assets" / "fonts" / "NotoSansCJKjp-Regular.otf"
if font_path.exists():
    fm.fontManager.addfont(str(font_path))
    prop = fm.FontProperties(fname=str(font_path))
    mpl.rcParams["font.family"] = prop.get_name()

mpl.rcParams["axes.unicode_minus"] = False
warnings.filterwarnings('ignore')

# --- カスタムCSS (UIの洗練) ---
st.markdown("""
<style>
    .metric-container { background-color: #f8f9fa; border-radius: 10px; padding: 20px; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.05); margin-bottom: 20px; }
    .metric-title { font-size: 1.2rem; color: #6c757d; margin-bottom: 5px; font-weight: 600; }
    .metric-value { font-size: 3.5rem; color: #2b2b2b; font-weight: 800; line-height: 1.2; }
    .metric-sub { font-size: 1rem; color: #28a745; font-weight: bold; }
    .metric-sub.negative { color: #dc3545; }
    .window-box { background-color: #e3f2fd; border-left: 5px solid #1976d2; padding: 20px; border-radius: 5px; margin-bottom: 20px; }
    .window-time { font-size: 2rem; color: #1976d2; font-weight: 800; }
</style>
""", unsafe_allow_html=True)

# --- サイドバー (設定・データ入力) ---
with st.sidebar:
    st.header("⚙️ データ入力")
    file_ts = st.file_uploader("1. 生体データ (CSV)", type=['csv'])
    file_sched = st.file_uploader("2. 予定表データ (CSV)", type=['csv'])
    
    with st.expander("🛠 詳細設定 (通常は変更不要)"):
        api_key = st.text_input("Gemini APIキー", type="password")
        RESAMPLE_FREQ = st.selectbox("分析単位", ['10T', '30T', '1H'], index=1)
        PREDICT_AHEAD = st.selectbox("予測先", ['10T', '30T', '1H'], index=1)
        TARGET_DATETIME_STR = st.text_input("予測基準日時 (空欄で最新)")
        target_col = '集中判定' # V2ではアクションに直結しやすい「集中」を主軸に固定
        
    st.markdown("---")
    st.button("🚀 今日のコンパスを更新", type="primary", use_container_width=True, key="run_btn")

freq_td = pd.Timedelta(RESAMPLE_FREQ)
ahead_steps = max(1, int(pd.Timedelta(PREDICT_AHEAD) / freq_td))
TARGET_DATETIME = TARGET_DATETIME_STR if TARGET_DATETIME_STR.strip() != "" else None

# --- 特徴量名日本語化ヘルパー (V1から継承) ---
def jp_feat_name(col_name: str) -> str:
    mapping = {'CVRR_SCORE_NEW': '集中スコア', '1分間歩数': '歩数', 'is_meeting': '会議', 'schedule_density_2h': '予定密度', '休憩判定': '休憩', '短時間歩行': '短時間歩行'}
    for k, v in mapping.items():
        if k in col_name: return col_name.replace(k, v)
    return col_name

# --- 分析エンジン (V1の強力なロジックを継承・隠蔽) ---
@st.cache_data(show_spinner=False)
def load_and_preprocess(file_ts_bytes, file_sched_bytes):
    # 生体データの読み込み
    import io
    df_ts = pd.read_csv(io.BytesIO(file_ts_bytes), skiprows=2)
    df_ts['timestamp_clean'] = df_ts['timestamp'].astype(str).str.split(' GMT').str[0]
    df_ts['datetime'] = pd.to_datetime(df_ts['timestamp_clean'], errors='coerce')
    df_ts = df_ts.dropna(subset=['datetime']).set_index('datetime').sort_index()

    # 予定データの読み込み
    df_sched = None
    if file_sched_bytes:
        df_sched = pd.read_csv(io.BytesIO(file_sched_bytes))
        df_sched = df_sched[df_sched['終日イベント'].astype(str).str.upper() != 'TRUE']
        df_sched['start_dt'] = pd.to_datetime(df_sched['開始日'].astype(str) + ' ' + df_sched['開始時刻'].astype(str), errors='coerce')
        df_sched['end_dt']   = pd.to_datetime(df_sched['終了日'].astype(str) + ' ' + df_sched['終了時刻'].astype(str), errors='coerce')
        df_sched = df_sched.dropna(subset=['start_dt', 'end_dt']).sort_values('start_dt')

    return df_ts, df_sched

def run_ml_pipeline(df_ts, df_sched):
    # --- 前処理 ---
    num_cols = df_ts.select_dtypes(include=[np.number]).columns
    df_resampled = df_ts[num_cols].resample(RESAMPLE_FREQ).mean()
    if '1分間歩数' in df_ts.columns:
        df_resampled['1分間歩数'] = df_ts['1分間歩数'].resample(RESAMPLE_FREQ).sum()

    if df_sched is not None:
        df_resampled['has_schedule'] = 0
        df_resampled['is_meeting'] = 0
        meeting_keywords = ['会議', '打合せ', 'MTG', '面談']
        for _, row in df_sched.iterrows():
            mask = (df_resampled.index < row['end_dt']) & ((df_resampled.index + freq_td) > row['start_dt'])
            df_resampled.loc[mask, 'has_schedule'] = 1
            if any(kw in str(row.get('件名', '')) for kw in meeting_keywords):
                df_resampled.loc[mask, 'is_meeting'] = 1
        win_steps = max(1, int(pd.Timedelta('2H') / freq_td))
        df_resampled['schedule_density_2h'] = df_resampled['has_schedule'].rolling(win_steps, min_periods=1).mean()

    # --- 特徴量生成 ---
    df_features = df_resampled.copy()
    if '集中判定' in df_features.columns:
        focus_mask = (df_features['集中判定'] >= 0.5).astype(int)
        group_id = (focus_mask != focus_mask.shift()).cumsum()
        df_features['現在の集中継続時間_分'] = (focus_mask.groupby(group_id).cumcount() + 1) * (freq_td.total_seconds() / 60) * focus_mask
    
    if '休憩判定' in df_features.columns: df_features['休憩判定_前'] = df_features['休憩判定'].shift(1)
    if '短時間歩行' in df_features.columns: df_features['短時間歩行_前'] = df_features['短時間歩行'].shift(1)

    df_features['target_ahead'] = (df_features[target_col].shift(-ahead_steps) >= 0.5).astype(int)
    
    # --- 学習 ---
    drop_cols = ['target_ahead']
    df_imp = df_features.ffill(limit=2).bfill(limit=2)
    train_df = df_imp.dropna(subset=drop_cols + [target_col])
    
    X = train_df.drop(columns=drop_cols)
    y = train_df['target_ahead']
    
    model = lgb.LGBMClassifier(objective='binary', n_estimators=100, learning_rate=0.05, random_state=42, verbose=-1)
    model.fit(X, y)
    
    return model, df_imp, X.columns, df_ts, df_sched

# === メイン処理 ===
if st.session_state.get('run_btn') or (file_ts is not None):
    if file_ts is None:
        st.info("👈 サイドバーから「1. 生体データ」をアップロードして分析を開始してください。")
        st.stop()
        
    with st.spinner("AIがコンディションを解析中..."):
        df_ts_raw, df_sched_raw = load_and_preprocess(file_ts.getvalue(), file_sched.getvalue() if file_sched else None)
        model, df_imp, feature_cols, df_ts_min, df_sched = run_ml_pipeline(df_ts_raw, df_sched_raw)
        
        # 基準日時の決定
        if TARGET_DATETIME is not None:
            try:
                current_time = pd.to_datetime(TARGET_DATETIME)
                target_data_all = df_imp[df_imp.index <= current_time]
                if not target_data_all.empty:
                    target_data = target_data_all.iloc[-1:]
                    current_time = target_data.index[0]
                else:
                    target_data = df_imp.iloc[-1:]
                    current_time = target_data.index[0]
            except:
                target_data = df_imp.iloc[-1:]
                current_time = target_data.index[0]
        else:
            target_data = df_imp.iloc[-1:]
            current_time = target_data.index[0]
            
        current_proba = model.predict_proba(target_data[feature_cols])[0, 1]
        
        # ==========================================
        # 🔋 Focus Battery ロジック
        # ==========================================
        # 過去の1日あたり平均集中分数を計算
        if '集中判定' in df_ts_min.columns:
            daily_focus = df_ts_min['集中判定'].resample('D').apply(lambda x: (x >= 0.5).sum()) # 1分単位想定
            base_focus_mins = daily_focus.mean() if not daily_focus.empty else 120
        else:
            base_focus_mins = 120
            
        # 今日の消化分
        today_str = current_time.strftime('%Y-%m-%d')
        if '集中判定' in df_ts_min.columns and today_str in df_ts_min.index.strftime('%Y-%m-%d'):
            today_data = df_ts_min[df_ts_min.index.date == current_time.date()]
            consumed_mins = (today_data.loc[:current_time, '集中判定'] >= 0.5).sum()
        else:
            consumed_mins = 0
            
        # コンディション係数（予測確率に基づく）
        context_factor = 0.5 + current_proba # 確率0%なら半分、100%なら1.5倍
        
        remaining_battery = max(0, int((base_focus_mins * context_factor) - consumed_mins))
        battery_delta = int((base_focus_mins * context_factor) - base_focus_mins)
        
        # ==========================================
        # 🕒 Deep Work Window ロジック
        # ==========================================
        window_text = "本日は終了モードです"
        window_desc = "しっかり休んで明日に備えましょう。"
        
        if current_time.hour < 19:
            # 今から20時までの空き時間を探す
            start_search = current_time
            end_search = current_time.replace(hour=20, minute=0, second=0)
            
            # 簡易的に予定表から空きブロックを抽出
            free_blocks = []
            curr_block_start = start_search
            
            if df_sched is not None and not df_sched.empty:
                today_sched = df_sched[(df_sched['start_dt'] >= start_search) & (df_sched['start_dt'] < end_search)].sort_values('start_dt')
                for _, row in today_sched.iterrows():
                    if row['start_dt'] > curr_block_start:
                        duration = (row['start_dt'] - curr_block_start).total_seconds() / 60
                        if duration >= 60: # 60分以上の空きを対象
                            free_blocks.append((curr_block_start, row['start_dt'], duration))
                    curr_block_start = max(curr_block_start, row['end_dt'])
                
                if curr_block_start < end_search:
                    duration = (end_search - curr_block_start).total_seconds() / 60
                    if duration >= 60:
                        free_blocks.append((curr_block_start, end_search, duration))
            else:
                # 予定表がない場合は、直近のキリの良い時間から90分を提案
                next_hour = current_time.replace(minute=0, second=0) + pd.Timedelta('1H')
                if next_hour < end_search:
                    free_blocks.append((next_hour, next_hour + pd.Timedelta('90T'), 90))
                    
            if free_blocks:
                # 最も長い（または直近の）ブロックを選択
                best_block = sorted(free_blocks, key=lambda x: x[2], reverse=True)[0]
                # 最大90分に制限
                w_start = best_block[0]
                w_end = w_start + pd.Timedelta(minutes=min(90, best_block[2]))
                window_text = f"{w_start.strftime('%H:%M')} – {w_end.strftime('%H:%M')}"
                window_desc = "この時間に「企画」「設計」「執筆」など最も重いタスクを配置してください。"

        # ==========================================
        # ⚠️ アラート判定
        # ==========================================
        fatigue_risk = False
        if '疲労判定' in target_data.columns and target_data['疲労判定'].values[0] >= 0.5:
            fatigue_risk = True
        elif 'schedule_density_2h' in target_data.columns and target_data['schedule_density_2h'].values[0] >= 0.6:
            fatigue_risk = True

    # --- UI 描画開始 ---
    st.markdown(f"<p style='text-align: right; color: gray;'>更新日時: {current_time.strftime('%Y/%m/%d %H:%M')}</p>", unsafe_allow_html=True)
    
    tab_today, tab_weekly = st.tabs(["🧭 Today's Compass (今日の行動)", "📊 Weekly Report (振り返り)"])

    with tab_today:
        st.markdown("## 10秒で決める、今日の最適解")
        
        col1, col2 = st.columns([1.2, 1])
        with col1:
            # 1. Focus Battery
            delta_color = "positive" if battery_delta >= 0 else "negative"
            delta_sign = "+" if battery_delta >= 0 else ""
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-title">🔋 本日の高品質集中 残り</div>
                <div class="metric-value">{remaining_battery} <span style="font-size: 1.5rem;">分</span></div>
                <div class="metric-sub {delta_color}">あなたの基準値比 {delta_sign}{battery_delta}分</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            # 2. Deep Work Window
            st.markdown(f"""
            <div class="window-box">
                <div class="metric-title">🕒 今日の勝負枠 (Deep Work Window)</div>
                <div class="window-time">{window_text}</div>
                <div style="color: #555; margin-top: 10px;">👉 {window_desc}</div>
            </div>
            """, unsafe_allow_html=True)
            
        # 3. アラート
        if fatigue_risk:
            st.error("⚠️ **疲労アラート**: 現在疲労が蓄積しているか、予定が過密です。10分程度の完全な休息をとるか、重要な意思決定を後回しにすることを推奨します。")
        else:
            st.success("✨ **コンディション良好**: 現在、集中を阻害する大きなノイズはありません。")

        # --- 4. ワンタップ・シミュレーション (キラー機能) ---
        st.markdown("---")
        st.markdown("### 🔮 アクション・シミュレーター")
        st.write("「今からどう行動を変えるか」で、未来の集中バッテリーがどう回復するかをAIが即座にシミュレーションします。")
        
        sim_col1, sim_col2, sim_col3 = st.columns(3)
        
        def simulate_battery(mod_dict):
            sim_data = target_data[feature_cols].copy()
            for k, v in mod_dict.items():
                if k in sim_data.columns: sim_data[k] = v
            sim_proba = model.predict_proba(sim_data)[0, 1]
            sim_factor = 0.5 + sim_proba
            return max(0, int((base_focus_mins * sim_factor) - consumed_mins))

        with sim_col1:
            if st.button("🚶 今から15分歩く", use_container_width=True):
                new_batt = simulate_battery({'短時間歩行': 1.0, '1分間歩数': 1000})
                gain = new_batt - remaining_battery
                if gain > 0: st.success(f"予測: バッテリーが **+{gain}分** 回復します！")
                else: st.info("予測: 現時点では大きな回復効果は見込めません。")
                
        with sim_col2:
            if st.button("☕ 予定の前に休憩をとる", use_container_width=True):
                new_batt = simulate_battery({'休憩判定': 1.0, 'time_since_prev_event_min': 30})
                gain = new_batt - remaining_battery
                if gain > 0: st.success(f"予測: バッテリーが **+{gain}分** 回復します！")
                else: st.info("予測: 現時点では大きな回復効果は見込めません。")
                
        with sim_col3:
            if st.button("🚫 直近の会議を1つスキップ", use_container_width=True):
                new_batt = simulate_battery({'is_meeting': 0.0, 'schedule_density_2h': max(0, target_data['schedule_density_2h'].values[0] - 0.25)})
                gain = new_batt - remaining_battery
                if gain > 0: st.success(f"予測: バッテリーが **+{gain}分** 節約できます！")
                else: st.info("予測: 現時点では大きな回復効果は見込めません。")

    # ==========================================
    # Tab 2: Weekly Report (従来のダッシュボード機能群)
    # ==========================================
    with tab_weekly:
        st.markdown("## 週末の振り返りと分析 (Weekly Report)")
        st.write("今週のパフォーマンスの推移と、AIが見つけた「あなた専用の集中ルール」を確認します。")
        
        # --- マイルール抽出 (決定木) ---
        st.markdown("#### 🌳 AIが見つけた「あなた専用の集中ルール」")
        action_cols = [c for c in ['休憩判定', '短時間歩行', 'is_meeting', 'schedule_density_2h'] if c in df_imp.columns]
        if len(action_cols) > 0 and len(df_imp) > 10:
            reg_df = df_imp.dropna(subset=action_cols + [target_col])
            X_rule = reg_df[action_cols]
            y_rule = reg_df[target_col]
            
            tree_model = DecisionTreeRegressor(max_depth=2, min_samples_leaf=10, random_state=42)
            tree_model.fit(X_rule, y_rule)
            
            feat_names = [jp_feat_name(c) for c in action_cols]
            is_bool = [X_rule[c].nunique() <= 2 for c in action_cols]
            
            # 簡易ルール抽出（ベストパターンの取得）
            best_leaf_idx = np.argmax(tree_model.tree_.value[:, 0, 0])
            path = tree_model.decision_path(X_rule.iloc[[0]]).toarray()[0] # ダミー
            # ※ 本格的なルールテキスト化はV1と同じ再帰関数が必要ですが、ここでは簡略化して視覚的なツリーを表示します。
            
            fig_tree, ax_tree = plt.subplots(figsize=(8, 4))
            plot_tree(tree_model, feature_names=feat_names, filled=True, rounded=True, ax=ax_tree, fontsize=10, precision=2)
            st.pyplot(fig_tree)
            st.caption("※ 上の箱からスタートし、条件に当てはまる(True)なら左、当てはまらない(False)なら右へ進みます。色が濃いほど集中確率が高い状態です。")
        
        # --- ウィークリーグラフ ---
        st.markdown("#### 📅 今週の集中推移グラフ")
        df_ts_min['date_str'] = df_ts_min.index.date.astype(str)
        week_start = (current_time - pd.to_timedelta(current_time.dayofweek, unit='d')).date()
        week_data = df_ts_min[df_ts_min.index.date >= week_start]
        
        if not week_data.empty and 'CVRR_SCORE_NEW' in week_data.columns:
            # 1日ごとの平均を棒グラフで表示
            daily_avg = week_data.groupby('date_str')['CVRR_SCORE_NEW'].mean()
            fig_week = px.bar(x=daily_avg.index, y=daily_avg.values, labels={'x': '日付', 'y': '平均集中スコア'}, title="日ごとの平均集中スコア")
            fig_week.update_traces(marker_color='#1976d2')
            fig_week.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_week, use_container_width=True)
            
        # --- Gemini 生成AIレポート ---
        if api_key:
            st.markdown("#### 🤖 専属AIコーチからの今週のフィードバック")
            with st.spinner("レポートを生成中..."):
                try:
                    genai.configure(api_key=api_key)
                    model_llm = genai.GenerativeModel('gemini-2.5-flash')
                    prompt = f"""
                    あなたはプロの生産性コーチです。以下のデータに基づき、ユーザーの今週の働き方を振り返り、来週に向けた「改善アクション」を3つ提案してください。
                    ・ユーザーの平均集中バッテリー残量基準: {base_focus_mins}分
                    ・最近の集中スコア平均: {week_data['CVRR_SCORE_NEW'].mean() if not week_data.empty and 'CVRR_SCORE_NEW' in week_data.columns else '不明'}
                    ・現在のコンディション: {'疲労リスクあり' if fatigue_risk else '良好'}
                    
                    出力形式:
                    1. 今週の総評（1〜2行）
                    2. 来週すぐできる改善アクション（箇条書きで3つ、具体的に）
                    """
                    resp = model_llm.generate_content(prompt)
                    st.success(resp.text)
                except Exception as e:
                    st.error(f"Gemini APIエラー: {e}")