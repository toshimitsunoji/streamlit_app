# ... existing code ...
    base_jp = col_name
    remainder = ""
    for k, v in mapping.items():
        if col_name.startswith(k):
            base_jp = v
            remainder = col_name[len(k):]
            break
            
    if remainder == "": return base_jp
    elif remainder == "_ssm_true_state": return f"ノイズ除去後の真の「{base_jp}」"
    elif remainder == "_ssm_diff1": return f"真の「{base_jp}」の変動幅"
    elif remainder == "_roll_mean": return f"最近の「{base_jp}」の平均的な高さ"
    elif remainder == "_roll_slope": return f"最近の「{base_jp}」の急な変化(トレンド)"
# ... existing code ...
    base_jp = feat
    remainder = ""
    for k, v in mapping.items():
        if feat.startswith(k):
            base_jp = v
            remainder = feat[len(k):]
            break
            
    if "_is_missing" in feat: return f"「{base_jp}」が未計測であること"
    elif "_ssm_true_state" in feat: return f"ノイズ除去後の真の「{base_jp}」"
    elif feat in ["has_schedule", "is_meeting"]: return f"「{base_jp}」が入っていること" if val > 0 else f"「{base_jp}」が入っていないこと"
# ... existing code ...
        for col in numeric_cols:
            df_features[f'{col}_is_missing'] = df_resampled[col].isna().astype(int)
            r = df_features[col].rolling(win, min_periods=win)
            df_features[f'{col}_roll_mean'] = r.mean()
            df_features[f'{col}_roll_slope'] = r.apply(rolling_slope, raw=True)
            df_features[f'{col}_diff1'] = df_features[col] - df_features[col].shift(1)

        # --- 状態空間モデル (SSM) の組み込み ---
        base_col_map = {
            '集中判定': 'CVRR_SCORE_NEW',
            '疲労判定': 'RMSSD_SCORE_NEW',
            '強い疲労判定': 'RMSSD_SCORE_NEW',
            '眠気判定': 'NEMUKE_SCORE_NEW',
            '強い眠気判定': 'NEMUKE_SCORE_NEW'
        }
        base_col = base_col_map.get(target_col, None)

        if base_col and base_col in df_features.columns:
            st.info(f"💡 状態空間モデル(SSM)を適用し、「{jp_feat_name(base_col)}」の真の潜在状態を推定しています...")
            try:
                import statsmodels.api as sm
                exog_cols = [c for c in ['休憩判定', '短時間歩行', 'is_meeting', 'schedule_density_2h'] if c in df_features.columns]
                
                # 欠損値を補間（SSMの安定化のため）
                endog = df_features[base_col].interpolate(method='linear').bfill().ffill()
                
                if exog_cols:
                    exog = df_features[exog_cols].fillna(0).astype(float)
                    # Local Levelモデル (観測ノイズと状態ノイズの分離)
                    mod = sm.tsa.UnobservedComponents(endog, level='local level', exog=exog)
                else:
                    mod = sm.tsa.UnobservedComponents(endog, level='local level')
                    
                res = mod.fit(disp=False)
                
                # 平滑化されたレベル状態（潜在的な真のベースライン）
                smoothed_level = res.smoothed_state[0]
                
                # 外生変数の効果を足し合わせて最終的な「真の状態」を算出
                if exog_cols:
                    exog_params = {col: res.params.get(col, 0) for col in exog_cols}
                    exog_effect = sum(df_features[col].fillna(0) * exog_params[col] for col in exog_cols)
                    true_state = smoothed_level + exog_effect
                else:
                    true_state = smoothed_level
                    
                df_features[f'{base_col}_ssm_true_state'] = true_state
                df_features[f'{base_col}_ssm_diff1'] = df_features[f'{base_col}_ssm_true_state'].diff()
                
            except Exception as e:
                st.warning(f"状態空間モデルの計算をスキップしました: {e}")

        df_features['date'] = df_features.index.date
        if '1分間歩数' in df_features.columns:
# ... existing code ...
                # ホバー・表示用の実際の推移線（黒色）
                fig_daily.add_trace(go.Scatter(
                    x=df_day.index, 
                    y=df_day['CVRR_SCORE_NEW'],
                    mode='lines',
                    line=dict(color='#333333', width=2),
                    name='CVRR SCORE',
                    hovertemplate="時刻: %{x|%H:%M}<br>スコア: %{y:.1f}<extra></extra>"
                ))
                
                # --- SSMの真の推移を重ねて表示 (リサンプリング後のデータ) ---
                df_imp_day = df_imp[(df_imp.index.date.astype(str) == selected_day) & 
                                    (df_imp.index.hour >= time_range[0]) & 
                                    (df_imp.index.hour <= time_range[1])]
                
                if 'CVRR_SCORE_NEW_ssm_true_state' in df_imp_day.columns and not df_imp_day.empty:
                    fig_daily.add_trace(go.Scatter(
                        x=df_imp_day.index, 
                        y=df_imp_day['CVRR_SCORE_NEW_ssm_true_state'],
                        mode='lines',
                        line=dict(color='rgba(255, 50, 50, 0.9)', width=3, dash='solid'),
                        name='真の集中推移 (SSM平滑化)',
                        hovertemplate="時刻: %{x|%H:%M}<br>真のスコア: %{y:.1f}<extra></extra>"
                    ))
                
                fig_daily.update_layout(
                    title=f"{selected_day} の集中と緩和の推移 ({time_range[0]}時〜{time_range[1]}時)",
# ... existing code ...
                    st.info(f"**【{selected_day} のデイリーインサイト】**\n\n"
                            f"- この日の設定時間帯（{time_range[0]}時〜{time_range[1]}時）における集中（CVRR SCORE）のピークは **{max_idx.strftime('%H:%M')}頃** （スコア: {max_val:.1f}）でした。\n"
                            f"- 平均スコアは **{avg_val:.1f}** となっています。\n"
                            f"- グラフにおいて基準値(50)より上側の**青い面**が「集中」している状態、下側の**オレンジの面**が「緩和（リラックス）」している状態を示しています。\n"
                            f"- **赤色の太線**は、状態空間モデル(SSM)によって観測ノイズを除去し、予定や行動の影響を加味して推定された**「真の集中推移」**です。")
                else:
                    st.write("この日の有効なスコアデータがありません。")
# ... existing code ...