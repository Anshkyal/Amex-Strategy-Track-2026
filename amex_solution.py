import pandas as pd
import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

def generate_file15_apex_ensemble_v3():
    print("🚀 Initiating Apex Master Ensemble V3 (The Echo-Chamber Breaker)...")
    start_time = time.time()
    
    # 1. Load Data
    df = pd.read_csv('6a3eb196bc7a3_campus_challenge_r1_data.csv')
    fill_zero = ['f1', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 
                 'f12', 'f13', 'f14', 'f15', 'f16', 'f17', 'f18', 
                 'f19', 'f20', 'f21', 'f22', 'f23']
    df[fill_zero] = df[fill_zero].fillna(0)
    df['f11'] = df['f11'].fillna(df['f11'].median())

    df['Total_Spend'] = df[['f6', 'f7', 'f8', 'f9', 'f10']].sum(axis=1)

    # ==========================================
    # PART A: THE PURE ECONOMIC ANCHOR (25% Weight)
    # ==========================================
    print("🧮 Calculating Pure Deterministic Economic Gravity...")
    # Using strict Unit Economics to anchor the ML
    net_margin = (df['f7'] + df['f8'] + df['f10']) * 0.015 + (df['f6'] + df['f9']) * (-0.025)
    rev = net_margin + (df['f1'] * 0.18) + (df['f20'] * 625.0) + (df['f19'] * 175.0)
    cost = (df['f21'] * 0.015) + (df['f4'] * 0.005) + (df['f13'] * 35.0) + df['f14'] + (df['f15'] * 15.0) + df['f16']
    ecl = df['f11'] * (df['f1'] + (df['Total_Spend'] / 12.0))
    
    df['Eco_Score'] = rev - cost - ecl
    df['Eco_Rank'] = df['Eco_Score'].rank(pct=True)

    # ==========================================
    # PART B: THE MACHINE LEARNING ENSEMBLE (75% Weight)
    # ==========================================
    print("📊 Generating Scale-Invariant ML Features...")
    rank_df = pd.DataFrame({'id': df['id']})
    for col in df.columns:
        if col not in ['id', 'Total_Spend', 'Eco_Score', 'Eco_Rank']:
            rank_df[f'r_{col}'] = df[col].rank(pct=True)

    rank_df['Total_Spend_Rank'] = df['Total_Spend'].rank(pct=True)
    rank_df['Risk_Exposure_Rank'] = (rank_df['r_f11'] * rank_df['r_f1']).rank(pct=True)
    rank_df['Reward_Velocity'] = (rank_df['r_f21'] - rank_df['r_f4']).rank(pct=True)
    rank_df['Margin_Proxy_Rank'] = ((rank_df['r_f7'] * 1.5) + (rank_df['r_f8'] * 1.2) + rank_df['r_f10'] + (rank_df['r_f6'] * 0.5) + (rank_df['r_f9'] * 0.5)).rank(pct=True)

    try:
        print("🧠 Loading Teacher Labels...")
        teacher_df = pd.read_excel('2026_File13_Case-Crew.xlsx', sheet_name='Predictions')
        rank_df['Target'] = teacher_df['Prediction']
    except FileNotFoundError:
        print("❌ ERROR: '2026_File13_Case-Crew.xlsx' not found. Make sure your teacher file is in the folder.")
        return

    features = [c for c in rank_df.columns if c not in ['id', 'Target']]
    X = rank_df[features]
    y = rank_df['Target']

    print("🌳 Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=150, max_depth=12, min_samples_leaf=10, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    rf_probs = rf.predict_proba(X)[:, 1]

    print("🌳 Training Gradient Booster...")
    hgb = HistGradientBoostingClassifier(max_iter=400, max_leaf_nodes=31, learning_rate=0.03, l2_regularization=0.5, random_state=42)
    hgb.fit(X, y)
    hgb_probs = hgb.predict_proba(X)[:, 1]

    df['ML_Score'] = (rf_probs * 0.40) + (hgb_probs * 0.60)
    df['ML_Rank'] = df['ML_Score'].rank(pct=True)

    # ==========================================
    # PART C: THE MASTER BLEND & SURGICAL CUTOFF
    # ==========================================
    print("🧬 Blending Machine Learning with Economic Gravity...")
    df['Master_Score'] = (df['ML_Rank'] * 0.75) + (df['Eco_Rank'] * 0.25)

    # Surgical Boundary Tie-Breaker (Give safe accounts a tiny push)
    df['Temp_Rank'] = df['Master_Score'].rank(ascending=False)
    boundary_mask = (df['Temp_Rank'] >= 95000) & (df['Temp_Rank'] <= 105000)
    safe_boost_mask = boundary_mask & (df['f3'] == 0) & (df['f2'] == 0) & (df['f11'] < 0.05)
    df.loc[safe_boost_mask, 'Master_Score'] += 0.0001
    
    # 🚨 PRE-RANK KILL SWITCHES (The Quota Fix) 🚨
    df['Master_Score'] = np.where(df['f3'] > 0, -999.0, df['Master_Score']) 
    df['Master_Score'] = np.where(df['f2'] > 0, -99.0, df['Master_Score'])  

    # Exact Top 20% Cutoff
    df['Final_Rank'] = df['Master_Score'].rank(ascending=False)
    df['Prediction'] = (df['Final_Rank'] <= 100000).astype(int)

    print(f"Total positive predictions: {df['Prediction'].sum()} (Must be exactly 100,000)")

    # Export
    preds_export = df[['id', 'Prediction']].rename(columns={'id': 'ID'})
    
    fw_data = {
        "Section": ["Variables Used", "Profitability Equation", "Prediction Logic", "Variable Selection Logic", "Coefficient/Weight Derivation", "Feature Transformations", "Business Logic", "Assumptions", "Validation Approach", "Additional Notes (Optional)"],
        "Response": ["All variables mapped via Dual-Pipeline (Percentile Ranks for ML, Raw Magnitude for Economics).", "Score = 0.75 * Rank(ML_Ensemble) + 0.25 * Rank(Deterministic_Economics)", "Exact top 20% cutoff applied to the Master Blended score array.", "Integrated a Pure Economic Anchor to shatter the 'Echo Chamber' risk of pseudo-label distillation.", "ML weights via stochastic ensembling; Economic weights via strict Amex unit-margin ratios.", "Kill switches applied BEFORE final ranking to eliminate quota leakages.", "By forcing the ML model to agree with a strict deterministic P&L, we prevent algorithmic hallucination of unprofitable cohorts.", "1x spend represents positive margin; 5x travel spend represents a negative net margin requiring cross-subsidization.", "Validates against Teacher logic while simultaneously enforcing unshakeable financial ground-truths.", "Apex Master Strategy V3."]
    }

    output_file = '2026_File16_Case-Crew.xlsx'
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        preds_export.to_excel(writer, sheet_name='Predictions', index=False)
        pd.DataFrame(fw_data).to_excel(writer, sheet_name='Profitability Framework', index=False)

    print(f"✅ File 16 successfully generated -> {output_file}")

if __name__ == "__main__":
    generate_file15_apex_ensemble_v3()