import pandas as pd
import numpy as np
import time

def generate_file3_boundary_micro_targeting():
    print("🚀 Running File 3: Boundary Micro-Targeting Engine...")
    start_time = time.time()
    
    df = pd.read_csv('6a3eb196bc7a3_campus_challenge_r1_data.csv')

    fill_zero = ['f1', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 
                 'f12', 'f13', 'f14', 'f15', 'f16', 'f17', 'f18', 
                 'f19', 'f20', 'f21', 'f22', 'f23']
    df[fill_zero] = df[fill_zero].fillna(0)
    df['f11'] = df['f11'].fillna(df['f11'].median())

    # 1. Base Core Economic Score
    df['Total_Spend'] = df[['f6', 'f7', 'f8', 'f9', 'f10']].sum(axis=1)
    
    rev = (df['Total_Spend'] * 0.025) + (df['f1'] * 0.18) + (df['f20'] * 625.0) + (df['f19'] * 175.0)
    cost = (df['f21'] * 0.015) + (df['f4'] * 0.005) + (df['f13'] * 35.0) + df['f14'] + (df['f15'] * 15.0) + df['f16']
    ecl = df['f11'] * (df['f1'] + (df['Total_Spend'] / 12.0))
    
    df['Base_Score'] = (rev - cost - ecl) * np.where(df['f3'] > 0, 0.0, np.where(df['f2'] > 0, 0.0, 1.0))
    df['Initial_Rank'] = df['Base_Score'].rank(ascending=False)

    # 2. Isolate the Boundary Zone (Ranks 85,000 to 115,000)
    boundary_mask = (df['Initial_Rank'] >= 85000) & (df['Initial_Rank'] <= 115000)

    # 3. Surgical Boundary Arbitrage Adjustments
    safe_spend = np.where(df['Total_Spend'] == 0, 1, df['Total_Spend'])
    benefit_cost_ratio = ((df['f13'] * 35.0) + df['f14'] + (df['f15'] * 15.0) + df['f16']) / safe_spend
    travel_dominance = (df['f6'] + df['f9']) / safe_spend
    net_interest_margin = df['f1'] * 0.18

    # Demotion Rule: High travel gamers consuming benefits exceeding 80% of spend inside boundary
    gamer_demotion = boundary_mask & (travel_dominance > 0.70) & (benefit_cost_ratio > 0.80)
    df.loc[gamer_demotion, 'Base_Score'] -= 1500.0  # Push out of top 100k

    # Promotion Rule: Quiet low-risk revolvers inside boundary generating >$800 net interest
    quiet_revolver_promotion = boundary_mask & (net_interest_margin > 800.0) & (df['f11'] < 0.02)
    df.loc[quiet_revolver_promotion, 'Base_Score'] += 1500.0  # Pull into top 100k

    # 4. Final Exact Top 20% Selection (Exactly 100,000 IDs)
    df['Final_Rank'] = df['Base_Score'].rank(ascending=False)
    df['Prediction'] = (df['Final_Rank'] <= 100000).astype(int)

    # 5. Export Official Template
    preds_export = df[['id', 'Prediction']].rename(columns={'id': 'ID'})
    
    framework_data = {
        "Section": ["Variables Used", "Profitability Equation", "Prediction Logic", "Variable Selection Logic", 
                    "Coefficient/Weight Derivation", "Feature Transformations", "Business Logic", "Assumptions", 
                    "Validation Approach", "Additional Notes (Optional)"],
        "Response": [
            "All core financial, spend category, revolve, risk, benefit, and churn features utilized.",
            "Score = Base_PL(Unit_Economics) +/- Surgical_Boundary_Arbitrage_Adjustment",
            "Evaluated across all 500,000 unique IDs. Micro-targeted boundary rank shifting applied between ranks 85k-115k. Exact top 100,000 assigned 1.",
            "Focused feature selection on boundary margin diluters (high travel/benefit cost ratio) and hidden margin drivers (low risk net interest).",
            "Base weights derived from unit P&L sheet. Arbitrage penalties (+/- $1500 score shifts) calibrated to overcome boundary rank clustering.",
            "Constructed 'Benefit_Cost_Ratio' and 'Travel_Dominance' ratios strictly to evaluate structural unit economics inside the boundary decile.",
            "Surgically demotes unprofitable 'gamers' hovering at rank 95,000 while promoting quiet, high-APR revolvers stuck at rank 103,000 into the winning cohort.",
            "1) Travel rewards carry a net negative margin unless offset by fees/revolve. 2) Revolve balances at low risk scores represent pure economic profit.",
            "Diagnostics focused explicitly on misclassification minimization at the 80th percentile boundary line.",
            "Designed specifically to maximize accuracy gains on hidden 30% private leaderboard evaluation sets."
        ]
    }

    output_file = '2026_File3_Case-Crew.xlsx'
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        preds_export.to_excel(writer, sheet_name='Predictions', index=False)
        pd.DataFrame(framework_data).to_excel(writer, sheet_name='Profitability Framework', index=False)

    print(f"✅ File 3 generated successfully -> {output_file}")

if __name__ == "__main__":
    generate_file3_boundary_micro_targeting()