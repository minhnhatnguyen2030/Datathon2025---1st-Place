# Datathon 2025 – Supply Chain Optimisation

## 🏆 Competition
Datathon 2025 focused on reducing supply chain costs and improving logistics efficiency through data-driven methods.  
Our team **Clesmor** (Lyra, Carmen, Calvin, Shane, Thomas) won **1st Place**.

---

## 📊 Problem Statement
- Logistics network built on raw GPS points, lacking structured design.  
- High shipping cost ($14.7M), long delivery deviations (5h 10m), high CO₂ emissions (0.812M kg).  

---

## 🔍 Approach
1. **Exploratory Data Analysis**  
   - Identified weak correlation of single features with cost/delay (max corr ≈ 0.01).  
   - Segmented by risk factors (traffic, weather, supplier, port).  
   - Confirmed marginal effects small → interactions matter.

2. **Machine Learning & Forecasting**  
   - Cost model (Gradient Boosting): R² < 0.  
   - Delivery deviation model: R² ≈ 0.  
   - Time series forecasting (ARIMA, ETS with BU/OLS/WLS reconciliation): poor predictive power.

3. **Optimisation (Gurobi)**  
   - Estimated stable parameters: cost uplifts, delay probabilities, emission factors.  
   - Multi-objective optimisation: minimise cost, delay, emissions.  
   - Constraints: capacity, reliability, emission caps.  

4. **Scenario Trade-offs**  
   - Cost-first: cheapest but higher risk.  
   - Low-carbon-first: +0.8% cost, –0.24% CO₂.  
   - On-time-first: +0.17% cost, –0.82% delays (most balanced).  

---

## 📈 Business Impact
- 54% lower cost  
- 35% lower average delays  
- 15% lower CO₂ emissions  
- Recommendation: **On-time-first strategy** for best balance of customer experience, efficiency, and sustainability.

---

## ⚙️ Tech Stack
- Python (pandas, scikit-learn, statsmodels)  
- Time Series (ARIMA, ETS + reconciliation)  
- Optimisation: **Gurobi**  
- Visualisation: matplotlib, seaborn  

---

## 📌 Outcome
🏆 **1st Place** in Datathon 2025
