-- ============================================
-- Business KPI Queries
-- Author: Mukul (github.com/phantom074)
-- ============================================

-- 1. Overall Churn KPIs
SELECT
    COUNT(*)                                                                AS total_customers,
    SUM(CASE WHEN churn THEN 1 ELSE 0 END)                                 AS total_churned,
    ROUND(100.0 * SUM(CASE WHEN churn THEN 1 ELSE 0 END) / COUNT(*), 2)   AS overall_churn_rate_pct,
    ROUND(AVG(monthly_charges), 2)                                          AS avg_monthly_charge,
    ROUND(SUM(CASE WHEN churn THEN monthly_charges ELSE 0 END), 2)         AS monthly_revenue_lost
FROM customers;

-- 2. Churn by Contract Type
SELECT
    contract,
    COUNT(*)                                                                AS total,
    SUM(CASE WHEN churn THEN 1 ELSE 0 END)                                 AS churned,
    ROUND(100.0 * SUM(CASE WHEN churn THEN 1 ELSE 0 END) / COUNT(*), 2)   AS churn_rate_pct
FROM customers
GROUP BY contract
ORDER BY churn_rate_pct DESC;

-- 3. Churn by Internet Service
SELECT
    internet_service,
    COUNT(*)                                                                AS total,
    SUM(CASE WHEN churn THEN 1 ELSE 0 END)                                 AS churned,
    ROUND(100.0 * SUM(CASE WHEN churn THEN 1 ELSE 0 END) / COUNT(*), 2)   AS churn_rate_pct,
    ROUND(AVG(monthly_charges), 2)                                          AS avg_charges
FROM customers
GROUP BY internet_service
ORDER BY churn_rate_pct DESC;

-- 4. Churn by Payment Method
SELECT
    payment_method,
    COUNT(*)                                                                AS total,
    SUM(CASE WHEN churn THEN 1 ELSE 0 END)                                 AS churned,
    ROUND(100.0 * SUM(CASE WHEN churn THEN 1 ELSE 0 END) / COUNT(*), 2)   AS churn_rate_pct
FROM customers
GROUP BY payment_method
ORDER BY churn_rate_pct DESC;

-- 5. Revenue at Risk (High Probability Churners)
SELECT
    ROUND(SUM(c.monthly_charges), 2)        AS monthly_revenue_at_risk,
    ROUND(SUM(c.monthly_charges) * 12, 2)  AS annual_revenue_at_risk,
    COUNT(*)                                AS high_risk_customers
FROM customers c
JOIN churn_predictions cp ON c.customer_id = cp.customer_id
WHERE cp.churn_probability >= 0.70;

-- 6. Average Tenure of Churned vs Retained
SELECT
    churn,
    ROUND(AVG(tenure), 1)           AS avg_tenure_months,
    ROUND(AVG(monthly_charges), 2)  AS avg_monthly_charges,
    ROUND(AVG(total_charges), 2)    AS avg_total_charges,
    COUNT(*)                        AS count
FROM customers
GROUP BY churn;

-- 7. Senior Citizen Churn Analysis
SELECT
    senior_citizen,
    COUNT(*)                                                                AS total,
    SUM(CASE WHEN churn THEN 1 ELSE 0 END)                                 AS churned,
    ROUND(100.0 * SUM(CASE WHEN churn THEN 1 ELSE 0 END) / COUNT(*), 2)   AS churn_rate_pct
FROM customers
GROUP BY senior_citizen;

-- 8. Top 10 Customers by Revenue at Risk
SELECT
    c.customer_id,
    c.contract,
    c.tenure,
    c.monthly_charges,
    cp.churn_probability,
    cp.top_reason
FROM customers c
JOIN churn_predictions cp ON c.customer_id = cp.customer_id
WHERE cp.churn_predicted = TRUE
ORDER BY c.monthly_charges DESC
LIMIT 10;
