
SELECT
    DATE_TRUNC('month', created_at)                                         AS month,
    COUNT(*)                                                                AS total_customers,
    SUM(CASE WHEN churn THEN 1 ELSE 0 END)                                 AS churned,
    ROUND(100.0 * SUM(CASE WHEN churn THEN 1 ELSE 0 END) / COUNT(*), 2)   AS churn_rate_pct
FROM customers
GROUP BY DATE_TRUNC('month', created_at)
ORDER BY month;

SELECT
    CASE
        WHEN tenure BETWEEN 0  AND 12  THEN '0-12 months'
        WHEN tenure BETWEEN 13 AND 24  THEN '13-24 months'
        WHEN tenure BETWEEN 25 AND 48  THEN '25-48 months'
        WHEN tenure BETWEEN 49 AND 72  THEN '49-72 months'
        ELSE '72+ months'
    END                                                                     AS tenure_bucket,
    COUNT(*)                                                                AS total_customers,
    SUM(CASE WHEN churn THEN 1 ELSE 0 END)                                 AS churned,
    ROUND(100.0 * SUM(CASE WHEN churn THEN 1 ELSE 0 END) / COUNT(*), 2)   AS churn_rate_pct,
    ROUND(AVG(monthly_charges), 2)                                          AS avg_monthly_charges
FROM customers
GROUP BY tenure_bucket
ORDER BY MIN(tenure);

WITH monthly_stats AS (
    SELECT
        DATE_TRUNC('month', created_at)                                             AS month,
        COUNT(*)                                                                    AS total,
        SUM(CASE WHEN churn THEN 1 ELSE 0 END)                                     AS churned
    FROM customers
    GROUP BY DATE_TRUNC('month', created_at)
)
SELECT
    month,
    total,
    churned,
    ROUND(100.0 * churned / total, 2)                                               AS monthly_churn_pct,
    ROUND(100.0 * SUM(churned) OVER (ORDER BY month ROWS BETWEEN 2 PRECEDING AND CURRENT ROW)
        / NULLIF(SUM(total) OVER (ORDER BY month ROWS BETWEEN 2 PRECEDING AND CURRENT ROW), 0), 2) AS rolling_3m_churn_pct
FROM monthly_stats
ORDER BY month;

WITH churn_scores AS (
    SELECT
        c.customer_id,
        c.contract,
        c.tenure,
        c.monthly_charges,
        c.internet_service,
        c.payment_method,
        cp.churn_probability,
        NTILE(10) OVER (ORDER BY cp.churn_probability DESC) AS risk_decile
    FROM customers c
    JOIN churn_predictions cp ON c.customer_id = cp.customer_id
)
SELECT *
FROM churn_scores
WHERE risk_decile = 1
ORDER BY churn_probability DESC;
