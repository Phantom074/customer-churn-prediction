-- ============================================
-- RFM Segmentation using Window Functions
-- Author: Mukul (github.com/phantom074)
-- R = Recency   (how recently did they transact?)
-- F = Frequency (how often do they transact?)
-- M = Monetary  (how much do they spend?)
-- ============================================

WITH rfm_base AS (
    SELECT
        c.customer_id,
        c.churn,
        c.contract,
        c.monthly_charges,
        MAX(t.transaction_date)                             AS last_transaction,
        COUNT(t.transaction_id)                             AS frequency,
        SUM(t.amount)                                       AS monetary,
        CURRENT_DATE - MAX(t.transaction_date)              AS recency_days
    FROM customers c
    LEFT JOIN transactions t ON c.customer_id = t.customer_id
    GROUP BY c.customer_id, c.churn, c.contract, c.monthly_charges
),
rfm_scores AS (
    SELECT *,
        NTILE(5) OVER (ORDER BY recency_days ASC)           AS r_score,
        NTILE(5) OVER (ORDER BY frequency DESC)             AS f_score,
        NTILE(5) OVER (ORDER BY monetary DESC)              AS m_score
    FROM rfm_base
),
rfm_segments AS (
    SELECT *,
        (r_score + f_score + m_score)                       AS rfm_total,
        CASE
            WHEN r_score >= 4 AND f_score >= 4 AND m_score >= 4  THEN 'Champions'
            WHEN r_score >= 3 AND f_score >= 3                   THEN 'Loyal Customers'
            WHEN r_score >= 4 AND f_score <= 2                   THEN 'Recent Customers'
            WHEN r_score >= 3 AND f_score <= 3 AND m_score >= 3  THEN 'Potential Loyalists'
            WHEN r_score <= 2 AND f_score >= 3                   THEN 'At Risk'
            WHEN r_score <= 2 AND f_score <= 2 AND m_score >= 3  THEN 'Cant Lose Them'
            WHEN r_score <= 1                                     THEN 'Lost'
            ELSE 'Needs Attention'
        END                                                 AS segment
    FROM rfm_scores
)
SELECT
    segment,
    COUNT(*)                                                AS customers,
    ROUND(AVG(recency_days), 1)                            AS avg_recency_days,
    ROUND(AVG(frequency), 1)                               AS avg_frequency,
    ROUND(AVG(monetary), 2)                                AS avg_monetary,
    SUM(CASE WHEN churn THEN 1 ELSE 0 END)                 AS churned_count,
    ROUND(100.0 * SUM(CASE WHEN churn THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_pct
FROM rfm_segments
GROUP BY segment
ORDER BY avg_monetary DESC;
