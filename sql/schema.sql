
DROP TABLE IF EXISTS churn_predictions CASCADE;
DROP TABLE IF EXISTS support_tickets CASCADE;
DROP TABLE IF EXISTS transactions CASCADE;
DROP TABLE IF EXISTS customers CASCADE;

CREATE TABLE customers (
    customer_id         VARCHAR(20) PRIMARY KEY,
    gender              VARCHAR(10),
    senior_citizen      BOOLEAN,
    partner             BOOLEAN,
    dependents          BOOLEAN,
    tenure              INT,
    phone_service       BOOLEAN,
    multiple_lines      VARCHAR(20),
    internet_service    VARCHAR(20),
    online_security     VARCHAR(20),
    online_backup       VARCHAR(20),
    device_protection   VARCHAR(20),
    tech_support        VARCHAR(20),
    streaming_tv        VARCHAR(20),
    streaming_movies    VARCHAR(20),
    contract            VARCHAR(20),
    paperless_billing   BOOLEAN,
    payment_method      VARCHAR(30),
    monthly_charges     DECIMAL(10, 2),
    total_charges       DECIMAL(10, 2),
    churn               BOOLEAN,
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE transactions (
    transaction_id      SERIAL PRIMARY KEY,
    customer_id         VARCHAR(20) REFERENCES customers(customer_id),
    transaction_date    DATE,
    amount              DECIMAL(10, 2),
    payment_method      VARCHAR(30),
    status              VARCHAR(20),
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE support_tickets (
    ticket_id           SERIAL PRIMARY KEY,
    customer_id         VARCHAR(20) REFERENCES customers(customer_id),
    ticket_date         DATE,
    issue_category      VARCHAR(50),
    priority            VARCHAR(10),
    resolved            BOOLEAN,
    resolution_days     INT,
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE churn_predictions (
    prediction_id       SERIAL PRIMARY KEY,
    customer_id         VARCHAR(20) REFERENCES customers(customer_id),
    churn_probability   DECIMAL(5, 4),
    churn_predicted     BOOLEAN,
    model_version       VARCHAR(20),
    prediction_date     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    top_reason          TEXT
);

CREATE INDEX idx_customers_churn    ON customers(churn);
CREATE INDEX idx_customers_contract ON customers(contract);
CREATE INDEX idx_customers_tenure   ON customers(tenure);
CREATE INDEX idx_transactions_cust  ON transactions(customer_id);
CREATE INDEX idx_tickets_cust       ON support_tickets(customer_id);
CREATE INDEX idx_predictions_cust   ON churn_predictions(customer_id);

CREATE VIEW v_churn_by_segment AS
SELECT
    contract,
    internet_service,
    payment_method,
    COUNT(*)                                                                AS total_customers,
    SUM(CASE WHEN churn THEN 1 ELSE 0 END)                                 AS churned,
    ROUND(100.0 * SUM(CASE WHEN churn THEN 1 ELSE 0 END) / COUNT(*), 2)   AS churn_rate_pct,
    ROUND(AVG(monthly_charges), 2)                                          AS avg_monthly_charges,
    ROUND(AVG(tenure), 1)                                                   AS avg_tenure_months
FROM customers
GROUP BY contract, internet_service, payment_method
ORDER BY churn_rate_pct DESC;
