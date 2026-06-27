-- Dimensions
CREATE TABLE IF NOT EXISTS dim_date (
    date_id     SERIAL PRIMARY KEY,
    full_date   DATE NOT NULL UNIQUE,
    year        INT,
    quarter     INT,
    month       INT,
    month_name  VARCHAR(20),
    week        INT,
    day_of_week VARCHAR(20)
);

CREATE TABLE IF NOT EXISTS dim_customer (
    customer_id     VARCHAR(50) PRIMARY KEY,
    city            VARCHAR(100),
    state           VARCHAR(100),
    rfm_segment     VARCHAR(30)
);

CREATE TABLE IF NOT EXISTS dim_product (
    product_id       VARCHAR(50) PRIMARY KEY,
    category_name    VARCHAR(100),
    product_name     VARCHAR(200),
    weight_kg        NUMERIC(8,2)
);

CREATE TABLE IF NOT EXISTS dim_department (
    department_id   SERIAL PRIMARY KEY,
    department_name VARCHAR(100),
    job_role        VARCHAR(100)
);

CREATE TABLE IF NOT EXISTS dim_geography (
    geo_id      SERIAL PRIMARY KEY,
    city        VARCHAR(100),
    state       VARCHAR(100),
    region      VARCHAR(100)
);

CREATE TABLE IF NOT EXISTS dim_attrition_label (
    label_id     SERIAL PRIMARY KEY,
    attrition    VARCHAR(5),
    risk_tier    VARCHAR(20)
);

-- Fact tables
CREATE TABLE IF NOT EXISTS fact_orders (
    order_id             VARCHAR(50) PRIMARY KEY,
    customer_id          VARCHAR(50) REFERENCES dim_customer(customer_id),
    product_id           VARCHAR(50) REFERENCES dim_product(product_id),
    date_id              INT         REFERENCES dim_date(date_id),
    geo_id               INT         REFERENCES dim_geography(geo_id),
    order_status         VARCHAR(50),
    payment_type         VARCHAR(50),
    payment_installments INT,
    price                NUMERIC(10,2),
    freight_value        NUMERIC(10,2),
    review_score         INT,
    delivery_days        INT
);

CREATE TABLE IF NOT EXISTS fact_employees (
    employee_id              SERIAL PRIMARY KEY,
    department_id            INT REFERENCES dim_department(department_id),
    date_id                  INT REFERENCES dim_date(date_id),
    label_id                 INT REFERENCES dim_attrition_label(label_id),
    age                      INT,
    gender                   VARCHAR(10),
    monthly_income           NUMERIC(10,2),
    job_satisfaction         INT,
    years_at_company         INT,
    overtime                 VARCHAR(5),
    distance_from_home       INT,
    education                INT,
    environment_satisfaction INT,
    performance_rating       INT
);
