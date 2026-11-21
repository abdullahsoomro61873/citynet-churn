# Data Card: `citynet_churn.csv`

- **Source:** CityNet customer records (synthetic/made-up data created for this course).
- **Rows:** 5,000
- **Columns:** 14
- **One row represents:** One customer, with information measured over the last 90 days.
- **Target:** `churned` = 1 if the customer cancelled their subscription in the following month. Share of 1s: **17.92% (896 customers)**.
- **Missing values:**
  - `age`: 411 (8.22%)
  - `payment_method`: 149 (2.98%)
  - All other columns: 0
  - **Total missing values:** 560

## Columns Not Used and Why

- `customer_id`: Excluded because it is a customer identifier, not a predictive feature. It was retained separately for customer identification in prediction reports.
- `cancel_request_logged`: Excluded because it has a strong association with churn and could introduce data leakage.

After separating the target variable (`churned`) and excluding these two columns, **11 input features** remained for model training.

## Known Issues

- The dataset is synthetic and may not represent real-world telecommunications customer behavior.
- The target classes are imbalanced: 17.92% churners and 82.08% non-churners.
- Missing values are present in `age` and `payment_method`; these were handled using median and mode imputation, respectively.
- `cancel_request_logged` has a strong association with churn and requires careful consideration because of potential data leakage.
- Observed relationships between customer features and churn indicate associations, not necessarily direct causes.
- Model performance should be validated on newer, real-world customer data before production use.