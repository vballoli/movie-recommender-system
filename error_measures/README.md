# Evaluation Measures

## Contents

1. RMSE - Root Mean Square Error
2. Precision in Top K
3. Spearman Correlation

### RMSE - Root Mean Square Error

Formula: `(sum((predicted - actual) ** 2) / n) ^ 0.5`

### Precision in Top K

Gives an estimate of how many of the predicted ratings are present in the
top K ratings of the user since only the good one's count in the error measure.

### Spearman Correlation

Formula: `1 - [sum(diff(predicted - actual)^2) / n((n^2)-1)]`
