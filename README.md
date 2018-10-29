# CS F469 Movie Recommender System

This project is a recommender system based on the MovieLens dataset built using

1. Collaborative filtering
2. SVD - Singular Value Decomposition
3. CUR

## Requirements

1. Numpy
2. Pandas
3. Jupyter

## Build instructions

`bash ./run.sh`

## Building docs

Navigate to `root/docs` and run
`make html`

## Results

| Recommender System algorithm | RMSE | Precision on Top K  | Spearman Rank Correlation | Time taken for prediction |
|:-------------:|:-------------:|:-----:|:-----:| :-----:|
| Collaborative Filtering | 1.31 | 0.18 | 0.99 | 19.28s |
| Collaborative Filtering with Baseline approach | 1.18 | 0.19 | 0.99 | 18.5s |
| SVD | 1.2 | 0.31 | 0.99 | 30s |
| SVD with 90% retention | 1.3 | 0.31 | 0.99 | 28s |
| CUR | 0.8 | 0.31 | 0.99 | 2.5s |
| CUR with 90% retention | 0.9 | 0.31 | 0.99 | 5s |
## Dataset

Link: `http://files.grouplens.org/datasets/movielens/ml-100k.zip`

## Contribution Guidelines

1. Branch the repository by feature and submit pull request.
