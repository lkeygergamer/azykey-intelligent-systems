# Referência da API

## data.py
- `load_iris_data(test_size=0.2, random_state=42)`

## model.py
- `build_pipeline(n_estimators=100)`

## train.py
- `cross_validate_pipeline(pipeline, X_train, y_train, cv=5)`
- `train_pipeline(pipeline, X_train, y_train)`

## evaluate.py
- `evaluate_pipeline(pipeline, X_test, y_test, target_names=None)`
- `plot_confusion_matrix(pipeline, X_test, y_test, target_names=None)`

## optimize.py
- `optimize_hyperparameters(X_train, y_train, param_grid=None, cv=5)`

## report.py
- `generate_html_report(report, cm, target_names, output_path='report.html')` 