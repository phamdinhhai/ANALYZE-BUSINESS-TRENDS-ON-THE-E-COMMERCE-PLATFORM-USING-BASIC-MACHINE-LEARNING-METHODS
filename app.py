from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

models = {
    'Linear Regression': LinearRegression(),
    'Lasso': Lasso(alpha=0.1),
    'Ridge': Ridge(alpha=1),
    'Decision Tree': DecisionTreeRegressor(max_depth=10),
    'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10),
    'AdaBoost': AdaBoostRegressor(n_estimators=50, learning_rate=0.1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5),
    'XGBoost': XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)
}

def evaluate_models(X_train, X_test, y_train, y_test, model_type):
    results = {}
    predictions = {}
    for name, model in models.items():
        print(f'Training {name}...')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        num_params = len(model.coef_) if hasattr(model, 'coef_') else model.get_params().get('n_estimators', 1)
        aic = calculate_aic(len(y_test), mean_squared_error(y_test, y_pred), num_params)
        
        predictions[name] = y_pred
        results[name] = {'MAE': mae, 'MSE': mse, 'R^2': r2, 'AIC': aic}
        print(f'{name}: MAE = {mae}, MSE = {mse}, R^2 = {r2}, AIC = {aic}')
        
        # Lưu mô hình
        joblib.dump(model, f'{model_type}_{name}.joblib')
    
    return results, predictions

results_single, predictions_single = evaluate_models(X_train_single_scaled, X_test_single_scaled, y_train_single, y_test_single, 'single')
