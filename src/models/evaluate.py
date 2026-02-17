"""
Model Evaluation Module
Implements evaluation metrics focused on prediction accuracy and direction
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, median_absolute_error, mean_squared_error, r2_score


def evaluate_model(y_true, y_pred, model_name="Model"):
    """
    Comprehensive model evaluation with focus on error metrics
    
    Args:
        y_true: True prices
        y_pred: Predicted prices
        model_name: Name of model for reporting
        
    Returns:
        dict with all evaluation metrics
    """
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Basic error metrics
    mae = mean_absolute_error(y_true, y_pred)
    median_ae = median_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # R² (for reference, but not primary metric)
    r2 = r2_score(y_true, y_pred)
    
    # Directional accuracy
    directional_acc = calculate_directional_accuracy(y_true, y_pred)
    
    # Asymmetric loss analysis
    asymmetric = analyze_asymmetric_loss(y_true, y_pred)
    
    # Percentage errors
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    results = {
        'model_name': model_name,
        'MAE': round(mae, 2),
        'Median_AE': round(median_ae, 2),
        'RMSE': round(rmse, 2),
        'R2': round(r2, 3),
        'MAPE': round(mape, 2),
        'Directional_Accuracy': round(directional_acc, 3),
        'Avg_Over_Prediction': round(asymmetric['avg_over_pred'], 2),
        'Avg_Under_Prediction': round(asymmetric['avg_under_pred'], 2),
        'Over_Pred_Count': asymmetric['over_pred_count'],
        'Under_Pred_Count': asymmetric['under_pred_count']
    }
    
    return results


def calculate_directional_accuracy(y_true, y_pred):
    """
    Calculate directional accuracy
    Measures how often the model correctly predicts price movement direction
    
    Args:
        y_true: True prices
        y_pred: Predicted prices
        
    Returns:
        Directional accuracy (0 to 1)
    """
    # Compare if prediction is above/below mean correctly
    mean_true = np.mean(y_true)
    
    # True direction: above or below mean
    true_direction = y_true > mean_true
    pred_direction = y_pred > mean_true
    
    # Accuracy
    directional_acc = np.mean(true_direction == pred_direction)
    
    return directional_acc


def analyze_asymmetric_loss(y_true, y_pred):
    """
    Analyze asymmetric prediction errors
    Under-prediction is worse for booking apps (user books, price actually higher)
    Over-prediction is less bad (user waits, might miss deal but doesn't overpay)
    
    Args:
        y_true: True prices
        y_pred: Predicted prices
        
    Returns:
        dict with asymmetric loss analysis
    """
    errors = y_pred - y_true
    
    # Over-predictions (predicted > actual)
    over_pred_mask = errors > 0
    over_predictions = errors[over_pred_mask]
    avg_over_pred = over_predictions.mean() if len(over_predictions) > 0 else 0
    
    # Under-predictions (predicted < actual) - MORE PROBLEMATIC
    under_pred_mask = errors < 0
    under_predictions = errors[under_pred_mask]
    avg_under_pred = under_predictions.mean() if len(under_predictions) > 0 else 0
    
    return {
        'avg_over_pred': avg_over_pred,
        'avg_under_pred': avg_under_pred,
        'over_pred_count': int(over_pred_mask.sum()),
        'under_pred_count': int(under_pred_mask.sum())
    }


def print_evaluation_report(results):
    """
    Print formatted evaluation report
    
    Args:
        results: Evaluation results dict
    """
    print("\n" + "="*70)
    print(f"MODEL EVALUATION REPORT: {results['model_name']}")
    print("="*70)
    
    print("\n PRIMARY METRICS:")
    print(f"  • MAE (Mean Absolute Error):      ${results['MAE']:.2f}")
    print(f"  • Median AE:                       ${results['Median_AE']:.2f}")
    print(f"  • Directional Accuracy:            {results['Directional_Accuracy']:.1%}")
    print(f"  • RMSE:                            ${results['RMSE']:.2f}")
    
    print("\n  ASYMMETRIC LOSS ANALYSIS:")
    print(f"  • Over-predictions (pred > actual): {results['Over_Pred_Count']} cases")
    print(f"    Average over-prediction:          ${results['Avg_Over_Prediction']:.2f}")
    print(f"  • Under-predictions (pred < actual): {results['Under_Pred_Count']} cases")
    print(f"    Average under-prediction:         ${results['Avg_Under_Prediction']:.2f}")
    
    # Assessment
    if abs(results['Avg_Under_Prediction']) > abs(results['Avg_Over_Prediction']):
        print("\n  Model tends to under-predict (risky for users)")
    else:
        print("\n  Model is conservative (slight over-prediction)")
    
    print("\n" + "="*70)