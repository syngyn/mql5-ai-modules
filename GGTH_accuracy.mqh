//+------------------------------------------------------------------+
//|                                              GGTH_Accuracy.mqh   |
//|                      Copyright 2025, Jason.W.Rusk@gmail.com      |
//|                                                                  |
//|          Module for tracking predictive model accuracy.          |
//+------------------------------------------------------------------+
#property copyright "Jason.W.Rusk@gmail.com"
#property version   "3.00"

// --- Required Dependencies ---
#include "GGTH_CoreTypes.mqh"
#include "GGTH_Globals.mqh"

// --- Forward Declaration to avoid include loop ---
bool FindPredictionForBar(datetime bar_time, BacktestPrediction &found_pred, bool reset_search_index=false);

//+------------------------------------------------------------------+
//| Adds new predictions to the tracking array for later evaluation. |
//+------------------------------------------------------------------+
void AddStepPredictions(const double &predicted_prices[], datetime current_bar_time)
{
    double prediction_bar_close = iClose(_Symbol, PERIOD_H1, 1);
    if(prediction_bar_close <= 0) return;
    
    for(int step = 0; step < PREDICTION_STEPS; step++)
    {
        StepPrediction pred;
        pred.target_price = predicted_prices[step];
        pred.prediction_bar_time = current_bar_time;
        pred.window_end_time = current_bar_time + (g_AccuracyWindowBars * PeriodSeconds(PERIOD_H1));
        pred.direction = (predicted_prices[step] > prediction_bar_close) ? DIR_BULLISH : DIR_BEARISH;
        pred.step = step;
        pred.evaluated = false;
        
        int size = ArraySize(g_step_predictions);
        ArrayResize(g_step_predictions, size + 1);
        g_step_predictions[size] = pred;
    }
}

//+------------------------------------------------------------------+
//| Checks if a prediction was 'hit' within its evaluation window.   |
//+------------------------------------------------------------------+
bool CheckPriceHitInWindow(const StepPrediction &prediction)
{
    int prediction_bar_index = iBarShift(_Symbol, PERIOD_H1, prediction.prediction_bar_time);
    if(prediction_bar_index < 0) return false;
    
    double base_price = iClose(_Symbol, PERIOD_H1, prediction_bar_index + 1);
    
    for(int i = 0; i < g_AccuracyWindowBars; i++)
    {
        int bar_index = prediction_bar_index - i;
        if(bar_index < 0) continue;
        
        double bar_high = iHigh(_Symbol, PERIOD_H1, bar_index);
        double bar_low = iLow(_Symbol, PERIOD_H1, bar_index);
        
        if((prediction.direction == DIR_BULLISH && bar_high >= prediction.target_price) ||
           (prediction.direction == DIR_BEARISH && bar_low <= prediction.target_price))
        {
            return true;
        }
    }
    return false;
}

//+------------------------------------------------------------------+
//| Evaluates completed prediction windows to update accuracy stats. |
//+------------------------------------------------------------------+
void CheckStepPredictionAccuracy()
{
    if(ArraySize(g_step_predictions) == 0) return;
    
    datetime current_time = TimeCurrent();
    
    for(int i = ArraySize(g_step_predictions) - 1; i >= 0; i--)
    {
        if(g_step_predictions[i].evaluated) continue;
        
        if(current_time >= g_step_predictions[i].window_end_time)
        {
            bool was_hit = CheckPriceHitInWindow(g_step_predictions[i]);
            g_total_predictions[g_step_predictions[i].step]++;
            if(was_hit) g_total_hits[g_step_predictions[i].step]++;
            
            g_step_predictions[i].evaluated = true;
        }
    }
    
    // Clean up old, evaluated predictions to save memory
    if(ArraySize(g_step_predictions) > 100)
    {
       int removed=0;
       for(int i=ArraySize(g_step_predictions)-1; i>=0; i--)
       {
          if(g_step_predictions[i].evaluated)
          {
             ArrayRemove(g_step_predictions,i,1);
             removed++;
             if(removed>50) break; // Clean in batches
          }
       }
    }
}

//+------------------------------------------------------------------+
//| Calculates initial accuracy based on historical data on init.    |
//+------------------------------------------------------------------+
void CalculateImprovedInitialAccuracy()
{
    if(g_AccuracyLookbackOnInit <= 0 || !EnablePricePredictionDisplay || ArraySize(g_backtest_predictions) == 0) return;
    
    MqlRates price_data[];
    int bars_needed = g_AccuracyLookbackOnInit + g_AccuracyWindowBars + 10;
    if(CopyRates(_Symbol, PERIOD_H1, 0, bars_needed, price_data) < bars_needed) return;
    ArraySetAsSeries(price_data, true);
    
    bool first_search = true;
    for(int i = g_AccuracyLookbackOnInit; i >= g_AccuracyWindowBars; i--)
    {
        if(i >= ArraySize(price_data)) continue;
        
        datetime prediction_time = price_data[i].time;
        BacktestPrediction pred_data;
        if(!FindPredictionForBar(prediction_time, pred_data, first_search)) continue;
        first_search = false;
        
        double base_price = price_data[i].close;
        for(int step = 0; step < PREDICTION_STEPS; step++)
        {
            double predicted_price = pred_data.predicted_prices[step];
            ENUM_PREDICTION_DIRECTION direction = (predicted_price > base_price) ? DIR_BULLISH : DIR_BEARISH;
            bool was_hit = false;
            
            for(int window_bar = 0; window_bar < g_AccuracyWindowBars; window_bar++)
            {
                int check_index = i - window_bar - 1;
                if(check_index < 0 || check_index >= ArraySize(price_data)) continue;
                
                if((direction == DIR_BULLISH && price_data[check_index].high >= predicted_price) ||
                   (direction == DIR_BEARISH && price_data[check_index].low <= predicted_price))
                { 
                   was_hit = true; 
                   break; 
                }
            }
            g_total_predictions[step]++;
            if(was_hit) g_total_hits[step]++;
        }
    }
    g_backtest_prediction_idx = 0; // Reset for OnTick
}
//+------------------------------------------------------------------+