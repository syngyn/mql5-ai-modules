//+------------------------------------------------------------------+
//|                                               GGTH_Globals.mqh   |
//|                      Copyright 2025, Jason.W.Rusk@gmail.com      |
//|                                                                  |
//|       Declares all global variables for the GGTH project         |
//|         using the 'extern' keyword to allow for modular          |
//|                     compilation and access.                      |
//+------------------------------------------------------------------+
#property copyright "Jason.W.Rusk@gmail.com"
#property version   "3.00"

// ---
// --- FIX: Include necessary dependencies to resolve compiler errors.
// ---
#include <Trade/Trade.mqh>      // Required for the CTrade class definition.
#include "GGTH_CoreTypes.mqh" // Required for ALL custom structs, enums, and #defines.
// ---

// --- Global Handles & Objects ---
extern int atr_handle, macd_handle, rsi_handle, stoch_handle, cci_handle, adx_handle, bb_handle;
extern CTrade trade;

// --- HTTP COMMUNICATION GLOBALS ---
extern string g_daemon_base_url;
extern int g_http_requests_sent;
extern int g_http_successful_responses;
extern int g_http_failed_responses;
extern datetime g_last_http_request_time;
extern string g_last_http_error;
extern bool g_daemon_health_checked;

// --- KELLY CRITERION GLOBAL VARIABLES ---
extern KellyTradeRecord g_kelly_trade_history[];
extern KellyMetrics g_kelly_metrics;
extern double g_current_kelly_fraction;
extern double g_smoothed_kelly_fraction;
extern double g_last_calculated_kelly;
extern datetime g_last_kelly_update;

// --- ADAPTIVE LEARNING & TESTER GLOBAL VARIABLES ---
extern TradeRecord g_trade_history[];
extern AdaptiveMetrics g_adaptive_metrics;
extern MarketCondition g_current_market_condition;
extern TesterResult g_tester_results[];
extern ScalpingPosition g_scalping_position;
extern datetime g_last_adaptation_time;
extern bool g_adaptive_system_initialized;
extern bool g_is_tester_mode;
extern double g_tester_start_balance;
extern double g_max_equity_peak;
extern double g_current_drawdown;

// --- CORE STRATEGY & STATE GLOBALS ---
extern StepPrediction g_step_predictions[];
extern double g_last_predictions[PREDICTION_STEPS];
extern double g_last_confidence_score;
extern double g_accuracy_pct[PREDICTION_STEPS];
extern int    g_total_hits[PREDICTION_STEPS];
extern int    g_total_predictions[PREDICTION_STEPS];
extern double g_active_trade_target_price;

// --- Daily Prediction Globals ---
extern double g_last_daily_prediction_price;
extern ENUM_PREDICTION_DIRECTION g_last_daily_prediction_direction;
extern datetime g_last_daily_prediction_time;

// --- Backtesting Globals ---
extern BacktestPrediction g_backtest_predictions[];
extern int g_backtest_prediction_idx;

// --- Global Parameter Cache ---
// These cache the 'input' variables for easier access across modules.
extern double g_RiskPercent, g_MinimumRiskRewardRatio, g_ATR_SL_Multiplier, g_ATR_TP_Multiplier, g_MinProfitPips, g_TrailingStartPips, g_TrailingStopPips, g_MinimumModelConfidence, g_MinimumSignalConfidence, g_ClassificationSignalThreshold;
extern int    g_RequiredConsistentSteps, g_StaticStopLossPips, g_StaticTakeProfitPips, g_ATR_Period, g_MaxPositionHoldBars, g_ExitBarMinute, g_ADX_Period, g_ADX_Threshold, g_AccuracyLookbackOnInit, g_AccuracyWindowBars, g_PredictionUpdateMinutes;
extern bool   g_EnableTimeBasedExit, g_EnableTrailingStop, g_EnableADXFilter;
extern ENUM_TRADING_MODE   g_TradingLogicMode;
extern ENUM_STOP_LOSS_MODE g_StopLossMode;
extern ENUM_TAKE_PROFIT_MODE g_TakeProfitMode;

// --- Legacy & Connection Status Globals ---
extern datetime g_last_successful_request;
extern datetime g_last_request_attempt;
extern datetime g_last_prediction_time;
extern int g_total_requests_sent;
extern int g_successful_responses;
extern string g_connection_status;
extern string g_last_error;
//+------------------------------------------------------------------+