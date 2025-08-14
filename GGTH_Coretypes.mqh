//+------------------------------------------------------------------+
//|                                              GGTH_CoreTypes.mqh  |
//|                      Copyright 2025, Jason.W.Rusk@gmail.com      |
//|                                                                  |
//|      Contains all core data structures, enumerations, and        |
//|        constants for the GGTH Expert Advisor project.            |
//+------------------------------------------------------------------+
#property copyright "Jason.W.Rusk@gmail.com"
#property version   "3.00"

// --- TYPE DEFINITIONS ---
enum ENUM_TRADING_MODE { MODE_TRADING_DISABLED, MODE_REGRESSION_ONLY, MODE_COMBINED };
enum ENUM_STOP_LOSS_MODE { SL_ATR_BASED, SL_STATIC_PIPS };
enum ENUM_TAKE_PROFIT_MODE { TP_REGRESSION_TARGET, TP_ATR_MULTIPLE, TP_STATIC_PIPS };
enum ENUM_TARGET_BAR { H_PLUS_1=0, H_PLUS_2, H_PLUS_3, H_PLUS_4, H_PLUS_5 };
enum ENUM_ADAPTIVE_MODE { ADAPTIVE_DISABLED, ADAPTIVE_CONSERVATIVE, ADAPTIVE_AGGRESSIVE };
enum ENUM_PREDICTION_DIRECTION { DIR_BULLISH, DIR_BEARISH, DIR_NEUTRAL };

// --- Constants ---
#define PREDICTION_STEPS 5
#define SEQ_LEN 20
#define FEATURE_COUNT 15
#define GUI_PREFIX "GGTHGUI_"
#define BACKTEST_PREDICTIONS_FILE "backtest_predictions.csv"
#define ADAPTIVE_LEARNING_FILE "adaptive_learning_data.csv"
#define TESTER_RESULTS_FILE "tester_results.csv"

// --- HTTP & DAEMON STRUCTURES ---
struct DaemonResponse { 
    double prices[PREDICTION_STEPS]; 
    double confidence_score; 
    double buy_prob; 
    double sell_prob; 
};

// --- KELLY CRITERION STRUCTURES ---
struct KellyTradeRecord {
    datetime trade_time;
    bool was_profitable;
    double profit_pips;
    double loss_pips;
    double r_multiple;           // Profit/Risk ratio
    double confidence_used;
    ENUM_ORDER_TYPE trade_type;
};

struct KellyMetrics {
    double win_rate;
    double avg_win_pips;
    double avg_loss_pips;
    double avg_r_multiple;
    double profit_factor;
    int total_trades;
    int winning_trades;
    double kelly_fraction;
    double confidence_adjusted_kelly;
};

// --- ADAPTIVE LEARNING STRUCTURES ---
struct TradeRecord 
{
    datetime trade_time;
    double entry_price;
    double exit_price;
    double profit;
    double confidence_used;
    double risk_used;
    int prediction_step_used;
    double accuracy_at_trade;
    bool was_profitable;
    double market_volatility;
    int hour_of_day;
    double drawdown_at_entry;
    double prediction_error;
    bool from_tester;
};

struct AdaptiveMetrics 
{
    double avg_accuracy[PREDICTION_STEPS];
    double step_weights[PREDICTION_STEPS];
    double dynamic_confidence_threshold;
    double dynamic_risk_multiplier;
    double success_rate_last_n_trades;
    double avg_profit_per_trade;
    double volatility_factor;
    int trades_analyzed;
    double max_drawdown_experienced;
    double best_performing_step;
    double worst_performing_step;
};

struct MarketCondition 
{
    double volatility_level;
    int trend_direction;  // -1=bear, 0=sideways, 1=bull
    int session_hour;
    double recent_accuracy;
    double correlation_strength;
};

struct TesterResult 
{
    datetime test_date;
    string symbol;
    string timeframe;
    double total_profit;
    double max_drawdown;
    int total_trades;
    double success_rate;
    double sharpe_ratio;
    double best_confidence_threshold;
    double best_risk_multiplier;
    double avg_step_weight[PREDICTION_STEPS];
    double market_conditions_score;
};

// --- SCALPING STRATEGY STRUCTURES ---
struct ScalpingPosition
{
    bool is_active;
    datetime entry_time;
    double entry_price;
    double target_price;
    int target_step;
    ENUM_ORDER_TYPE order_type;
    double lot_size;
    ulong ticket;
    datetime timeout_time;
    string comment;
};

// --- ACCURACY AND BACKTESTING STRUCTURES ---
struct StepPrediction 
{
    double target_price;
    datetime prediction_bar_time;
    datetime window_end_time;
    ENUM_PREDICTION_DIRECTION direction;
    int step;
    bool evaluated;
    bool hit_within_window;
};

struct BacktestPrediction { 
    datetime timestamp; 
    double buy_prob;
    double sell_prob; 
    double hold_prob; 
    double confidence_score; 
    double predicted_prices[PREDICTION_STEPS]; 
};
//+------------------------------------------------------------------+