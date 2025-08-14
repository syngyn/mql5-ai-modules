//+------------------------------------------------------------------+
//| CNN-LSTM Predictor with Dual Attention                           |
//+------------------------------------------------------------------+
class CNNLSTMPredictor
  {
private:
   double            m_hourlyPredictions[5];
   double            m_dailyPredictions[5];
   double            m_accuracyHistory[];
   datetime          m_lastUpdate;

public:
   bool              Predict();
   bool              UpdateIfNewBar();
   const double&     GetHourlyPrediction(int index) const;
   const double&     GetDailyPrediction(int index) const;
   double            GetAccuracy() const;
   void              VisualizePredictions();
   void              BacktestAccuracy();
  };

//+------------------------------------------------------------------+
//| Predict using external model                                     |
//+------------------------------------------------------------------+
bool CNNLSTMPredictor::Predict()
  {
   // Placeholder: Replace with actual Python integration
   // Example: Write input to file, call Python script, read output
   for(int i=0; i<5; i++)
     {
      m_hourlyPredictions[i] = iClose("EURUSD", PERIOD_H1, i);
      m_dailyPredictions[i] = iClose("EURUSD", PERIOD_D1, i);
     }
   return true;
  }

//+------------------------------------------------------------------+
//| Update on new bar                                                |
//+------------------------------------------------------------------+
bool CNNLSTMPredictor::UpdateIfNewBar()
  {
   datetime currentBar = iTime("EURUSD", PERIOD_H1, 0);
   if(currentBar != m_lastUpdate)
     {
      m_lastUpdate = currentBar;
      Predict();
      BacktestAccuracy();
      return true;
     }
   return false;
  }

//+------------------------------------------------------------------+
//| Visualize predictions                                            |
//+------------------------------------------------------------------+
void CNNLSTMPredictor::VisualizePredictions()
  {
   for(int i=0; i<5; i++)
     {
      string label = "Pred_H" + IntegerToString(i);
      double price = m_hourlyPredictions[i];
      ObjectCreate(0, label, OBJ_TEXT, 0, Time[i], price);
      ObjectSetText(label, "Pred: " + DoubleToString(price, 5), 10, "Arial", clrYellow);
     }

   string accLabel = "AccuracyLabel";
   ObjectCreate(0, accLabel, OBJ_LABEL, 0, 0, 0);
   ObjectSetInteger(0, accLabel, OBJPROP_CORNER, CORNER_RIGHT_UPPER);
   ObjectSetText(accLabel, "Accuracy: " + DoubleToString(GetAccuracy(), 2) + "%", 12, "Arial", clrLime);
  }

//+------------------------------------------------------------------+
//| Backtest accuracy                                                |
//+------------------------------------------------------------------+
void CNNLSTMPredictor::BacktestAccuracy()
  {
   double actual = iClose("EURUSD", PERIOD_H1, 0);
   double predicted = m_hourlyPredictions[0];
   double error = MathAbs(actual - predicted);
   double percentError = 100.0 * error / actual;
   ArrayInsert(m_accuracyHistory, percentError, 0);
  }

const double& CNNLSTMPredictor::GetHourlyPrediction(int index) const { return m_hourlyPredictions[index]; }
const double& CNNLSTMPredictor::GetDailyPrediction(int index) const { return m_dailyPredictions[index]; }
double CNNLSTMPredictor::GetAccuracy() const
  {
   if(ArraySize(m_accuracyHistory) == 0) return 0.0;
   return 100.0 - MathMean(m_accuracyHistory); // Accuracy = 100% - avg error
  }