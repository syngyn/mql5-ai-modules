import os
import json
import time
import sys
import numpy as np
import pandas as pd
import torch
import joblib
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading
import logging
from urllib.parse import urlparse, parse_qs

# --- MODIFIED IMPORTS ---
# Import the new model and dataset for online learning
from train import CNN_LSTM_AttentionModel, Config, ForexDataset
from torch.utils.data import DataLoader
from utils import generate_probabilities, generate_confidence


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('daemon.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- GLOBAL VARIABLES & CONFIG ---
cfg = Config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
feature_scaler = None
label_scaler = None

# --- NEW: Online Learning Configuration ---
FEEDBACK_FILE = "online_feedback_data.csv"
RETRAIN_THRESHOLD = 100  # Number of new data points to collect before retraining
model_lock = threading.Lock() # Lock for thread-safe model prediction and swapping
feedback_lock = threading.Lock() # Lock for thread-safe writing to the feedback file
retraining_thread = None

# Server configuration
SERVER_HOST = "127.0.0.1"  # localhost only for security
SERVER_PORT = 8888
MAX_REQUEST_SIZE = 1024 * 1024  # 1MB max request size

# Statistics tracking
request_count = 0
successful_predictions = 0
failed_predictions = 0
start_time = time.time()

class PredictionHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        """Override to use our logger instead of stderr"""
        logger.info(f"{self.address_string()} - {format % args}")
    
    def do_GET(self):
        """Handle GET requests for health check and stats (from original)"""
        global request_count, successful_predictions, failed_predictions, start_time
        
        parsed_path = urlparse(self.path)
        
        if parsed_path.path == '/health':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            health_data = {
                "status": "healthy",
                "model_loaded": model is not None,
                "scalers_loaded": feature_scaler is not None and label_scaler is not None,
                "device": str(device),
                "uptime_seconds": int(time.time() - start_time),
                "seq_len": cfg.SEQ_LEN,
                "feature_count": cfg.FEATURE_COUNT,
                "prediction_steps": cfg.PREDICTION_STEPS,
                "retraining_status": "in_progress" if retraining_thread and retraining_thread.is_alive() else "idle"
            }
            
            self.wfile.write(json.dumps(health_data, indent=2).encode())
            
        elif parsed_path.path == '/stats':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            stats_data = {
                "total_requests": request_count,
                "successful_predictions": successful_predictions,
                "failed_predictions": failed_predictions,
                "success_rate": (successful_predictions / max(request_count, 1)) * 100,
                "uptime_seconds": int(time.time() - start_time),
                "requests_per_minute": (request_count / max((time.time() - start_time) / 60, 1))
            }
            
            self.wfile.write(json.dumps(stats_data, indent=2).encode())
            
        else:
            self.send_error(404, "Endpoint not found")
    
    def do_POST(self):
        """MODIFIED: Route POST requests to the correct handler."""
        parsed_path = urlparse(self.path)
        
        if parsed_path.path in ['/predict', '/']:
            self.handle_predict()
        elif parsed_path.path == '/feedback':
            self.handle_feedback()
        else:
            self.send_error(404, "Endpoint not found")

    def handle_predict(self):
        """Handles POST requests for predictions (from original)"""
        global request_count, successful_predictions, failed_predictions
        request_count += 1
        
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length > MAX_REQUEST_SIZE:
                self.send_error(413, "Request too large"); return
            
            post_data = self.rfile.read(content_length)
            request_data = json.loads(post_data.decode('utf-8'))
            
            start_time_processing = time.time()
            response_data = process_request(request_data)
            processing_time = (time.time() - start_time_processing) * 1000
            
            response_data["processing_time_ms"] = round(processing_time, 2)
            
            if response_data.get("status") == "success":
                successful_predictions += 1
                self.send_response(200)
            else:
                failed_predictions += 1
                # Use 400 for bad request, 500 for server error
                error_code = 400 if "message" in response_data else 500
                self.send_response(error_code)

            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(response_data, indent=2).encode())
            
            logger.info(f"Request processed in {processing_time:.2f}ms. Status: {response_data.get('status')}")
            
        except Exception as e:
            failed_predictions += 1
            logger.error(f"Error processing prediction request: {e}")
            self.send_error(500, f"Internal Server Error: {e}")

    def handle_feedback(self):
        """NEW: Handles feedback data from the MQL5 EA for online learning."""
        logger.info("Received feedback request.")
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length)
            feedback_data = json.loads(post_data.decode('utf-8'))

            if 'features' not in feedback_data or 'actuals' not in feedback_data:
                self.send_error(400, "Invalid feedback format."); return

            with feedback_lock:
                df_new = pd.DataFrame([feedback_data])
                # Flatten lists into separate columns for robust CSV storage
                features_df = pd.DataFrame(df_new['features'].tolist(), index=df_new.index, columns=[f'f_{i}' for i in range(len(df_new['features'].iloc[0]))])
                actuals_df = pd.DataFrame(df_new['actuals'].tolist(), index=df_new.index, columns=[f'a_{i}' for i in range(len(df_new['actuals'].iloc[0]))])
                df_to_save = pd.concat([features_df, actuals_df], axis=1)

                is_new_file = not os.path.exists(FEEDBACK_FILE)
                df_to_save.to_csv(FEEDBACK_FILE, mode='a', header=is_new_file, index=False)
                
                num_records = sum(1 for line in open(FEEDBACK_FILE)) - 1
                logger.info(f"Feedback saved. Total records for retraining: {num_records}/{RETRAIN_THRESHOLD}")
                if num_records >= RETRAIN_THRESHOLD:
                    trigger_retraining()

            self.send_response(200)
            self.send_header('Content-Type', 'application/json'); self.end_headers()
            self.wfile.write(json.dumps({"status": "feedback received"}).encode())
        except Exception as e:
            logger.error(f"Error processing feedback: {e}")
            self.send_error(500, f"Internal server error: {e}")

    def do_OPTIONS(self):
        """Handle CORS preflight requests (from original)"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

def verify_paths_and_permissions():
    """Verify model files exist and are accessible (from original)"""
    logger.info("Verifying model files and permissions...")
    paths_to_check = {
        "Model": cfg.MODEL_PATH,
        "Feature Scaler": cfg.FEATURE_SCALER_PATH,
        "Label Scaler": cfg.LABEL_SCALER_PATH
    }
    missing_files = [name for name, path in paths_to_check.items() if not os.path.exists(path)]
    if missing_files:
        logger.error(f"Missing required files: {', '.join(missing_files)}")
        logger.error("Please run 'python train.py' to generate model files.")
        return False
    logger.info("All required files found.")
    return True

def load_models_and_scalers():
    """Load the ML models and scalers, updated for the new model"""
    global model, feature_scaler, label_scaler
    logger.info("Loading models and scalers...")
    try:
        # --- UPDATED to load the CNN_LSTM_AttentionModel ---
        model = CNN_LSTM_AttentionModel(
            input_dim=cfg.FEATURE_COUNT, hidden_dim=100, output_dim=cfg.PREDICTION_STEPS, n_layers=2
        )
        model.load_state_dict(torch.load(cfg.MODEL_PATH, map_location=device))
        model.to(device)
        model.eval()
        logger.info(f"Model '{cfg.MODEL_PATH}' (CNN-LSTM-Attention) loaded on {device}.")
        
        feature_scaler = joblib.load(cfg.FEATURE_SCALER_PATH)
        label_scaler = joblib.load(cfg.LABEL_SCALER_PATH)
        logger.info("Scalers loaded successfully.")
        return True
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        import traceback; logger.error(traceback.format_exc())
        return False

def process_request(request_data):
    """Process a prediction request (from original, with thread safety added)"""
    try:
        # Detailed validation from your original script
        required_fields = ['features', 'current_price', 'atr']
        if not all(field in request_data for field in required_fields):
            return {"status": "error", "message": "Missing required fields."}
        
        features_flat = request_data['features']
        expected_size = cfg.SEQ_LEN * cfg.FEATURE_COUNT
        if len(features_flat) != expected_size:
            return {"status": "error", "message": f"Invalid features array size. Expected {expected_size}, got {len(features_flat)}"}
        
        current_price = float(request_data['current_price'])
        atr_val = float(request_data['atr'])

        # Reshape and scale features
        features_np = np.array(features_flat).reshape(cfg.SEQ_LEN, cfg.FEATURE_COUNT)
        features_scaled = feature_scaler.transform(features_np)
        features_tensor = torch.tensor([features_scaled], dtype=torch.float32).to(device)
        
        # --- ADDED: Thread-safe prediction using the lock ---
        with model_lock:
            with torch.no_grad():
                prediction_scaled = model(features_tensor).cpu().numpy()
        
        predicted_price_diffs = label_scaler.inverse_transform(prediction_scaled)[0]
        predicted_prices = current_price + predicted_price_diffs
        
        buy_prob, sell_prob = generate_probabilities(predicted_prices, current_price)
        confidence = generate_confidence(predicted_prices, atr_val)
        
        if np.any(np.isnan(predicted_prices)) or np.any(np.isinf(predicted_prices)):
            return {"status": "error", "message": "Model generated invalid (NaN/Inf) predictions"}

        # Return the detailed response from your original script
        return {
            "status": "success",
            "predicted_prices": predicted_prices.tolist(),
            "confidence_score": float(confidence),
            "buy_probability": float(buy_prob),
            "sell_probability": float(sell_prob),
            "request_id": request_data.get('request_id', 'unknown'),
            "current_price": current_price,
            "atr_value": atr_val
        }
    except Exception as e:
        logger.error(f"Error in process_request: {e}")
        import traceback; logger.error(traceback.format_exc())
        return {"status": "error", "message": f"Internal server error: {e}"}

# --- NEW: Online Learning Functions ---

def run_online_training():
    """Loads feedback data and fine-tunes the existing model."""
    logger.info("--- Starting Online Fine-Tuning Process ---")
    try:
        with feedback_lock:
            if not os.path.exists(FEEDBACK_FILE): return False
            df = pd.read_csv(FEEDBACK_FILE)
        
        logger.info(f"Loaded {len(df)} new data points for fine-tuning.")
        
        feature_cols = [f for f in df.columns if f.startswith('f_')]
        label_cols = [a for a in df.columns if a.startswith('a_')]
        X_new, y_new = df[feature_cols].values, df[label_cols].values
        
        X_new_scaled = feature_scaler.transform(X_new)
        y_new_scaled = label_scaler.transform(y_new)
        
        online_dataset = ForexDataset(X_new_scaled, y_new_scaled, cfg.SEQ_LEN)
        online_loader = DataLoader(online_dataset, batch_size=16, shuffle=True)
        
        with model_lock:
            model.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE / 10)
            criterion = torch.nn.MSELoss()
            
            logger.info("Fine-tuning for 2 epochs with new data...")
            for epoch in range(2):
                for features, labels in online_loader:
                    features, labels = features.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(features)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                logger.info(f"Fine-tuning epoch {epoch+1} completed. Loss: {loss.item():.6f}")
            
            model.eval()
            torch.save(model.state_dict(), cfg.MODEL_PATH)

        logger.info("--- Online Fine-Tuning Complete. Model updated. ---")
        
        with feedback_lock:
            archive_path = f"archive_feedback_{int(time.time())}.csv"
            os.rename(FEEDBACK_FILE, archive_path)
            logger.info(f"Feedback data archived to {archive_path}")
        return True
    except Exception as e:
        logger.error(f"Error during online training: {e}")
        import traceback; logger.error(traceback.format_exc())
        with model_lock: model.eval() # Ensure model is back in eval mode on failure
        return False

def trigger_retraining():
    """Starts the retraining process in a new thread."""
    global retraining_thread
    if retraining_thread and retraining_thread.is_alive():
        logger.info("Retraining already in progress. Skipping trigger.")
        return
    
    logger.info("Threshold reached. Triggering online model retraining.")
    retraining_thread = threading.Thread(target=run_online_training)
    retraining_thread.daemon = True
    retraining_thread.start()

# --- Main Server Logic (from original) ---

def start_server():
    try:
        server = HTTPServer((SERVER_HOST, SERVER_PORT), PredictionHandler)
        logger.info("=" * 60)
        logger.info("Trading Daemon with Online Learning - Started")
        logger.info("=" * 60)
        logger.info(f"Server URL: http://{SERVER_HOST}:{SERVER_PORT}")
        logger.info("Available endpoints:")
        logger.info(f"  POST /predict  - Make predictions")
        logger.info(f"  POST /feedback - Submit trade outcomes for online learning")
        logger.info(f"  GET  /health   - Health check")
        logger.info(f"  GET  /stats    - Server statistics")
        logger.info("=" * 60)
        
        server.serve_forever()
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        sys.exit(1)

def main():
    logger.info("=" * 60)
    logger.info("LSTM Trading Daemon - HTTP Server Version")
    logger.info("=" * 60)
    
    if not verify_paths_and_permissions():
        sys.exit(1)
    
    if not load_models_and_scalers():
        sys.exit(1)
    
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        logger.info("Test mode not applicable for online learning daemon. Starting normally.")
        # Note: The original test_server function would need to be adapted if still needed.
    
    start_server()

if __name__ == "__main__":
    main()