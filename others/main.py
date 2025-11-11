import logging
from datetime import datetime
import pandas as pd
from src.data_loader import load_data
#from src.preprocess import preprocess_data
#from src.model import train_model, evaluate_model

# Generate a unique log file per run
log_filename = f"outputs/logs/pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_filename),  # Save to file
        logging.StreamHandler()  # Print to console
    ]
)

def main():
    logging.info("Starting the machine learning pipeline...")

    # Step 1: Load Data
    data_PATH="D:\\download\\PD project\\datasets\\ICDAR 2013 - Gender Identification Competition Dataset"
    logging.info("Loading data...")


    dataset,dataloader = load_data(data_PATH)
    if dataset is None:
        logging.error("Failed to load data. Exiting pipeline.")
        return

    print(dataset[0])
    '''# Step 2: Data Exploration (Optional)
    logging.info(f"Dataset Shape: {df.shape}")
    logging.info(f"Missing values:\n{df.isnull().sum()}")

    # Step 3: Preprocess Data
    logging.info("Preprocessing data...")
    df_processed = preprocess_data(df)

    # Step 4: Train Model
    logging.info("Training the model...")
    model, X_test, y_test = train_model(df_processed)

    # Step 5: Evaluate Model
    logging.info("Evaluating the model...")
    evaluate_model(model, X_test, y_test)

    logging.info("Pipeline execution completed.")'''

if __name__ == "__main__":
    main()
