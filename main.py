import os
import argparse
import yaml
from loguru import logger
from src.utils.config import load_config

def main(args):
    """
    Main entry point for the Crop Yield Prediction Pipeline.
    """
    logger.info("Initializing Crop Yield Prediction Pipeline...")
    
    # Load configuration
    if args.config:
        config = load_config(args.config)
        logger.info(f"Loaded config from {args.config}")
    else:
        logger.warning("No configuration file provided. Using defaults.")
        config = {}

    # Inject CLI overrides into config
    if args.region:
        config["region"] = args.region
    if args.year:
        config["year"] = args.year
        config["time_range"] = (f"{args.year}-01-01", f"{args.year}-12-31")

    if args.mode == "download":
        logger.info(f"Starting Data Download Phase for {config.get('region')} in {config.get('year')}...")
        from src.data.downloader import download_multi_modal_batch
        download_multi_modal_batch(config, config.get("region"), config.get("crop", "Rice"))
    
    elif args.mode == "preprocess":
        logger.info("Starting Preprocessing Phase...")
        from src.data.preprocessing import preprocess_all
        preprocess_all(config)
    
    elif args.mode == "train":
        logger.info("Starting Model Training Phase...")
        from src.training.train import run_training_pipeline
        run_training_pipeline(args.config)
    
    elif args.mode == "eval":
        logger.info("Starting Model Evaluation Phase...")
        from src.evaluation.evaluator import EvaluationManager
        evaluator = EvaluationManager(config)
        # Note: In a real run, a separate test_loader would be passed
        evaluator.run()
    
    elif args.mode == "deploy":
        logger.info("Starting API Deployment...")
        import uvicorn
        uvicorn.run("deployment.api.app:app", host="0.0.0.0", port=8000, reload=True)

    else:
        logger.error(f"Unknown mode: {args.mode}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crop Yield Prediction Pipeline")
    parser.add_argument("--mode", type=str, default="train", 
                        choices=["download", "preprocess", "train", "eval", "deploy"],
                        help="Pipeline phase to execute.")
    parser.add_argument("--config", type=str, default="configs/data_config.yaml",
                        help="Path to YAML configuration file.")
    parser.add_argument("--year", type=int, default=2023,
                        help="Target year for analysis.")
    parser.add_argument("--region", type=str,
                        help="Target region (e.g., 'Punjab', 'Iowa').")
    
    args = parser.parse_args()
    main(args)
