import argparse

def get_parser():
    """
    Defines the CLI argument parser for the Climate-Aware Crop Yield pipeline.
    """
    parser = argparse.ArgumentParser(description="Climate-Aware Crop Yield Forecasting Pipeline")
    
    parser.add_argument("--mode", type=str, required=True, 
                        choices=["download", "preprocess", "train", "predict", "benchmark", "dashboard"],
                        help="Execution mode")
    
    parser.add_argument("--config", type=str, default="configs/data_config.yaml",
                        help="Path to data configuration file")
    
    parser.add_argument("--region", type=str, default="Burdwan, West Bengal",
                        help="Target study area for download/prediction. Must match configs/data_config.yaml.")
    
    parser.add_argument("--year", type=int, default=2023,
                        help="Target year for data download")
    
    parser.add_argument("--crop", type=str, default="Rice",
                        help="Target crop for download. Live inference currently uses region/year artifacts.")

    return parser
