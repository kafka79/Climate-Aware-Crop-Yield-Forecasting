import subprocess
from loguru import logger
from src.utils.config import load_config
from src.cli.parser import get_parser
from src.inference.runtime import InferenceUnavailableError

def main(args):
    """
    Main entry point for the Crop Yield Prediction Pipeline.
    """
    logger.info("Initializing Crop Yield Prediction Pipeline...")
    
    # Load configuration
    config = load_config(args.config)
    logger.info(f"Loaded config from {args.config}")

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
    
    elif args.mode == "predict":
        logger.info("Starting Inference and Agronomy Advice Phase...")
        from src.inference.runtime import run_inference

        try:
            result = run_inference(
                region=args.region,
                year=args.year,
                crop=args.crop,
                config_path=args.config,
            )
        except InferenceUnavailableError as exc:
            logger.error(str(exc))
            raise SystemExit(1) from exc

        print("\n" + "="*50)
        print(f"REGION: {result['region']} | YEAR: {result['year']}")
        print(f"PREDICTED YIELD: {result['predicted_yield']:.2f} t/ha")
        print(
            "CONFIDENCE INTERVAL (95%): "
            f"[{result['lower_bound']:.2f} - {result['upper_bound']:.2f}] t/ha"
        )
        print(f"RISK ASSESSMENT: {result['risk']}")
        if result["historical_average"] is not None:
            print(f"HISTORICAL AVERAGE: {result['historical_average']:.2f} t/ha")
        if result["observed_yield"] is not None:
            print(f"OBSERVED YIELD FOR {result['year']}: {result['observed_yield']:.2f} t/ha")
        print("-" * 50)
        print("MODEL ATTRIBUTION BY MODALITY:")
        for name, score in result["attribution"].items():
            print(f"- {name}: {score:.4f}")
        print(f"SOIL INPUT SOURCE: {result['soil_source']}")
        print("-" * 50)
        print("AGRONOMIC ADVICE & RECOMMENDATIONS:")
        for advice in result.get("recommendations", []):
            print(f"- {advice}")
        print("="*50 + "\n")

    elif args.mode == "benchmark":
        logger.info("Starting Model Benchmarking...")
        from src.training.train import run_benchmark_pipeline
        run_benchmark_pipeline(args.config)

    elif args.mode == "dashboard":
        logger.info("Launching Streamlit Dashboard...")
        subprocess.run(["streamlit", "run", "app.py"])

    else:
        logger.error(f"Unknown mode: {args.mode}")

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
