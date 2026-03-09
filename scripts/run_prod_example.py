#!/usr/bin/env python3
"""
Simple example: Run final_metric_refactor on prod_data_sample.csv

This script demonstrates how to:
1. Preprocess prod_data_sample.csv (extract user_context from input)
2. Run final_metric_refactor on the processed data
"""

from pathlib import Path
from preprocess_and_run_prod_data import run_final_metric_on_prod_data

if __name__ == "__main__":
    # Path to prod_data_sample.csv
    prod_csv = Path(__file__).resolve().parent.parent / "data" / "prod_data_sample.csv"

    # Example 1: Run with all data in integrated mode
    print("Running final_metric_refactor on prod_data_sample.csv...")
    run_final_metric_on_prod_data(
        source_csv=prod_csv,
        max_rows=0,  # Use all data
        inspection_mode="integrated",
    )

    # Example 2: Run with limited rows in detailed mode
    # Uncomment to run:
    # run_final_metric_on_prod_data(
    #     source_csv=prod_csv,
    #     max_rows=100,
    #     inspection_mode="detailed",
    # )
