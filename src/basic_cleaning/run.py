#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning, exporting the result to a new artifact
"""
import argparse
import logging
import wandb
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    # artifact_local_path = run.use_artifact(args.input_artifact).file()

    ######################
    # YOUR CODE HERE     #
    ######################
    logger.info("Downloading input artifact")
    artifact_local_path = run.use_artifact(args.input_artifact).file()
    df = pd.read_csv(artifact_local_path)

    logger.info("Cleaning data")
    # Remove outliers in price
    min_price = args.min_price
    max_price = args.max_price
    idx = df['price'].between(min_price, max_price)
    df = df[idx].copy()
    
    # Convert last_review to datetime, only if the column exists
    if 'last_review' in df.columns:
        df['last_review'] = pd.to_datetime(df['last_review'])

    # Save the cleaned data
    logger.info("Saving cleaned data")
    df.to_csv("clean_sample.csv", index=False)

    logger.info("Saving cleaned data to wandb")
    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description
    )
    artifact.add_file("clean_sample.csv")
    run.log_artifact(artifact)
    logger.info("Cleaned data saved to wandb")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A very basic data cleaning")


    parser.add_argument(
        "--input_artifact", 
        type=str,
        help="Name of the input artifact in wandb in which we load the raw data (csv file called sample.csv)",
        required=True
    )

    parser.add_argument(
        "--output_artifact", 
        type=str,
        help="Name of output artifact in wandb in which we save the cleaned data",
        required=True
    )

    parser.add_argument(
        "--output_type", 
        type=str,
        help="Type of the output artifact in wandb (csv file)",
        required=True
    )

    parser.add_argument(
        "--output_description", 
        type=str,
        help="Save the cleaned data in csv format called clean_sample.csv",
        required=True
    )

    parser.add_argument(
        "--min_price", 
        type=float,
        help="Minimum price to consider when we are cleaning the data & remove outliers",
        required=True
    )

    parser.add_argument(
        "--max_price", 
        type=float,
        help="Maximum price to consider when we are cleaning the data & remove outliers",
        required=True
    )


    args = parser.parse_args()

    go(args)
