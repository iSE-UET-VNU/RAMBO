import argparse

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer

model_name = "deepseek-ai/deepseek-coder-6.7b-base"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


def count_token(df: pd.DataFrame) -> pd.DataFrame:
    """Count and set limit number of tokens for samples

    Args:
        df (pd.DataFrame): Input dataset

    Returns:
        pd.DataFrame: Output dataset, after filtered long samples
    """
    input_ls = []
    output_ls = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Counting length"):
        model_input = tokenizer(row["masked_class"])
        input_ls.append(len(model_input["input_ids"]))
        model_input = tokenizer(row["func_body"])
        output_ls.append(len(model_input["input_ids"]))

    df["len_input"] = input_ls
    df["len_output"] = output_ls
    df["total"] = df["len_input"] + df["len_output"]

    return df


def main(args):
    df = pd.read_parquet(args.input)
    new_df = count_token(df)
    print(new_df.info())
    print(new_df.head())
    print("-" * 100)
    new_df["len_input"].hist(bins=10, ec="black")
    new_df["len_output"].hist(bins=10, ec="black")
    new_df["total"].hist(bins=10, ec="black")
    plt.show()
    new_df.to_parquet(args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", dest="input")
    parser.add_argument("--output", dest="output")
    args = parser.parse_args()
    main(args)
