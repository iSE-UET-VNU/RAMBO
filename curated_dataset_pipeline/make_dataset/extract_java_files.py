import argparse
import collections
import os
import subprocess
from typing import List, Tuple

import pandas as pd


def get_java_file_urls(dir: str) -> List[str]:
    """Get all java file url in directory"""
    os.chdir(dir)
    all_files = []
    for p in os.listdir():
        cmd = f"""
        find {p} -name *.java
        """
        data = subprocess.run(cmd, shell=True, text=True, capture_output=True)
        if data.stdout:
            files = data.stdout.split("\n")
            print(f"{p}: {len(files)}")
            all_files.extend(files)
    return all_files


def filter_url(base_dir: str, java_file_urls: List[str]) -> List[str]:
    """Filter invalid java url"""
    # url not contain "test"
    java_file_urls = list(filter(lambda url: "test" not in url, java_file_urls))
    print("Length after filter 1:", len(java_file_urls))

    # url must contain "src/main/java"
    src_pattern = os.path.join("src", "main", "java")
    java_file_urls = list(
        filter(lambda url: src_pattern in url, java_file_urls)
    )
    print("Length after filter 2:", len(java_file_urls))

    # java file must have corresponding class (mean that file is compiled)
    class_pattern = os.path.join("target", "classes")

    def has_corresponding_class_file(url: str) -> bool:
        class_path = url.replace(src_pattern, class_pattern).replace(
            ".java", ".class"
        )
        return os.path.exists(f"{base_dir}/{class_path}")

    valid_java_files = [
        url for url in java_file_urls if has_corresponding_class_file(url)
    ]
    print("Length after filter 3:", len(valid_java_files))
    return valid_java_files


def construct_df(java_file_urls: List[str]) -> pd.DataFrame:
    """Dividing java file urls into fields"""

    def func(url: str) -> Tuple[str, str]:
        parts = os.path.split(url)
        project_name = parts[0]
        relative_path = os.path.join(parts[1:])
        return project_name, relative_path

    df = pd.DataFrame()
    df["java_file_urls"] = java_file_urls
    df["proj_name"], df["relative_path"] = zip(
        *df["java_file_urls"].apply(func)
    )
    print(df.info())
    print("-" * 100)
    print(df.describe())
    return df


def normalize_df(df: pd.DataFrame, max_num_sample: int) -> pd.DataFrame:
    """This function ensure no project has too many samples so that make bias in the dataset

    Args:
        df (pd.DataFrame): Input dataset
        max_num_sample (int): A threshold that all project have less or equal samples in the output dataset

    Returns:
        pd.DataFrame: Output dataset
    """
    cnt = collections.Counter(df["proj_name"].tolist())
    new = pd.DataFrame(
        {"java_file_urls": [], "proj_name": [], "relative_path": []}
    )
    for p in cnt:
        print(p)
        if cnt[p] <= max_num_sample:
            new = pd.concat([new, df[df["proj_name"] == p]], axis="index")
        else:
            tmp = df[df["proj_name"] == p]
            tmp = tmp.sample(n=max_num_sample, random_state=0)
            new = pd.concat([new, tmp], axis="index")
    new.reset_index(drop=True, inplace=True)
    return new


def main(args):
    java_file_urls = get_java_file_urls(args.dir)
    print("Total java file:", len(java_file_urls))
    java_file_urls = filter_url(args.dir, java_file_urls)
    df = construct_df(java_file_urls)
    df = normalize_df(df)
    df.to_parquet(args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", dest="dir")
    parser.add_argument("--output", dest="output")
    args = parser.parse_args()
    main(args)
