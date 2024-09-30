import argparse
import multiprocessing
import random
import re
from typing import List, NamedTuple, Optional, Tuple

import pandas as pd
from antlr4 import *
from make_data.java.java8.JavaLexer import JavaLexer
from make_data.java.java8.JavaParser import JavaParser
from make_data.java.java8.JavaParserListener import JavaParserListener
from tqdm import tqdm


class ASample(NamedTuple):
    class_name: str
    func_name: str
    masked_class: str
    func_body: str


class Location(NamedTuple):
    start_line: int
    start_col: int
    end_line: int
    end_col: int


class Function(NamedTuple):
    class_name: str
    class_loc: Location
    func_name: str
    func_body_loc: Location


class ExtractFunc(JavaParserListener):
    def __init__(self, java_code):
        super().__init__()
        self.functions = []
        self.classes = []
        self.java_code = java_code

    def enterClassDeclaration(self, ctx):
        class_name = ctx.identifier().getText()
        class_loc = Location(
            ctx.start.line,
            ctx.start.column,
            ctx.stop.line,
            ctx.stop.column + len(ctx.stop.text),
        )
        self.classes.append({"class_name": class_name, "class_loc": class_loc})

    def enterMethodDeclaration(self, ctx):
        body = ctx.methodBody().block()
        if not body:
            return
        func_name = ctx.identifier().getText()
        func_body_loc = Location(
            body.start.line,
            body.start.column,
            body.stop.line,
            body.stop.column + len(body.stop.text),
        )
        try:
            self.functions.append(
                {
                    "func_name": func_name,
                    "func_body_loc": func_body_loc,
                }
            )
        except Exception:
            pass

    def get_functions(self):
        for i in range(len(self.functions)):
            func_start_idx, func_end_idx = get_location(
                self.java_code, self.functions[i]["func_body_loc"]
            )
            for cl in self.classes:
                class_start_idx, class_end_idx = get_location(
                    self.java_code, cl["class_loc"]
                )
                if (
                    class_start_idx < func_start_idx
                    and class_end_idx > func_end_idx
                ):
                    self.functions[i]["class_name"] = cl["class_name"]
                    self.functions[i]["class_loc"] = cl["class_loc"]
                    break
            else:
                print("Something go wrong")
        return self.functions


def get_location(java_code: str, loc: Location) -> Tuple[int, int]:
    lines = java_code.split("\n")
    start_idx = 0
    for i in range(loc.start_line - 1):
        start_idx += len(lines[i])
    start_idx = start_idx + loc.start_col + loc.start_line - 1

    end_idx = 0
    for i in range(loc.end_line - 1):
        end_idx += len(lines[i])
    end_idx = end_idx + loc.end_col + loc.end_line - 1
    return start_idx, end_idx


def get_functions(java_code: str) -> Optional[List[Function]]:
    try:
        input_stream = InputStream(java_code)
        lexer = JavaLexer(input_stream)
        token_stream = CommonTokenStream(lexer)
        parser = JavaParser(token_stream)
        tree = parser.compilationUnit()
        # Create listener
        listener = ExtractFunc(java_code)
        # Walk the parse tree
        walker = ParseTreeWalker()
        walker.walk(listener, tree)
        functions = listener.get_functions()
    except Exception:
        return None
    return functions


def mask_function(java_code: str) -> Optional[ASample]:
    functions = get_functions(java_code)
    if not functions:
        return None
    # Randomly select a function
    random_function = random.choice(functions)

    # Extract function body
    class_start_idx, class_end_idx = get_location(
        java_code, random_function.class_loc
    )
    func_body_start_idx, func_body_end_idx = get_location(
        java_code, random_function.func_body_loc
    )
    masked_class = (
        java_code[class_start_idx : func_body_start_idx + 1]
        + "<FILL_FUNCTION_BODY>"
        + java_code[func_body_end_idx - 1 : class_end_idx]
    )
    func_body = java_code[func_body_start_idx + 1 : func_body_end_idx - 1]

    return ASample(
        class_name=random_function.class_name,
        func_name=random_function.func_name,
        masked_class=masked_class,
        func_body=func_body,
    )


# def modified_mask_function(java_code: str) -> Optional[List[ASample]]:
#     functions = get_functions(java_code)
#     if not functions:
#         return None

#     result = []
#     for function in functions:
#         # Extract function body
#         class_start_idx, class_end_idx = get_location(
#             java_code, function["class_loc"]
#         )
#         func_body_start_idx, func_body_end_idx = get_location(
#             java_code, function["func_body_loc"]
#         )
#         masked_class = (
#             java_code[class_start_idx : func_body_start_idx + 1]
#             + "<FILL_FUNCTION_BODY>"
#             + java_code[func_body_end_idx - 1 : class_end_idx]
#         )
#         func_body = java_code[func_body_start_idx + 1 : func_body_end_idx - 1]

#         result.append(
#             ASample(
#                 class_name=function["class_name"],
#                 func_name=function["func_name"],
#                 masked_class=masked_class,
#                 func_body=func_body,
#             )
#         )
#     return result


def modified_mask_function(
    java_code: str, expected_func_name, expected_func_body: str
) -> Optional[List[ASample]]:
    functions = get_functions(java_code)
    if not functions:
        return None

    result = []
    for function in functions:
        # Extract function body
        class_start_idx, class_end_idx = get_location(
            java_code, function["class_loc"]
        )
        func_body_start_idx, func_body_end_idx = get_location(
            java_code, function["func_body_loc"]
        )
        masked_class = (
            java_code[class_start_idx : func_body_start_idx + 1]
            + "<FILL_FUNCTION_BODY>"
            + java_code[func_body_end_idx - 1 : class_end_idx]
        )
        func_body = java_code[func_body_start_idx + 1 : func_body_end_idx - 1]

        if function["func_name"] == expected_func_name:
            if " ".join(func_body.split()) == " ".join(
                expected_func_body.split()
            ):
                result.append(
                    ASample(
                        class_name=function["class_name"],
                        func_name=function["func_name"],
                        masked_class=masked_class,
                        func_body=func_body,
                    )
                )
            else:
                print(repr(func_body))
                print("-" * 100)
                print(repr(expected_func_body))

    return result


def make_samples(argument: Tuple[str, str, str]):
    # java_file_url, project_name, relative_path = argument
    # Change to debug
    java_file_url, project_name, relative_path, func_name, func_body = argument

    with open(java_file_url, "r", encoding="utf-8", errors="ignore") as f:
        try:
            java_code = f.read()
            # samples = modified_mask_function(java_code)
            # Change to debug
            samples = modified_mask_function(java_code, func_name, func_body)
            if samples:
                return [
                    {
                        "proj_name": project_name,
                        "relative_path": relative_path,
                        "class_name": sample.class_name,
                        "func_name": sample.func_name,
                        "masked_class": sample.masked_class,
                        "func_body": sample.func_body,
                    }
                    for sample in samples
                ]
            else:
                return [
                    {
                        "proj_name": project_name,
                        "relative_path": relative_path,
                        "class_name": None,
                        "func_name": None,
                        "masked_class": None,
                        "func_body": None,
                    }
                ]
        except Exception:
            return [
                {
                    "proj_name": None,
                    "relative_path": None,
                    "class_name": None,
                    "func_name": None,
                    "masked_class": None,
                    "func_body": None,
                }
            ]


def make_dataset(
    java_files: pd.DataFrame, repos_directory: str, num_process: int = 10
) -> pd.DataFrame:
    """Make dataset

    Args:
        java_file_urls (str): java file urls
        repos_directory (str): path to diretory of repositories
        num_process (int): number of concurrent processes. Default: 10.

    Returns:
        pd.DataFrame: Dataset
    """
    iteration = len(java_files)
    java_files["absolute_url"] = java_files.apply(
        lambda row: f"{repos_directory}/{row['proj_name']}/{row['relative_path']}",
        axis=1,
    )

    arguments = list(
        zip(
            java_files["absolute_url"],
            java_files["proj_name"],
            java_files["relative_path"],
            # Add for debugging
            java_files["func_name"],
            java_files["func_body"],
        )
    )
    with multiprocessing.Pool(processes=num_process) as pool:
        rows = list(
            tqdm(
                pool.imap(make_samples, arguments),
                total=iteration,
                desc="Making data",
            )
        )
    flatten_rows = []
    for item in rows:
        flatten_rows.extend(item)
    return pd.DataFrame(flatten_rows)


def post_processing(dataset: pd.DataFrame) -> pd.DataFrame:
    def std(x):
        return re.sub(r"[\t\n\r ]", "", x)

    dataset["std_func_body"] = dataset["func_body"].apply(std)
    dataset = dataset[dataset["std_func_body"] != ""]
    dataset.drop(columns=["std_func_body"], inplace=True)
    dataset.reset_index(drop=True, inplace=True)
    return dataset


def main(args):
    java_files = pd.read_parquet(args.input)
    java_files.reset_index(drop=True, inplace=True)
    df = make_dataset(
        java_files=java_files,
        repos_directory=args.dir,
        num_process=int(args.proc),
    )
    df.to_parquet(args.output, "fastparquet")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", dest="input")
    parser.add_argument("--dir", dest="dir")
    parser.add_argument("--output", dest="output")
    parser.add_argument("--proc", dest="proc")
    args = parser.parse_args()
    main(args)
