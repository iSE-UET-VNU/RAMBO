import json
import os
from tqdm import tqdm
from utils.utils import Tools, FilePathBuilder


class ExtractRepoElements:
    def __init__(self, benchmark, repo_base_dir, parsed_repo_base_dir, repos):
        self.benchmark = benchmark
        self.repo_base_dir = repo_base_dir
        self.parsed_repo_base_dir = parsed_repo_base_dir
        self.repos = repos
    
    def _extract_context_for_a_method(self, method):
        method_context = {}
        error = None
        fpath = os.path.join(self.repo_base_dir, method["relative_path"])
        base_dir_list = os.path.normpath(self.repo_base_dir).split(os.sep)
        try:
            with open(fpath, "r", encoding="utf8") as f:
                file_content = f.read()
        except Exception as e:
            error = (fpath, e)
            return None, error
        file_code = file_content.replace("\r\n", "\n")
        method_content = file_code[method["start"] : method["end"]]
        method_fpath_tuple = tuple(
            os.path.normpath(fpath).split(os.sep)[len(base_dir_list) :]
        )
        method_context = {
            "content": method_content,
            "metadata": {
                "fpath_tuple": method_fpath_tuple,
                "name": method["name"],
                "class": method["class"],
                "start_line_no": method["start"],
                "end_line_no": method["end"],
                "return_type": method["return_type"],
                "parameters": method["parameters"],
                "body_raw": method["body_raw"],
                # 'modifiers': method['modifiers'],
            },
        }
        return method_context, error

    def _extract_methods_for_a_repo(self, parsed_repo_methods):
        all_method_code_contexts = []
        skipped_methods = []
        for method in parsed_repo_methods:
            method_context, error = self._extract_context_for_a_method(method)
            if error:
                skipped_methods.append(error)
            else:
                all_method_code_contexts.append(method_context)
        
        if len(skipped_methods) > 0:
            print(
                f"Skipped {len(skipped_methods)} out of {len(parsed_repo_methods)} files due to I/O errors"
            )
            for fname, e in skipped_methods:
                print(f"{fname}: {e}")
        return all_method_code_contexts
        
    def _extract_types_for_a_repo(self, parsed_repo_methods):
        types_list = {}
        for method in parsed_repo_methods:
            key = (method["class"], method["relative_path"])
            method_name = method["name"]
            if key not in types_list:
                types_list[key] = []
            types_list[key].append(method_name)
        
        return types_list
    
    def _extract_elements_for_a_repo(self, repo):
        parsed_repo_methods_dir = os.path.join(
            self.parsed_repo_base_dir, f"{repo}_methods.json"
        )
        with open(parsed_repo_methods_dir, "r", encoding="utf8") as f:
            parsed_repo_methods = json.loads(f.read())
        all_method_code_contexts = self._extract_methods_for_a_repo(parsed_repo_methods)
        types_list = self._extract_types_for_a_repo(parsed_repo_methods)
        print(f"built {len(all_method_code_contexts)} methods for {repo}")
        print(f"built {len(types_list)} types for {repo}")
        return all_method_code_contexts, types_list

    def extract_elements(self):
        for repo in tqdm(self.repos):
            all_method_code_contexts, types_list = self._extract_elements_for_a_repo(repo)
            Tools.dump_pickle(
                all_method_code_contexts, FilePathBuilder.repo_methods_path(repo, self.benchmark)
            )
            Tools.dump_pickle(types_list, FilePathBuilder.repo_types_path(repo, self.benchmark))