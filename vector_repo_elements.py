from tqdm import tqdm
from utils.utils import Tools, FilePathBuilder

class VectorRepoElements:
    def __init__(self, benchmark, vector_builder, repos):
        self.benchmark = benchmark
        self.vector_builder = vector_builder
        self.repos = repos

    def _build_method_vectors(self, repo_method_lines):
        # lines = lines[:1000]
        method_body_docstrings = []
        method_name_docstrings = []
        for line in repo_method_lines:
            method_body_docstrings.append(line['metadata']['body_raw'])
            method_name_docstrings.append(line['metadata']['name'])
        method_signature_docstrings = []
        for line in repo_method_lines:
            # print(line)
            method_body_docstrings.append(line['metadata']['body_raw'])
            method_signature_docstrings.append(
                Tools.build_signature(
                    line['metadata']['return_type'],
                    line['metadata']['name'],
                    line['metadata']['parameters'])
                )
        repo_embedding_methods = []
        batch_size = 64
        # with tqdm(total=len(repo_method_lines)//batch_size) as pbar:
        for idx in range(0, len(repo_method_lines), batch_size):
            start, end = idx, min(idx + batch_size, len(repo_method_lines))
            lines_batch = repo_method_lines[start:end]
            signature_docstrings_batch = method_signature_docstrings[start:end]
            for line, signature in zip(lines_batch, signature_docstrings_batch):
                repo_embedding_method = {
                    'content': line['content'],
                    'metadata': line['metadata'],
                    'data': [{
                        'signature_embedding': self.vector_builder.build(signature),
                        'body_embedding': self.vector_builder.build(line['metadata']['body_raw']),
                        'name_embedding': self.vector_builder.build(line['metadata']['name'])
                    }]
                }
                repo_embedding_method['metadata']['signature'] = signature
                repo_embedding_methods.append(repo_embedding_method)
                # pbar.update(1)
        return repo_embedding_methods

    def _build_type_vectors(self, repo_type_lines):
        repo_type_name_docstrings = []
        repo_type_methods = {}    
        for key, methods in repo_type_lines.items():
            class_name, relative_path = key
            if class_name not in repo_type_methods:
                repo_type_name_docstrings.append(class_name)
                repo_type_methods[class_name] = []
            repo_type_methods[class_name].extend(methods)
        repo_embedding_types = []
        batch_size = 64
        # with tqdm(total=len(repo_type_name_docstrings)//batch_size) as pbar:
        for idx in range(0, len(repo_type_name_docstrings), batch_size):
            start, end = idx, min(idx + batch_size, len(repo_type_name_docstrings))
            lines_batch = repo_type_name_docstrings[start:end]
            for line in lines_batch:
                repo_embedding_types.append({
                    'name': line,
                    'data': [{
                        'name_embedding': self.vector_builder.build(line)
                    }]
                })
            # pbar.update(1)
        
        return repo_embedding_types

    def vector_elements(self):
        for repo in tqdm(self.repos):
            repo_method_input_file = FilePathBuilder.repo_methods_path(repo, self.benchmark)
            # print(f'building {self.vector_builder} vector for {repo_method_input_file}')
            repo_method_lines = Tools.load_pickle(repo_method_input_file)
            repo_embedding_methods = self._build_method_vectors(repo_method_lines)
            repo_type_input_file = FilePathBuilder.repo_types_path(repo, self.benchmark)
            # print(f'building {self.vector_builder} vector for {repo_type_input_file}')
            repo_type_lines = Tools.load_pickle(repo_type_input_file)
            repo_embedding_types = self._build_type_vectors(repo_type_lines)
            Tools.dump_pickle(repo_embedding_methods, self.vector_builder.vector_file_path(repo_method_input_file))
            Tools.dump_pickle(repo_embedding_types, self.vector_builder.vector_file_path(repo_type_input_file))
        
    