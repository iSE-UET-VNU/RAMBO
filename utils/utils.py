# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import glob
import pickle
import json
import torch
import tiktoken
from transformers import AutoTokenizer
from utils.unixcoder import UniXcoder

class FilePathBuilder:
    @staticmethod
    def make_needed_dir(file_path):
        dir_path = os.path.dirname(file_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    
    @staticmethod
    def repo_methods_path(repo, benchmark):
        out_path = os.path.join(f'cache/{benchmark}/window/methods', f'{repo}_methods.pkl')
        FilePathBuilder.make_needed_dir(out_path)
        return out_path
    
    @staticmethod
    def repo_types_path(repo, benchmark):
        out_path = os.path.join(f'cache/{benchmark}/window/types', f'{repo}_types.pkl')
        FilePathBuilder.make_needed_dir(out_path)
        return out_path

    @staticmethod
    def repo_windows_path(repo, benchmark, window_size, slice_size):
        out_path = os.path.join(f'cache/{benchmark}/window/repos', f'{repo}_ws{window_size}_slice{slice_size}.pkl')
        FilePathBuilder.make_needed_dir(out_path)
        return out_path

    @staticmethod
    def bow_vector_path(window_file):
        vector_path = window_file.replace('/window/', '/vector/')
        out_path = vector_path.replace('.pkl', '.bow.pkl')
        FilePathBuilder.make_needed_dir(out_path)
        return out_path

    @staticmethod
    def UniXcoder_vector_path(window_file):
        vector_path = window_file.replace('/window/', '/vector/')
        out_path = vector_path.replace('.pkl', '.unixcoder.pkl')
        FilePathBuilder.make_needed_dir(out_path)
        return out_path


class CodexTokenizer:
    def __init__(self):
        self.tokenizer = tiktoken.get_encoding("p50k_base")
    
    def tokenize(self, text):
        # return self.tokenizer.encode(text)
        return self.tokenizer.encode_ordinary(text)

    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids)

class CodeGenTokenizer:
    def __init__(self):
        # self.tokenizer = AutoTokenizer.from_pretrained('Salesforce/codegen-6B-mono')
        self.tokenizer = AutoTokenizer.from_pretrained('deepseek-ai/deepseek-coder-6.7b-base')
        
    def tokenize(self, text):
        return self.tokenizer.encode(text)

    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids)

class Tools:
    @staticmethod
    def read_code(fname):
        with open(fname, 'r', encoding='utf8') as f:
            return f.read()
    
    @staticmethod
    def load_pickle(fname):
        with open(fname, 'rb') as f:
            return pickle.load(f)
    
    @staticmethod
    def dump_pickle(obj, fname):
        with open(fname, 'wb') as f:
            pickle.dump(obj, f)
    
    @staticmethod
    def dump_json(obj, fname):
        with open(fname, 'w', encoding='utf8') as f:
            json.dump(obj, f)

    @staticmethod
    def dump_jsonl(obj, fname):
        with open(fname, 'w', encoding='utf8') as f:
            for item in obj:
                f.write(json.dumps(item) + '\n')
    
    @staticmethod
    def load_jsonl(fname):
        with open(fname, 'r', encoding='utf8') as f:
            lines = []
            for line in f:
                lines.append(json.loads(line))
            return lines
    
    @staticmethod
    def iterate_repository(repo):
        base_dir = FilePathBuilder.repo_base_dir
        pattern = os.path.join(f'{base_dir}/{repo}', "**", "*.java")
        files = glob.glob(pattern, recursive=True)
        skipped_files = []
        loaded_code_files = dict()
        base_dir_list = os.path.normpath(base_dir).split(os.sep)
        for fname in files:
            try:
                code = Tools.read_code(fname)
                fpath_tuple = tuple(os.path.normpath(fname).split(os.sep)[len(base_dir_list):])
                loaded_code_files[fpath_tuple]= code
            except Exception as e:
                skipped_files.append((fname, e))
                continue

        if len(skipped_files) > 0:
            print(f"Skipped {len(skipped_files)} out of {len(files)} files due to I/O errors")
            for fname, e in skipped_files:
                print(f"{fname}: {e}")
        return loaded_code_files

    @staticmethod
    def tokenize(code):
        tokenizer = CodexTokenizer()
        return tokenizer.tokenize(code)
    

    def UniXcoder_tokenize(code):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        emb_model = UniXcoder("microsoft/unixcoder-base")
        emb_model.to(device)
        tokens_ids = emb_model.tokenize([code], mode="<encoder-only>")
        source_ids = torch.tensor(tokens_ids).to(device)
        tokens_embeddings, code_embedding = emb_model(source_ids)
        return code_embedding.cpu().detach()
    
    @staticmethod
    def build_signature(return_type, name, parameters):
        def params_to_str(params):
            list_params_type = []
            for param in params:
                list_params_type.append(param['type'])
            
            return ' '.join(list_params_type)
        return f'{return_type} {name} {params_to_str(parameters)}'

    @staticmethod
    def is_finding_method(task, method):
        try:
            if tuple(method['metadata']['fpath_tuple']) == tuple(task['metadata']['fpath_tuple']) and \
            method['metadata']['class'] == task['metadata']['class_name']:
                return True
        except:
            return False
        return False
    
    @staticmethod
    def get_lcontext_method(method, repo_base_dir, MAX_NUMBER_LINES=10):
        try:
            fpath_list = method['fpath_tuple']
            relative_path = os.path.join(repo_base_dir, os.path.join(*fpath_list))
            with open(relative_path, 'r', encoding='utf8') as f:
                file_content = f.read()
            source_code = file_content.replace("\r\n", "\n")
            cur_number_line = 0
            start_pos = method['start_line_no']
            end_pos = method['end_line_no']
            for i in range(method['start_line_no'] - 1, 0, -1):
                start_pos = i
                if source_code[i] == '\n':
                    cur_number_line += 1
                    if (cur_number_line > MAX_NUMBER_LINES):
                        start_pos = i + 1
                        break
            return source_code[start_pos: end_pos]
        except:
            return method['method']
    
    @staticmethod
    def clean_output(output):
        cur_bracket = 0
        for idx, c in enumerate(output):
            if c == '{':
                cur_bracket += 1
            elif c == '}':
                cur_bracket -= 1
            
            if cur_bracket < 0:
                return output[:idx]
        
        return output