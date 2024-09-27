from utils.utils import Tools, FilePathBuilder
from utils.vector_utils import BagOfWordsEmbedding, UniXcoderEmbedding
from utils.rerank_utils import SingleReranking
device = "cpu"

class SketchPromptBuilder:
    def __init__(self, benchmark, reranker_type, repos, tasks, tokenizer):
        self.benchmark = benchmark
        if reranker_type == 'bow':
            self.vector_builder = BagOfWordsEmbedding()
        elif reranker_type == 'unixcoder':
            self.vector_builder = UniXcoderEmbedding(device=device)
        else:
            raise NotImplementedError
        self.reranker = SingleReranking(self.vector_builder)
        self.repos = repos
        self.tasks = tasks
        self.tokenizer = tokenizer
    
    def build_prompt(self, output_file):
        new_prompt_lines = []
        task_cnt = 0
        for repo in self.repos:
            repo_method_input_file = FilePathBuilder.repo_methods_path(repo, self.benchmark)
            repo_embedding_methods_file =  self.vector_builder.vector_file_path(repo_method_input_file)
            repo_embedding_methods = Tools.load_pickle(repo_embedding_methods_file)
            for task in self.tasks:
                if task['metadata']['task_id'].split('/')[0] == repo:
                    task_cnt += 1
                    print('Re-ranking for task: ', task['metadata']['task_id'], 'Task count: ', task_cnt)
                    task_signature = Tools.build_signature(
                        task['metadata']['return_type'],
                        task['metadata']['function_name'],
                        task['metadata']['parameters']
                    )
                    doc_embeddings = []
                    docs = []
                    
                    for line in repo_embedding_methods:
                        if not Tools.is_finding_method(task, line):
                            docs.append(line)
                            doc_embeddings.append(line['data'][0]['signature_embedding'])
                    top_k_signatures = self.reranker.rerank_batch(task_signature, docs, doc_embeddings)
                    
                    prepend_context = "// Here are some relevant code fragments from other files of the repo:\n"
                    seperator = '// ' + '-' * 50
                    prepend_context += seperator + '\n'
                    
                    len_current_token = self.tokenizer(prepend_context + task['metadata']['left_context'] + '<FILL_FUNCTION_BODY>' + task['metadata']['right_context'] + task['metadata']['ground_truth'], return_tensors="pt").to(device)['input_ids'].size()[1]
                    
                    print('Query: ', task_signature)
                    top_method_query = top_k_signatures[0][0]
                    print('Top method signature: ', top_method_query['metadata']['signature'])
                    print('Score: ', top_k_signatures[0][1])
                    
                    chosen_context = []
                    for top_k_signature in top_k_signatures:
                        content = top_k_signature[0]
                        content_lines = content['content'].split('\n')
                        content_lines_comment = [f'// {line}' for line in content_lines]
                        block_str = '\n'.join(content_lines_comment + [seperator]) + '\n'
                        block_token_len = self.tokenizer(block_str, return_tensors="pt").to(device)['input_ids'].size()[1]
                        
                        if len_current_token + block_token_len > 8000:
                            break
                        prepend_context += block_str
                        len_current_token += block_token_len
                        chosen_context.append((content['metadata']['fpath_tuple'], content['content'], top_k_signature[1], content['metadata']['signature'], content['metadata']['start_line_no'], content['metadata']['end_line_no']))
                    prepend_context += """// Based on above, complete the method body of the class\n"""
                    if len(chosen_context) > 0: 
                        prompt = prepend_context + '\n' + task['metadata']['left_context'] + '<FILL_FUNCTION_BODY>' + task['metadata']['right_context']
                    else:
                        prompt = task['metadata']['left_context'] + '<FILL_FUNCTION_BODY>' + task['metadata']['right_context'] 
                    new_prompt_line = {
                        'prompt': prompt,
                        'metadata': task['metadata'],
                    }
                    
                    new_prompt_line['metadata']['top_k_context'] = [
                        {
                            'fpath': x[0],
                            'method': x[1],
                            'sim_score': x[2],
                            'signature': x[3],
                            'start_line_no': x[4],
                            'end_line_no': x[5],
                        } for x in chosen_context
                    ]
                    
                    new_prompt_lines.append(new_prompt_line)
                
        Tools.dump_jsonl(new_prompt_lines, output_file)