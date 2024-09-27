import time
from utils.utils import Tools, FilePathBuilder
from utils.vector_utils import BagOfWordsEmbedding, UniXcoderEmbedding
from utils.rerank_utils import SingleReranking
device = "cpu"

class RUEPromptBuilder:
    def __init__(self, benchmark, repo_base_dir, reranker_type, repos, tasks, tokenizer):
        self.benchmark = benchmark
        self.repo_base_dir = repo_base_dir
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
        start_time = time.time()
        max_prompt_token = 0
        for repo in self.repos:
            repo_method_input_file = FilePathBuilder.repo_methods_path(repo, self.benchmark)
            repo_embedding_methods_file =  self.vector_builder.vector_file_path(repo_method_input_file)
            repo_embedding_methods = Tools.load_pickle(repo_embedding_methods_file)
            repo_type_input_file = FilePathBuilder.repo_types_path(repo, self.benchmark)
            repo_embedding_types_file =  self.vector_builder.vector_file_path(repo_type_input_file)
            repo_embedding_types = Tools.load_pickle(repo_embedding_types_file)
            
            repo_type_lines = Tools.load_pickle(repo_type_input_file)
            repo_type_name_docstrings = []
            
            repo_type_methods = {}    
            for key, methods in repo_type_lines.items():
                class_name, relative_path = key
                if class_name not in repo_type_methods:
                    repo_type_name_docstrings.append(class_name)
                    repo_type_methods[class_name] = []
                repo_type_methods[class_name].extend(methods)
            
            repo_method_name_docstrings = []
            for line in repo_embedding_methods:
                repo_method_name_docstrings.append(line['metadata']['name'])
            
            batch_size = 64
            for task in self.tasks:
                print(task['metadata']['task_id'])
                if task['metadata']['task_id'].split('/')[0] == repo:
                    task_cnt += 1
                    print('Re-ranking for task: ', task['metadata']['task_id'], 'Task count: ', task_cnt)
                    score_candidate_methods = []
                    for top_k_signature in task['metadata']['top_k_context']:
                        score_candidate_methods.append({
                            'fpath_tuple': top_k_signature['fpath_tuple'],
                            'method': top_k_signature['method'],
                            'sim_score': top_k_signature['sim_score'],
                            'start_line_no': top_k_signature['start_line_no'],
                            'end_line_no': top_k_signature['end_line_no'],
                            'type_query': 'signature'
                        })
                    prediction_samples = task['choices']
                    clean_prediction = Tools.clean_output(prediction_samples[0]['text'])
                    extracted_types = task['prediction.types']
                    extracted_methods = task['prediction.methods']
                    # clean_prediction = Tools.clean_output(prediction_samples)
                    # extracted_types = task['ground_truth.types']
                    # extracted_methods = task['ground_truth.methods']           

                    print('Start reranking for extracted types for task: ', task['metadata']['task_id'])
                    candidate_method_names = []
                    # new_extracted_types = []
                    for extracted_type in extracted_types:                      
                        doc_embeddings = []
                        docs = []
                        for line in repo_embedding_types:
                            docs.append(line)
                            doc_embeddings.append(line['data'][0]['name_embedding'])
                        top_k_extracted_types = self.reranker.rerank_batch(extracted_type, docs, doc_embeddings, 1)
                        max_score_retreived_type = top_k_extracted_types[0][0]['name']
                        print(extracted_type, max_score_retreived_type, top_k_extracted_types[0][1])
                        candidate_method_names.extend(repo_type_methods[max_score_retreived_type])
                    
                    print('Start reranking for extracted methods for task: ', task['metadata']['task_id'])
                    for extracted_method in extracted_methods:                     
                        doc_embeddings = []
                        docs = []
                        for line in repo_embedding_methods:
                            docs.append(line)
                            doc_embeddings.append(line['data'][0]['name_embedding'])
                        top_k_extracted_methods = self.reranker.rerank_batch(extracted_method, docs, doc_embeddings, 1)
                        max_score_retreived_method = top_k_extracted_methods[0][0]['metadata']['name']
                        print(extracted_method, max_score_retreived_method, top_k_extracted_methods[0][1])
                        candidate_method_names.append(max_score_retreived_method)
                    
                    candidate_methods = []
                    print('Start retrieve candidate methods for task: ', task['metadata']['task_id'])
                    print('Number of repo methods: ', len(repo_embedding_methods))
                    print('Number of candidate method names: ', len(candidate_method_names))
                    for repo_method in repo_embedding_methods:
                        if not Tools.is_finding_method(task, repo_method):
                            for candidate_method_name in candidate_method_names:
                                if candidate_method_name in repo_method['metadata']['body_raw']:
                                    candidate_methods.append(repo_method)
                                    break
                    
                    print('Number of candidate methods: ', len(candidate_methods))
                    if len(candidate_methods) > 0:
                        doc_embeddings = []
                        docs = []
                        for line in candidate_methods:
                            if not Tools.is_finding_method(task, line):
                                docs.append(line)
                                doc_embeddings.append(line['data'][0]['body_embedding'])
                        top_k_draft_methods = self.reranker.rerank_batch(clean_prediction, docs, doc_embeddings, 20)
                        print('Query: ', clean_prediction)
                        top_method_query = top_k_draft_methods[0][0]
                        print('Top similar draft method: ', top_method_query['metadata']['body_raw'])
                        print('Score: ', top_k_draft_methods[0][1])
                        
                        for top_k_draft_method in top_k_draft_methods:
                            score_candidate_methods.append({
                                'fpath_tuple': top_k_draft_method[0]['metadata']['fpath_tuple'],
                                'method': top_k_draft_method[0]['metadata']['name'],
                                'sim_score': top_k_draft_method[1],
                                'start_line_no': top_k_draft_method[0]['metadata']['start_line_no'],
                                'end_line_no': top_k_draft_method[0]['metadata']['end_line_no'],
                                'type_query': 'draft'
                            })
                    
                    score_candidate_methods = sorted(score_candidate_methods, key=lambda x: x['sim_score'], reverse=True)
                    print('Start build prompt for extracted methods for task: ', task['metadata']['task_id'])
                    prepend_context = "// Here are some relevant code fragments from other files of the repo:\n"
                    seperator = '// ' + '-' * 50
                    prepend_context += seperator + '\n'
                    
                    context_class = task['metadata']['left_context'] + '<FILL_FUNCTION_BODY>' + task['metadata']['right_context'] 
                    # context_class = task['prompt'].split('Based on above, complete the method body of the class\n\n')[-1]
                    len_current_token = self.tokenizer(prepend_context + context_class + task['metadata']['ground_truth'], return_tensors="pt")['input_ids'].size()[1]
                    
                    chosen_context = []
                    for score_method in score_candidate_methods[:20]:
                        # content = repo_embedding_methods[doc_id]
                        f_path = '/'.join(score_method['fpath_tuple'])
                        f_paths_str = f'// {f_path}'
                        left_content = Tools.get_lcontext_method(score_method, self.repo_base_dir)
                        left_content_lines = left_content.split('\n')
                        left_content_lines_comment = [f'// {line}' for line in left_content_lines]
                        f_path_comment = f'// the below code fragment can be found in:'
                        
                        block_str = '\n'.join([f_path_comment, f_paths_str, seperator] + left_content_lines_comment + [seperator]) + '\n'
                        # block_str = '\n'.join(left_content_lines_comment + [seperator]) + '\n'
                        block_token_len = self.tokenizer(block_str, return_tensors="pt")['input_ids'].size()[1]
                        
                        if len_current_token + block_token_len > 8000:
                            break
                        prepend_context += block_str
                        len_current_token += block_token_len
                        chosen_context.append(
                            (score_method['fpath_tuple'], 
                            left_content,
                            score_method['sim_score'], 
                            score_method['method'],
                            score_method['start_line_no'], 
                            score_method['end_line_no'],
                            score_method['type_query']),
                        )
                    prepend_context += """// Based on above, complete the method body of the class\n"""
                    if len(chosen_context) > 0: 
                        prompt = prepend_context + '\n' + context_class
                    else:
                        prompt = context_class
                    
                    max_prompt_token = max(max_prompt_token, self.tokenizer(prompt, return_tensors="pt")['input_ids'].size()[1])
                    
                    new_prompt_line = task.copy()
                    
                    new_prompt_line['prompt'] = prompt
                    new_prompt_line['metadata']['top_k_context'] = [
                        {
                            'fpath_tuple': x[0],
                            'left_content': x[1],
                            'sim_score': x[2],
                            'method': x[3],
                            'start_line_no': x[4],
                            'end_line_no': x[5],
                            'type_query': x[6]
                        } for x in chosen_context
                    ]
                    
                    new_prompt_lines.append(new_prompt_line)
                
        Tools.dump_jsonl(new_prompt_lines, output_file)
        print('Total time: ', time.time() - start_time)