from transformers import AutoTokenizer
from utils.utils import Tools
from extract_repo_elements import ExtractRepoElements
from vector_repo_elements import VectorRepoElements
from utils.vector_utils import BagOfWordsEmbedding, UniXcoderEmbedding
from EEI_build import SketchPromptBuilder
from RUE_build import RUEPromptBuilder

repo_base_dir = 'repositories/defects4j'
parsed_repo_base_dir = 'parsed_repositories/defects4j'
repo_list = 'repositories/defects4j/repo_names.txt'
# repo_base_dir = 'repositories/rambo'
# parsed_repo_base_dir = 'parsed_repositories/rambo'
# repo_list = 'repositories/rambo/repo_names.txt'

repos = open(repo_list, 'r').read().split('\n')

vectorizer = BagOfWordsEmbedding()
# vectorizer = UniXcoderEmbedding()
# extractor = ExtractRepoElements('defects4j', repo_base_dir, parsed_repo_base_dir, repos)
# extractor.extract_elements()
# VectorRepoElements('defects4j', vectorizer, repos).vector_elements()

tasks = Tools.load_jsonl('datasets/defects4j_2k_context_with_type_paramters.jsonl')
model_id = 'deepseek-ai/deepseek-coder-6.7b-base'
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, )
sketchbuilder = SketchPromptBuilder('defects4j', 'bow', repos, tasks, tokenizer)
sketchbuilder.build_prompt('prompts/defects4j_2k_context_with_type_paramters_sketch.jsonl')

# tasks = Tools.load_jsonl('prompts/re_fixed_rambo_bamboo_sketch_unixcoder_repohyper_prompt_deepseek-coder-6.7b-base.jsonl')
# RUEPromptBuilder('rambo', vectorizer, repos, tasks, tokenizer).build_prompt('prompts/re_fixed_sensitivity_analysis_100_lines_deepseek-coder-6.7b-base-RUE.jsonl')