def make_an_extended_block(retrieved_context, tokenizer):
    """
    Generates a formatted code block with comments indicating the file path and the code content.
    """
    content = retrieved_context["left_content"]
    # print(retrieved_context)
    # content = retrieved_context['context']
    # put the file path in the comment
    f_path_comment = f"// The below code fragment can be found in:\n"
    # print(retrieved_context['metadata'][0]['fpath_tuple'])
    # f_paths_str = '// '+'/'.join(retrieved_context['metadata'][0]['fpath_tuple']) + '\n'
    f_paths_str = "// " + "/".join(retrieved_context["fpath_tuple"]) + "\n"
    # put code lines in the comment
    code_lines = content.splitlines(keepends=True)
    content_lines_comment = [f"// {line.rstrip()}\n" for line in code_lines]
    # aggregate the comment and the code lines
    seperator = "// " + "-" * 50 + "\n"
    block_str = "".join(
        [f_path_comment, f_paths_str, seperator] + content_lines_comment + [seperator]
    )
    # block_str = "".join(content_lines_comment + [seperator])
    tokenized_block = tokenizer.tokenize(block_str)
    token_len = len(tokenized_block)
    return block_str, token_len


def make_str_block_with_max_token_length(
    tokenizer, max_token_num: int, context_str: str, with_comment=False
):
    str_block = ""
    new_line = context_str.splitlines(keepends=True)
    if with_comment:
        context_str_lines_comment = [f"# {line}" for line in new_line]
        new_line = context_str_lines_comment
    curr_len = 0
    for i in range(1, len(new_line) + 1):
        line_len = len(tokenizer.tokenize(new_line[-i]))
        if line_len + curr_len < max_token_num:
            str_block = new_line[-i] + str_block
            curr_len += line_len
        else:
            break
    return str_block


def build_infile_prompt(case, tokenizer, max_num_tokens):
    comment = "# Complete the body of the following codes:\n"
    comment_length = len(tokenizer.tokenize(comment))
    max_num_tokens = max_num_tokens // 2 - comment_length
    context = "".join(case["context"])
    prompt = make_str_block_with_max_token_length(tokenizer, max_num_tokens, context)
    return comment + prompt


def build_retrieval_prompt(case, tokenizer, max_num_tokens, max_top_k):
    # original context
    context_max_tokens = max_num_tokens // 2
    comment = "// Based on above, complete the body of the unfinished method:\n"
    comment_length = len(tokenizer.tokenize(comment))
    # left_context = case['prompt'].split('--------------------------------------------------\n\n')[-1]
    context = make_str_block_with_max_token_length(
        tokenizer,
        context_max_tokens - comment_length,
        "".join(case["metadata"]["left_context"]),
    )
    # context = make_str_block_with_max_token_length(tokenizer, context_max_tokens - comment_length, left_context)
    context_prompt = comment + context

    # retrieved example
    seperator = "// " + "-" * 50
    retrieval_prompt = (
        "// Here are some relevant code fragments from other files of the repo:\n"
    )
    retrieval_prompt += seperator + "\n"

    num_chosen_context = 0
    max_retrieval_length = (
        max_num_tokens
        - len(tokenizer.tokenize(context_prompt))
        - len(tokenizer.tokenize(case["metadata"]["right_context"]))
    )
    current_token_length = len(tokenizer.tokenize(retrieval_prompt))
    retrival_blocks = []
    top_k_context = case["metadata"]["top_k_context"]
    for i in range(0, len(top_k_context)):
        retrieval_context = top_k_context[i]
        if num_chosen_context >= max_top_k:
            break
        block_str, token_len = make_an_extended_block(retrieval_context, tokenizer)
        if current_token_length + token_len < max_retrieval_length:
            retrival_blocks.insert(0, block_str)
            current_token_length += token_len
            num_chosen_context += 1
        else:
            continue
    retrieval_prompt += "".join(retrival_blocks)
    return retrieval_prompt + context_prompt


def build_prompt(case, tokenizer, max_num_tokens, max_top_k=20, mode="retrieval"):
    prompt = ""
    prompt = build_retrieval_prompt(case, tokenizer, max_num_tokens, max_top_k)
    return prompt
