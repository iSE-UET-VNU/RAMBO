import torch
import numpy as np
from utils.utils import Tools, FilePathBuilder
from utils.unixcoder import UniXcoder

class SimilarityScore:
    @staticmethod
    def cosine_similarity(embedding_vec1, embedding_vec2):
        return 1 - scipy.spatial.distance.cosine(embedding_vec1, embedding_vec2)
    
    @staticmethod
    def jaccard_similarity(list1, list2):
        set1 = set(list1)
        set2 = set(list2)
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return float(intersection) / union

class BagOfWordsEmbedding:
    def build(self, text):
        return np.array(Tools.tokenize(text))
    
    def build_list(self, text_list):
        return [np.array(Tools.tokenize(text)) for text in text_list]
    
    def vector_file_path(self, file_path):
        return FilePathBuilder.bow_vector_path(file_path)
    
    def similarity_score_list(self, query_embedding, lines_batch_embeddings):
        scores_batch = []
        for line in lines_batch_embeddings:
            scores_batch.append(SimilarityScore.jaccard_similarity(query_embedding, line))
        return scores_batch
    
    def __str__(self):
        return 'BagOfWords'
    
class UniXcoderEmbedding:
    def __init__(self, device='cpu'):
        self.device = device
        self.model = UniXcoder("microsoft/unixcoder-base").to(device)
    
    def build(self, code):
        inputs = self.model.tokenize([code], mode="<encoder-only>", truncation=True, max_length=512, padding=True)
        with torch.no_grad():
            _, embedding = self.model(torch.tensor(inputs).to(self.device))
        return embedding.detach().cpu()

    
    def build_list(self, code_list):
        # get the embedding of the first token of the code [CLS] token
        embeddings = []
        # texts = [dict_code[key] for key in dict_code]
        inputs = self.model.tokenize(code_list, mode="<encoder-only>", truncation=True, max_length=512, padding=True)
        inputs = torch.concat([torch.tensor(_input).reshape(1, -1) for _input in inputs], dim=0).to(self.device)
        with torch.no_grad():
            for batch in inputs.split(256):
                _, embedding = self.model(batch)
                embedding = embedding.detach().cpu()
                embeddings.extend(torch.chunk(embedding, embedding.shape[0], dim=0))
        # concat_embeddings = torch.concat(embeddings, dim=0)
        # print(embeddings)
        # print(concat_embeddings.shape)
        
        return embeddings
    
    def similarity_score(self, embedding_vec1, embedding_vec2):
        return SimilarityScore.cosine_similarity(embedding_vec1, embedding_vec2)
    
    def similarity_score_list(self, query_embedding, lines_batch_embeddings):
        concat_lines_batch_embeddings = torch.cat(lines_batch_embeddings, dim=0).to(self.device)
        scores_batch = torch.nn.functional.cosine_similarity(concat_lines_batch_embeddings, query_embedding, dim=1).tolist()
        return scores_batch
    
    def vector_file_path(self, file_path):
        return FilePathBuilder.UniXcoder_vector_path(file_path)
    
    def __str__(self):
        return 'UniXcoder'