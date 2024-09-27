from utils.vector_utils import BagOfWordsEmbedding, UniXcoderEmbedding

class SingleReranking:

    def __init__(self, vector_builder, batch_size=128):
        self.vector_builder = vector_builder
        self.batch_size = batch_size
    
    def rerank_batch(self, query, docs, doc_embeddings, top_k=20):
        scores = []
        query_embedding = self.vector_builder.build(query)
        
        for idx in range(0, len(docs), self.batch_size):
            start, end = idx, min(idx + self.batch_size, len(docs))
            lines_batch_embeddings = doc_embeddings[start:end]
            scores_batch = self.vector_builder.similarity_score_list(query_embedding, lines_batch_embeddings)
            scores.extend(scores_batch)
        
        doc_ids = [x for _, x in sorted(zip(scores, range(len(docs))), reverse=True)]
        top_k_docs_scores = [(docs[x], scores[x]) for x in doc_ids[:top_k]]
        return top_k_docs_scores
    

class HybridRereanking:
    def __init__(self, bow_w=0.5, unixcoder_w=0.5, batch_size=128):
        self.bow_w = bow_w
        self.unixcoder_w = unixcoder_w
        self.batch_size = batch_size
        self.bow = BagOfWordsEmbedding()
        self.unixcoder = UniXcoderEmbedding()
    
    def rerank_batch(self, query, docs, doc_bows, doc_unixcoders, top_k=20):
        scores = []
        query_unixcoder = self.unixcoder.build(query)
        query_bow = self.bow.build(query)
        
        for idx in range(0, len(docs), self.batch_size):
            start, end = idx, min(idx + self.batch_size, len(docs))
            lines_batch_bow = doc_bows[start:end]
            lines_batch_unixcoder = doc_unixcoders[start:end]
            unixcoder_scores_batch = self.unixcoder.similarity_score_list(query_unixcoder, lines_batch_bow)
            bow_scores_batch = self.bow.similarity_score_list(query_bow, lines_batch_unixcoder)
            scores_batch = [self.bow_w * b + self.unixcoder_w * u for b, u in zip(bow_scores_batch, unixcoder_scores_batch)]
            scores.extend(scores_batch)
            
        doc_ids = [x for _, x in sorted(zip(scores, range(len(docs))), reverse=True)]
        top_k_docs_scores = [(docs[x], scores[x]) for x in doc_ids[:top_k]]

        return top_k_docs_scores