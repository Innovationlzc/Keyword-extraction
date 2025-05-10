import os
import glob
from keybert import KeyBERT
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
from collections import defaultdict
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# 1. 数据集加载函数
def load_semeval_data(data_dir):
    """加载SemEval格式数据集"""
    texts = []
    keywords = []
    
    txt_files = glob.glob(os.path.join(data_dir, "*.txt"))
    
    for txt_path in txt_files:
        with open(txt_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()
            
        ann_path = txt_path.replace('.txt', '.ann')
        doc_keywords = []
        
        if os.path.exists(ann_path):
            with open(ann_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    parts = line.split('\t')
                    if len(parts) >= 3:
                        surface_form = parts[2].strip()
                        if surface_form:
                            doc_keywords.append(surface_form)
        
        # 去重并保留顺序
        seen = set()
        unique_kws = [kw for kw in doc_keywords if not (kw in seen or seen.add(kw))]
        
        texts.append(text)
        keywords.append(unique_kws)
    
    return texts, keywords

# 2. 关键词预处理函数
class KeywordProcessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
    
    def preprocess(self, keyword):
        tokens = nltk.word_tokenize(keyword.lower())
        tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
        tokens = [t for t in tokens if t not in self.stop_words]
        return ' '.join(tokens)

# 3. 评估指标计算
class EnhancedEvaluator:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        # 初始化语义编码模型（与KeyBERT默认模型保持一致）
        self.encoder = SentenceTransformer(model_name)
        self.processor = KeywordProcessor()
    
    def _get_embeddings(self, keywords):
        """将关键词列表编码为语义向量"""
        if not keywords:
            return np.empty((0, 384))  # 适配MiniLM的维度
        return self.encoder.encode(keywords, convert_to_tensor=False)
    
    def _calculate_cosine(self, true_kws, pred_kws, K):
        """计算单个文档的余弦相似度指标"""
        # 获取编码向量
        true_embs = self._get_embeddings(true_kws)
        pred_embs = self._get_embeddings(pred_kws[:K])
        
        if len(true_embs) == 0 or len(pred_embs) == 0:
            return 0.0  # 处理空值情况
        
        # 计算相似度矩阵
        sim_matrix = cosine_similarity(true_embs, pred_embs)
        
        # 计算两种角度的相似度
        max_sim = np.mean(np.max(sim_matrix, axis=0))  # 预测到真实的最近邻
        mean_sim = np.mean(sim_matrix)                # 全局平均
        
        return {
            'max_similarity': max_sim,
            'mean_similarity': mean_sim
        }
    
    def evaluate(self, true_labels, pred_labels, K_values):
        metrics = defaultdict(dict)
        
        for K in K_values:
            total = {
                'precision': 0,
                'recall': 0,
                'f1': 0,
                'cosine_max': 0,
                'cosine_mean': 0
            }
            
            for true, pred in zip(true_labels, pred_labels):
                # 传统指标计算
                true_set = set(true)
                pred_set = set(pred[:K])
                
                tp = len(true_set & pred_set)
                fp = len(pred_set - true_set)
                fn = len(true_set - pred_set)
                
                # 余弦相似度计算
                cosine_metrics = self._calculate_cosine(true, pred, K)
                
                # 累加指标
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                total['precision'] += precision
                total['recall'] += recall
                total['f1'] += f1
                total['cosine_max'] += cosine_metrics['max_similarity']
                total['cosine_mean'] += cosine_metrics['mean_similarity']
            
            # 计算宏平均
            n = len(true_labels)
            metrics[K] = {
                'precision': round(total['precision'] / n, 4),
                'recall': round(total['recall'] / n, 4),
                'f1': round(total['f1'] / n, 4),
                'cosine_max': round(total['cosine_max'] / n, 4),
                'cosine_mean': round(total['cosine_mean'] / n, 4)
            }
        
        return metrics

# 主流程
if __name__ == "__main__":
    # 配置参数
    TEST_DIR = "./semeval_articles_test"
    K_VALUES = [5, 15,25]
    NGRAM_RANGE = (1, 2)  # 根据数据集关键词长度调整
    
    # 初始化组件
    processor = KeywordProcessor()
    kw_model = KeyBERT()
    
    # 加载并预处理数据
    texts, raw_keywords = load_semeval_data(TEST_DIR)
    
    # 预处理真实标签
    true_labels = [
        list(set(processor.preprocess(kw) for kw in kws if kw.strip()))
        for kws in raw_keywords
    ]
    
    # 生成预测标签
    pred_labels = []
    for doc in texts:
        keywords = kw_model.extract_keywords(
            doc,
            keyphrase_ngram_range=NGRAM_RANGE,
            #use_mmr=True,          # 启用最大边际相关去重
            stop_words=None,
            top_n=max(K_VALUES))
        processed = [processor.preprocess(kw[0]) for kw in keywords]
        pred_labels.append(processed)
    
    # 计算评估指标
    evaluator = EnhancedEvaluator()
    metrics = evaluator.evaluate(true_labels, pred_labels, K_VALUES)
    
    # 打印结果
    print("评估结果（宏平均）:")
    for K in K_VALUES:
        print(f"\nK = {K}:")
        print(f"  Precision@K: {metrics[K]['precision']}")
        print(f"  Recall@K:    {metrics[K]['recall']}")
        print(f"  F1@K:        {metrics[K]['f1']}")
        print(f"  Cosine-Max:  {metrics[K]['cosine_max']} (预测关键词与真实集的最大相似度平均)")
        print(f"  Cosine-Mean: {metrics[K]['cosine_mean']} (全局相似度平均)")