import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, TimeDistributed
import keybert
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, TimeDistributed, Dense
from sklearn.model_selection import train_test_split

def train_evaluate_bilstm(texts, true_keywords, K_values=[5, 10, 15], max_len=128, embedding_dim=100, lstm_units=64, epochs=10, batch_size=32):
    """
    完整的Bi-LSTM训练评估流程
    返回评估指标字典和训练历史
    """
    
    # 1. 数据预处理
    class BilstmPreprocessor:
        def __init__(self):
            self.word2idx = {"<PAD>": 0, "<UNK>": 1}
            self.tag2idx = {"O": 0, "B-KEY": 1, "I-KEY": 2}
            self.idx2word = {v: k for k, v in self.word2idx.items()}
            self.lemmatizer = WordNetLemmatizer()
            self.stop_words = set(stopwords.words('english'))
            
        def preprocess(self, texts, keywords):
            X, y = [], []
            vocab = set()
            
            # 构建词汇表和标签
            for text, kws in zip(texts, true_keywords):
                # 清洗文本
                tokens = nltk.word_tokenize(text.lower())
                tokens = [self.lemmatizer.lemmatize(t) for t in tokens if t not in self.stop_words]
                
                # 生成关键词集合（词形还原后）
                key_lemmas = set()
                for kw in kws:
                    kw_tokens = nltk.word_tokenize(kw.lower())
                    kw_lemmas = [self.lemmatizer.lemmatize(t) for t in kw_tokens]
                    key_lemmas.update(kw_lemmas)
                
                # 生成BIO标签
                bio_tags = []
                for t in tokens:
                    if t in key_lemmas:
                        bio_tags.append("B-KEY")  # 简化为单标签
                    else:
                        bio_tags.append("O")
                
                # 构建词汇表
                for token in tokens:
                    if token not in self.word2idx:
                        self.word2idx[token] = len(self.word2idx)
                        self.idx2word[len(self.word2idx)-1] = token
                
                # 转换为索引
                x_seq = [self.word2idx.get(t, 1) for t in tokens]  # 1=UNK
                y_seq = [self.tag2idx[tag] for tag in bio_tags]
                
                X.append(x_seq)
                y.append(y_seq)
            
            # 填充序列
            X = pad_sequences(X, maxlen=max_len, padding='post', truncating='post', value=0)
            y = pad_sequences(y, maxlen=max_len, padding='post', truncating='post', value=0)
            y = to_categorical(y, num_classes=len(self.tag2idx))
            
            return X, y
        
        def decode_predictions(self, X, y_pred):
            """将预测结果转换为关键词列表"""
            pred_keywords = []
            for i in range(len(X)):
                tokens = [self.idx2word.get(idx, "<UNK>") for idx in X[i] if idx != 0]
                tags = [np.argmax(vec) for vec in y_pred[i][:len(tokens)]]  # 裁剪填充部分
                
                keywords = []
                current = []
                for t, tag in zip(tokens, tags):
                    if tag == self.tag2idx["B-KEY"]:
                        if current:
                            keywords.append(" ".join(current))
                            current = []
                        current.append(t)
                    elif tag == self.tag2idx["I-KEY"]:
                        current.append(t)
                    else:
                        if current:
                            keywords.append(" ".join(current))
                            current = []
                if current:
                    keywords.append(" ".join(current))
                
                pred_keywords.append(keywords[:max(K_values)])
            
            return pred_keywords
    
    # 2. 预处理数据
    preprocessor = BilstmPreprocessor()
    X, y = preprocessor.preprocess(texts, true_keywords)
    
    # 3. 划分数据集
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, range(len(texts)), test_size=0.2, random_state=42
    )
    test_texts = [texts[i] for i in idx_test]
    test_true_keywords = [true_keywords[i] for i in idx_test]
    
    # 4. 构建模型
    model = Sequential()
    model.add(Embedding(
        input_dim=len(preprocessor.word2idx),
        output_dim=embedding_dim,
        mask_zero=True
    ))
    model.add(Bidirectional(LSTM(
        lstm_units,
        return_sequences=True,
        dropout=0.3,
        recurrent_dropout=0.3
    )))
    model.add(TimeDistributed(Dense(len(preprocessor.tag2idx), activation='softmax')))
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # 5. 训练模型
    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.1,
        verbose=1
    )
    
    # 6. 预测并解码
    y_pred = model.predict(X_test)
    pred_keywords = preprocessor.decode_predictions(X_test, y_pred)
    
    # 7. 预处理真实关键词（与KeyBERT一致）
    processor = KeywordProcessor()
    processed_true = [
        list(set(processor.preprocess(kw) for kw in kws if kw.strip()))
        for kws in test_true_keywords
    ]
    
    # 8. 评估
    evaluator = EnhancedEvaluator()
    metrics = evaluator.evaluate(processed_true, pred_keywords, K_values)
    
    return metrics, history, model



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

class BioProcessor:
    def __init__(self, max_len=128):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.max_len = max_len
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.tag2idx = {"O": 0, "B-KEY": 1, "I-KEY": 2}
        
    def preprocess(self, text, keywords):
        # 分词与词形还原
        tokens = nltk.word_tokenize(text.lower())
        lemmas = [self.lemmatizer.lemmatize(t) for t in tokens]
        
        # 生成BIO标签
        key_lemmas = set()
        for kw in keywords:
            kw_tokens = nltk.word_tokenize(kw.lower())
            kw_lemmas = [self.lemmatizer.lemmatize(t) for t in kw_tokens]
            key_lemmas.update(kw_lemmas)
        
        bio_tags = []
        for lemma in lemmas:
            if lemma in key_lemmas:
                bio_tags.append("B-KEY")  # 简化为单标签
            else:
                bio_tags.append("O")
        
        # 构建词汇表
        for lemma in lemmas:
            if lemma not in self.word2idx:
                self.word2idx[lemma] = len(self.word2idx)
        
        return lemmas, bio_tags
    
    def encode_samples(self, all_texts, all_keywords):
        X, y = [], []
        for text, keywords in zip(all_texts, all_keywords):
            lemmas, tags = self.preprocess(text, keywords)
            
            # 转换为索引
            x_seq = [self.word2idx.get(lemma, 1) for lemma in lemmas]  # 1=UNK
            y_seq = [self.tag2idx[tag] for tag in tags]
            
            X.append(x_seq)
            y.append(y_seq)
        
        # 填充序列
        X = pad_sequences(X, maxlen=self.max_len, padding='post', truncating='post')
        y = pad_sequences(y, maxlen=self.max_len, padding='post', truncating='post', value=0)
        y = to_categorical(y, num_classes=len(self.tag2idx))
        
        return X, y




def build_bilstm(vocab_size, embedding_dim=128, lstm_units=64):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, mask_zero=True))
    model.add(Bidirectional(LSTM(units=lstm_units, return_sequences=True)))
    model.add(TimeDistributed(Dense(len(self.tag2idx), activation='softmax')))
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def extract_keywords_from_bio(tokens, tags):
    keywords = []
    current_key = []
    for token, tag in zip(tokens, tags):
        if tag == "B-KEY":
            if current_key:
                keywords.append(" ".join(current_key))
                current_key = []
            current_key.append(token)
        elif tag == "I-KEY":
            current_key.append(token)
        else:
            if current_key:
                keywords.append(" ".join(current_key))
                current_key = []
    if current_key:
        keywords.append(" ".join(current_key))
    return keywords

K_VALUES = [5, 10, 15]
# 加载数据
texts, keywords = load_semeval_data("./semeval_articles_test")
evaluator = EnhancedEvaluator()

# 运行KeyBERT评估
kw_metrics = evaluator.evaluate(texts, keywords,K_VALUES)

# 运行Bi-LSTM评估
bilstm_metrics, history = train_evaluate_bilstm(texts, keywords)


# 结果对比
def print_comparison(kw_metrics, bilstm_metrics):
    print("\n方法对比结果:")
    for K in K_VALUES:
        print(f"\nK = {K}:")
        print(f"               Precision@K  Recall@K  F1@K    Cosine-Max")
        print(f"KeyBERT         {kw_metrics[K]['precision']:.3f}       {kw_metrics[K]['recall']:.3f}    {kw_metrics[K]['f1']:.3f}   {kw_metrics[K]['cosine_max']:.3f}")
        print(f"Bi-LSTM         {bilstm_metrics[K]['precision']:.3f}       {bilstm_metrics[K]['recall']:.3f}    {bilstm_metrics[K]['f1']:.3f}   {bilstm_metrics[K]['cosine_max']:.3f}")

print_comparison(kw_metrics, bilstm_metrics)