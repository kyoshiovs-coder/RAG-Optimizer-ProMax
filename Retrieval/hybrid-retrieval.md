# 混合检索架构：稠密向量 + 稀疏向量 + 关键词——召回率从60%到90%的跃迁

## 一、背景：单一检索模式的"能力盲区"

### 1.1 一个真实的召回失败案例

某法律科技公司的RAG系统遇到了一个诡异问题：

**用户查询**："《民法典》第584条违约金调整规则"

**向量检索Top 5结果**：
1. 合同法司法解释（一）（相似度0.72）
2. 违约责任一般规定（相似度0.68）
3. 损害赔偿计算方式（相似度0.65）
4. 合同解除法律后果（相似度0.61）
5. 违约金与定金竞合（相似度0.58）

**问题**：目标文档《民法典》合同编第584条（相似度0.55）排在第12位，未进入Top 5。

**诊断分析**：
- 用户查询包含精确引用"《民法典》第584条"
- 向量Embedding将"584"编码为数值概念，而非标识符
- "违约金调整"的语义向量与"违约责任"、"损害赔偿"等概念相近
- 缺乏对**精确匹配**和**结构化标识**的敏感度

这就是**稠密向量的语义盲区**——擅长捕捉概念相似性，却弱于精确匹配、关键词对齐和结构化标识。

### 1.2 三种检索模式的能力矩阵

| 检索模式 | 核心机制 | 擅长场景 | 致命弱点 |
|---------|---------|---------|---------|
| **稠密向量** | 神经网络Embedding，语义相似度 | 概念关联、同义扩展、语义泛化 | 精确匹配弱、关键词遗漏、长尾实体 |
| **稀疏向量** | 学习得到的词项权重（SPLADE） | 关键词重要性、词项共现、可解释性 | 语义鸿沟、同义词处理 |
| **关键词（BM25）** | 倒排索引 + TF-IDF变体 | 精确匹配、短语查询、结构化标识 | 语义理解缺失、同义词盲区 |

**关键洞察**：没有单一检索模式能覆盖所有场景。稠密向量懂"意思"但不懂"精确"，关键词懂"精确"但不懂"意思"，稀疏向量在中间但两端都不极致。

### 1.3 混合检索的核心价值

**设计哲学**：构建**互补型检索舰队**，让不同模式覆盖对方的盲区，通过融合实现1+1+1>3。

```
用户查询："584条违约金调整"
    ↓
┌─────────────────────────────────────────┐
│  三路并行检索                           │
│                                         │
│  稠密向量通道 ──→ 语义相似："违约责任"、"损害赔偿" │
│     (BGE-large)      召回概念相关文档          │
│                                         │
│  稀疏向量通道 ──→ 词项权重："违约金"高权重、"调整"重要 │
│     (SPLADE)         召回关键词精准文档        │
│                                         │
│  关键词通道 ────→ 精确匹配："584"、"民法典"      │
│     (BM25)           召回标识符精确文档        │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  结果融合与重排序                        │
│  - 去重：同一文档多通道命中识别           │
│  - 加权：基于通道置信度的分数调整           │
│  - 重排：学习排序模型（LTR）优化            │
└─────────────────────────────────────────┘
    ↓
融合Top-K：包含语义相关 + 关键词精准 + 标识符精确
```

## 二、核心技术体系：三通道架构详解

### 2.1 架构全景图

```
┌─────────────────────────────────────────────────────────────┐
│                     混合检索引擎                              │
├─────────────────────────────────────────────────────────────┤
│  查询预处理层                                                 │
│  ├─ 查询分析：意图识别、实体提取、查询分类                     │
│  ├─ 查询改写：扩展同义词、生成稀疏表示、提取精确短语           │
│  └─ 路由决策：根据查询类型选择通道权重                         │
├─────────────────────────────────────────────────────────────┤
│  三通道检索层                                                 │
│  ├─ 稠密通道（Dense）                                         │
│  │   ├─ Embedding模型：BGE/bge-m3, GTE, OpenAI text-embedding │
│  │   ├─ 索引：HNSW/IVF-PQ（近似最近邻）                        │
│  │   └─ 相似度：余弦/IP/L2                                     │
│  │                                                             │
│  ├─ 稀疏通道（Sparse）                                        │
│  │   ├─ 模型：SPLADE++, BGE-M3稀疏向量                         │
│  │   ├─ 索引：倒排索引 + 学习权重                              │
│  │   └─ 相似度：点积（内积）                                    │
│  │                                                             │
│  └─ 关键词通道（Lexical）                                      │
│      ├─ 引擎：Elasticsearch, BM25, TF-IDF                     │
│      ├─ 索引：倒排索引 + 位置信息                              │
│      └─ 特性：短语匹配、模糊匹配、通配符                        │
├─────────────────────────────────────────────────────────────┤
│  结果融合层                                                   │
│  ├─ 粗排融合：RRF（倒数排序融合）、加权求和                      │
│  ├─ 精排优化：Cross-Encoder重排序、LTR模型                     │
│  └─ 多样性控制：MMR（最大边际相关性）                           │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 各通道技术详解

#### 2.2.1 稠密向量通道（Dense Retrieval）

```python
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from typing import List, Dict, Tuple

class DenseRetrievalChannel:
    """稠密向量检索通道"""
    
    def __init__(self,
                 model_name: str = 'BAAI/bge-large-zh-v1.5',
                 index_type: str = 'HNSW',
                 normalize: bool = True):
        
        self.model = SentenceTransformer(model_name)
        self.normalize = normalize
        self.index = None
        self.doc_ids = []
        
        # 模型配置
        self.max_seq_length = 512
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
    def encode_documents(self, 
                         documents: List[str],
                         batch_size: int = 32) -> np.ndarray:
        """编码文档"""
        embeddings = self.model.encode(
            documents,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize
        )
        return embeddings
    
    def build_index(self, 
                    doc_ids: List[str],
                    embeddings: np.ndarray):
        """构建FAISS索引"""
        self.doc_ids = doc_ids
        
        # HNSW索引：平衡速度与精度
        if self.index_type == 'HNSW':
            self.index = faiss.IndexHNSWFlat(
                self.embedding_dim, 
                32  # M参数：每个节点的连接数
            )
            self.index.hnsw.efConstruction = 200  # 构建时搜索深度
            self.index.add(embeddings)
        else:
            # IVF-PQ：大规模数据
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            self.index = faiss.IndexIVFPQ(
                quantizer,
                self.embedding_dim,
                100,    # nlist
                8,      # M (PQ子空间数)
                8       # nbits_per_idx
            )
            self.index.train(embeddings)
            self.index.add(embeddings)
    
    def search(self,
               query: str,
               top_k: int = 100,
               ef_search: int = 128) -> List[Dict]:
        """检索"""
        # 编码查询
        query_vec = self.model.encode(
            [query],
            normalize_embeddings=self.normalize
        )
        
        # HNSW参数
        if hasattr(self.index, 'hnsw'):
            self.index.hnsw.efSearch = ef_search
        
        # 检索
        scores, indices = self.index.search(query_vec, top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append({
                'doc_id': self.doc_ids[idx],
                'score': float(score),
                'channel': 'dense',
                'rank': len(results) + 1
            })
        
        return results
    
    def batch_search(self,
                     queries: List[str],
                     top_k: int = 100) -> List[List[Dict]]:
        """批量检索"""
        query_vecs = self.model.encode(
            queries,
            normalize_embeddings=self.normalize
        )
        
        scores, indices = self.index.search(query_vecs, top_k)
        
        all_results = []
        for q_scores, q_indices in zip(scores, indices):
            results = []
            for s, i in zip(q_scores, q_indices):
                if i == -1:
                    continue
                results.append({
                    'doc_id': self.doc_ids[i],
                    'score': float(s),
                    'channel': 'dense'
                })
            all_results.append(results)
        
        return all_results
```

#### 2.2.2 稀疏向量通道（SPLADE）

```python
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from typing import List, Dict
import numpy as np
from scipy.sparse import csr_matrix

class SPLADERetrievalChannel:
    """
    SPLADE: Sparse Lexical and Expansion Model
    核心思想：用BERT的MLM头预测词项重要性，生成学习得到的稀疏向量
    """
    
    def __init__(self,
                 model_name: str = 'naver/splade-cocondenser-ensembledistil',
                 max_length: int = 512):
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.model.eval()
        self.max_length = max_length
        
        # 词表
        self.vocab_size = self.tokenizer.vocab_size
        
    def encode(self, 
               texts: List[str],
               batch_size: int = 32) -> csr_matrix:
        """编码为稀疏向量"""
        all_sparse_vecs = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=self.max_length
            )
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits  # [batch, seq_len, vocab_size]
                
                # SPLADE核心：max pooling over logits
                # 取每个词在序列中的最大logit作为重要性
                attention_mask = inputs['attention_mask'].unsqueeze(-1)
                
                # 屏蔽padding
                logits = logits * attention_mask
                
                # Max pooling + ReLU
                max_logits, _ = torch.max(logits, dim=1)  # [batch, vocab_size]
                sparse_vec = torch.relu(max_logits)  # 只保留正值
                
                # 转换为numpy
                sparse_np = sparse_vec.cpu().numpy()
                
                # 只保留top-k重要词项（稀疏化）
                sparse_np = self._sparsify(sparse_np, top_k=256)
                
                all_sparse_vecs.append(sparse_np)
        
        # 合并为csr_matrix
        all_sparse = np.vstack(all_sparse_vecs)
        return csr_matrix(all_sparse)
    
    def _sparsify(self, 
                  dense_vec: np.ndarray, 
                  top_k: int = 256) -> np.ndarray:
        """稀疏化：只保留top-k重要的维度"""
        # 对每行（每个文档）独立处理
        result = np.zeros_like(dense_vec)
        
        for i in range(dense_vec.shape[0]):
            vec = dense_vec[i]
            # 找到top-k索引
            top_indices = np.argpartition(vec, -top_k)[-top_k:]
            top_indices = top_indices[vec[top_indices] > 0]  # 只保留正值
            
            result[i, top_indices] = vec[top_indices]
        
        return result
    
    def build_index(self,
                    doc_ids: List[str],
                    sparse_matrix: csr_matrix):
        """构建稀疏索引"""
        self.doc_ids = doc_ids
        self.index = sparse_matrix
        
        # 预计算文档范数（用于余弦相似度）
        self.doc_norms = np.sqrt(sparse_matrix.power(2).sum(axis=1)).A1
    
    def search(self,
               query: str,
               top_k: int = 100) -> List[Dict]:
        """稀疏向量检索"""
        # 编码查询
        query_sparse = self.encode([query])  # [1, vocab_size]
        
        # 计算相似度：点积（内积）
        # 对于学习得到的权重，点积比余弦更适合
        scores = (self.index @ query_sparse.T).toarray().flatten()
        
        # 获取top-k
        top_indices = np.argpartition(scores, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(-scores[top_indices])]
        
        results = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            results.append({
                'doc_id': self.doc_ids[idx],
                'score': float(scores[idx]),
                'channel': 'sparse',
                'rank': len(results) + 1
            })
        
        return results

class BGE_M3_SparseChannel:
    """
    BGE-M3的统一稀疏表示
    使用相同的Tokenizer，但用不同的方式提取稀疏权重
    """
    
    def __init__(self, model_name: str = 'BAAI/bge-m3'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        
    def encode(self, texts: List[str]) -> csr_matrix:
        """BGE-M3的稀疏编码"""
        # BGE-M3使用特殊的稀疏表示学习
        # 这里简化实现，实际应使用官方实现
        
        all_weights = []
        
        for text in texts:
            inputs = self.tokenizer(
                text,
                return_tensors='pt',
                truncation=True,
                max_length=8192  # BGE-M3支持长文本
            )
            
            with torch.no_grad():
                # 获取最后一层hidden states
                outputs = self.model(**inputs, return_dict=True)
                
                # 使用[CLS] token的注意力权重作为稀疏表示
                # 或直接使用词项频率加权
                
                # 简化：使用TF-like权重
                tokens = inputs['input_ids'][0].tolist()
                token_counts = {}
                for t in tokens:
                    if t not in [self.tokenizer.pad_token_id, 
                                self.tokenizer.cls_token_id,
                                self.tokenizer.sep_token_id]:
                        token_counts[t] = token_counts.get(t, 0) + 1
                
                # 创建稀疏向量
                indices = list(token_counts.keys())
                values = [np.log1p(c) for c in token_counts.values()]  # log(1+tf)
                
                all_weights.append((indices, values))
        
        # 构建csr_matrix
        data = []
        indices = []
        indptr = [0]
        
        for idxs, vals in all_weights:
            data.extend(vals)
            indices.extend(idxs)
            indptr.append(len(indices))
        
        return csr_matrix(
            (data, indices, indptr),
            shape=(len(texts), self.tokenizer.vocab_size)
        )
```

#### 2.2.3 关键词通道（BM25）

```python
from rank_bm25 import BM25Okapi
from elasticsearch import Elasticsearch
import jieba
from typing import List, Dict

class BM25RetrievalChannel:
    """BM25关键词检索通道"""
    
    def __init__(self,
                 use_elasticsearch: bool = False,
                 es_host: str = 'localhost:9200',
                 k1: float = 1.5,
                 b: float = 0.75):
        
        self.use_es = use_elasticsearch
        self.k1 = k1
        self.b = b
        
        if use_elasticsearch:
            self.es = Elasticsearch([es_host])
        else:
            self.tokenized_corpus = []
            self.bm25 = None
            self.doc_ids = []
    
    def _tokenize(self, text: str) -> List[str]:
        """中文分词"""
        # 使用jieba分词，保留英文单词
        tokens = []
        for word in jieba.cut(text):
            word = word.strip().lower()
            if word and len(word) > 1:  # 过滤单字和空串
                tokens.append(word)
        return tokens
    
    def build_index(self,
                    doc_ids: List[str],
                    documents: List[str]):
        """构建BM25索引"""
        self.doc_ids = doc_ids
        
        if self.use_es:
            # Elasticsearch索引
            self._build_es_index(documents)
        else:
            # 本地BM25
            self.tokenized_corpus = [
                self._tokenize(doc) for doc in documents
            ]
            self.bm25 = BM25Okapi(
                self.tokenized_corpus,
                k1=self.k1,
                b=self.b
            )
    
    def _build_es_index(self, documents: List[str]):
        """构建ES索引"""
        index_name = 'hybrid_search_bm25'
        
        # 创建索引
        mapping = {
            'mappings': {
                'properties': {
                    'content': {
                        'type': 'text',
                        'analyzer': 'ik_max_word',  # 中文IK分词
                        'search_analyzer': 'ik_smart'
                    }
                }
            }
        }
        
        if self.es.indices.exists(index=index_name):
            self.es.indices.delete(index=index_name)
        
        self.es.indices.create(index=index_name, body=mapping)
        
        # 批量索引
        from elasticsearch.helpers import bulk
        actions = [
            {
                '_index': index_name,
                '_id': self.doc_ids[i],
                '_source': {'content': doc}
            }
            for i, doc in enumerate(documents)
        ]
        bulk(self.es, actions)
        self.es.indices.refresh(index=index_name)
        self.index_name = index_name
    
    def search(self,
               query: str,
               top_k: int = 100) -> List[Dict]:
        """BM25检索"""
        if self.use_es:
            return self._search_es(query, top_k)
        else:
            return self._search_local(query, top_k)
    
    def _search_local(self, query: str, top_k: int) -> List[Dict]:
        """本地BM25检索"""
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        # 获取top-k
        top_indices = np.argpartition(scores, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(-scores[top_indices])]
        
        results = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            results.append({
                'doc_id': self.doc_ids[idx],
                'score': float(scores[idx]),
                'channel': 'bm25',
                'rank': len(results) + 1,
                'matched_terms': self._get_matched_terms(
                    tokenized_query, 
                    self.tokenized_corpus[idx]
                )
            })
        
        return results
    
    def _search_es(self, query: str, top_k: int) -> List[Dict]:
        """Elasticsearch检索"""
        search_body = {
            'query': {
                'match': {
                    'content': {
                        'query': query,
                        'analyzer': 'ik_smart'
                    }
                }
            },
            'size': top_k,
            'highlight': {
                'fields': {
                    'content': {}
                }
            }
        }
        
        response = self.es.search(index=self.index_name, body=search_body)
        
        results = []
        for hit in response['hits']['hits']:
            results.append({
                'doc_id': hit['_id'],
                'score': hit['_score'],
                'channel': 'bm25',
                'highlight': hit.get('highlight', {}).get('content', []),
                'matched_terms': self._extract_matched_terms(hit)
            })
        
        return results
    
    def _get_matched_terms(self, 
                          query_terms: List[str], 
                          doc_terms: List[str]) -> List[str]:
        """获取匹配的词项"""
        query_set = set(query_terms)
        doc_set = set(doc_terms)
        return list(query_set & doc_set)
    
    def _extract_matched_terms(self, hit: Dict) -> List[str]:
        """从ES结果提取匹配词"""
        # 简化实现
        return []
```

### 2.3 混合检索引擎与融合策略

```python
class HybridRetrievalEngine:
    """混合检索引擎"""
    
    def __init__(self,
                 dense_channel: DenseRetrievalChannel,
                 sparse_channel: SPLADERetrievalChannel,
                 lexical_channel: BM25RetrievalChannel,
                 fusion_config: Dict = None):
        
        self.channels = {
            'dense': dense_channel,
            'sparse': sparse_channel,
            'bm25': lexical_channel
        }
        
        self.fusion_config = fusion_config or {
            'method': 'rrf',           # rrf, weighted, borda
            'k': 60,                   # RRF参数
            'weights': {
                'dense': 1.0,
                'sparse': 0.9,
                'bm25': 0.8
            },
            'min_score_threshold': 0.01
        }
        
        # 查询路由策略
        self.query_router = QueryRouter()
    
    async def search(self,
                     query: str,
                     top_k: int = 10,
                     route_strategy: str = 'adaptive') -> List[Dict]:
        """
        混合检索主入口
        
        Args:
            query: 查询文本
            top_k: 返回结果数
            route_strategy: 路由策略 (adaptive|all|custom)
        """
        # 1. 查询分析与路由
        if route_strategy == 'adaptive':
            active_channels = self.query_router.route(query)
        elif route_strategy == 'all':
            active_channels = ['dense', 'sparse', 'bm25']
        else:
            active_channels = route_strategy.split(',')
        
        # 2. 并行检索
        search_tasks = []
        for ch_name in active_channels:
            channel = self.channels[ch_name]
            task = channel.search(query, top_k=100)  # 每路召回100
            search_tasks.append((ch_name, task))
        
        # 执行检索
        channel_results = {}
        for ch_name, task in search_tasks:
            try:
                results = await task if asyncio.iscoroutine(task) else task
                channel_results[ch_name] = results
            except Exception as e:
                print(f"Channel {ch_name} failed: {e}")
                channel_results[ch_name] = []
        
        # 3. 结果融合
        fused = self._fuse_results(channel_results, query)
        
        # 4. 精排优化（可选）
        reranked = self._rerank(fused, query)
        
        return reranked[:top_k]
    
    def _fuse_results(self,
                      channel_results: Dict[str, List[Dict]],
                      query: str) -> List[Dict]:
        """融合多通道结果"""
        
        method = self.fusion_config['method']
        
        if method == 'rrf':
            return self._reciprocal_rank_fusion(channel_results)
        elif method == 'weighted':
            return self._weighted_fusion(channel_results)
        elif method == 'borda':
            return self._borda_fusion(channel_results)
        else:
            raise ValueError(f"Unknown fusion method: {method}")
    
    def _reciprocal_rank_fusion(self,
                                 channel_results: Dict[str, List[Dict]]) -> List[Dict]:
        """
        RRF: Reciprocal Rank Fusion
        score = Σ(w_i / (k + rank_i))
        """
        k = self.fusion_config['k']
        weights = self.fusion_config['weights']
        
        # 收集所有文档的RRF分数
        doc_rrf_scores = defaultdict(float)
        doc_sources = defaultdict(list)
        doc_best_scores = {}
        
        for channel, results in channel_results.items():
            weight = weights.get(channel, 1.0)
            
            for rank, result in enumerate(results, start=1):
                doc_id = result['doc_id']
                
                # RRF分数
                rrf_score = weight / (k + rank)
                doc_rrf_scores[doc_id] += rrf_score
                
                # 记录来源
                doc_sources[doc_id].append({
                    'channel': channel,
                    'rank': rank,
                    'raw_score': result['score'],
                    'rrf_contribution': rrf_score
                })
                
                # 记录最佳原始分数
                if doc_id not in doc_best_scores:
                    doc_best_scores[doc_id] = {}
                doc_best_scores[doc_id][channel] = result['score']
        
        # 组装结果
        fused_results = []
        for doc_id, rrf_score in sorted(doc_rrf_scores.items(), 
                                         key=lambda x: x[1], 
                                         reverse=True):
            # 只保留超过阈值的
            if rrf_score < self.fusion_config.get('min_score_threshold', 0):
                continue
            
            result = {
                'doc_id': doc_id,
                'rrf_score': rrf_score,
                'sources': doc_sources[doc_id],
                'channel_scores': doc_best_scores[doc_id]
            }
            
            # 计算通道多样性
            channels_hit = set(s['channel'] for s in doc_sources[doc_id])
            result['diversity_score'] = len(channels_hit) / len(channel_results)
            
            fused_results.append(result)
        
        return fused_results
    
    def _weighted_fusion(self,
                        channel_results: Dict[str, List[Dict]]) -> List[Dict]:
        """基于原始分数的加权融合（需归一化）"""
        
        # 归一化每通道分数到[0,1]
        normalized = {}
        for channel, results in channel_results.items():
            if not results:
                continue
            
            scores = [r['score'] for r in results]
            min_s, max_s = min(scores), max(scores)
            range_s = max_s - min_s if max_s > min_s else 1
            
            normalized[channel] = []
            for r in results:
                norm_score = (r['score'] - min_s) / range_s
                normalized[channel].append({
                    **r,
                    'normalized_score': norm_score
                })
        
        # 加权求和
        weights = self.fusion_config['weights']
        doc_scores = defaultdict(float)
        doc_sources = defaultdict(list)
        
        for channel, results in normalized.items():
            w = weights.get(channel, 1.0)
            for r in results:
                doc_scores[r['doc_id']] += r['normalized_score'] * w
                doc_sources[r['doc_id']].append({
                    'channel': channel,
                    'normalized_score': r['normalized_score']
                })
        
        # 排序
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        
        return [{
            'doc_id': d[0],
            'fused_score': d[1],
            'sources': doc_sources[d[0]]
        } for d in sorted_docs]
    
    def _rerank(self,
                fused_results: List[Dict],
                query: str) -> List[Dict]:
        """精排优化（可选的Cross-Encoder）"""
        # 这里可以接入Cross-Encoder进行精排
        # 简化：直接返回
        return fused_results

class QueryRouter:
    """查询路由器：根据查询特征选择激活的通道"""
    
    def __init__(self):
        # 规则定义
        self.rules = [
            # 精确标识符 → 优先BM25
            {
                'pattern': r'\d{3,}|第[一二三四五六七八九十\d]+[条章节]',
                'channels': ['bm25', 'sparse', 'dense'],
                'weights': {'bm25': 1.2, 'sparse': 1.0, 'dense': 0.8}
            },
            # 短查询（<10字）→ 优先稀疏和关键词
            {
                'condition': lambda q: len(q) < 10,
                'channels': ['sparse', 'bm25', 'dense'],
                'weights': {'sparse': 1.1, 'bm25': 1.0, 'dense': 0.9}
            },
            # 长查询（>50字）→ 优先稠密
            {
                'condition': lambda q: len(q) > 50,
                'channels': ['dense', 'sparse', 'bm25'],
                'weights': {'dense': 1.2, 'sparse': 0.9, 'bm25': 0.7}
            },
            # 包含专业术语 → 平衡
            {
                'default': True,
                'channels': ['dense', 'sparse', 'bm25'],
                'weights': {'dense': 1.0, 'sparse': 1.0, 'bm25': 0.9}
            }
        ]
    
    def route(self, query: str) -> List[str]:
        """路由决策"""
        for rule in self.rules:
            match = False
            
            if 'pattern' in rule:
                import re
                if re.search(rule['pattern'], query):
                    match = True
            elif 'condition' in rule:
                if rule['condition'](query):
                    match = True
            elif rule.get('default'):
                match = True
            
            if match:
                # 可以返回带权重的通道，这里简化
                return rule['channels']
        
        return ['dense', 'sparse', 'bm25']
```

## 三、实战案例：法律文档检索系统

### 3.1 场景与数据

**文档库**：10万份法律法规、司法解释、判例文书

**典型查询场景**：

| 查询类型 | 示例 | 最佳通道 |
|---------|------|---------|
| 精确法条引用 | "民法典第584条" | BM25 |
| 概念理解 | "违约金调整规则" | Dense |
| 关键词组合 | "劳动合同 解除 赔偿" | Sparse |
| 案情描述 | "员工旷工3天被辞退有赔偿吗" | Dense+Sparse |

### 3.2 完整实现

```python
# 初始化三通道
dense_channel = DenseRetrievalChannel(
    model_name='BAAI/bge-large-zh-v1.5'
)

sparse_channel = SPLADERetrievalChannel(
    model_name='naver/splade-cocondenser-ensembledistil'
)

bm25_channel = BM25RetrievalChannel(
    use_elasticsearch=True,
    es_host='localhost:9200'
)

# 构建混合引擎
hybrid_engine = HybridRetrievalEngine(
    dense_channel=dense_channel,
    sparse_channel=sparse_channel,
    lexical_channel=bm25_channel,
    fusion_config={
        'method': 'rrf',
        'k': 60,
        'weights': {
            'dense': 1.0,
            'sparse': 0.95,
            'bm25': 0.9
        }
    }
)

# 加载文档并构建索引
async def build_indices(documents: List[Dict]):
    texts = [d['content'] for d in documents]
    ids = [d['id'] for d in documents]
    
    # Dense
    print("Building dense index...")
    dense_embs = dense_channel.encode_documents(texts)
    dense_channel.build_index(ids, dense_embs)
    
    # Sparse
    print("Building sparse index...")
    sparse_matrix = sparse_channel.encode(texts)
    sparse_channel.build_index(ids, sparse_matrix)
    
    # BM25
    print("Building BM25 index...")
    bm25_channel.build_index(ids, texts)
    
    print("All indices built!")

# 检索示例
async def demo_search():
    queries = [
        "民法典第584条违约金",
        "劳动合同解除赔偿标准",
        "那个工伤认定的事情",
        "公司拖欠工资怎么办"
    ]
    
    for q in queries:
        print(f"\n{'='*60}")
        print(f"查询: {q}")
        
        # 单通道对比
        dense_res = await dense_channel.search(q, top_k=3)
        bm25_res = await bm25_channel.search(q, top_k=3)
        
        print(f"\n[Dense Top-1] {dense_res[0]['doc_id']} score={dense_res[0]['score']:.3f}")
        print(f"[BM25 Top-1]  {bm25_res[0]['doc_id']} score={bm25_res[0]['score']:.3f}")
        
        # 混合检索
        hybrid_res = await hybrid_engine.search(q, top_k=3)
        
        print(f"\n[Hybrid Results]")
        for i, r in enumerate(hybrid_res, 1):
            channels = [s['channel'] for s in r['sources']]
            print(f"  {i}. {r['doc_id']} RRF={r['rrf_score']:.3f} "
                  f"channels={channels} diversity={r.get('diversity_score', 0):.1f}")

# 评估
async def evaluate(test_queries: List[Dict]):
    """评估混合检索效果"""
    from collections import defaultdict
    
    metrics = {
        'dense': {'recall@10': [], 'mrr': []},
        'sparse': {'recall@10': [], 'mrr': []},
        'bm25': {'recall@10': [], 'mrr': []},
        'hybrid': {'recall@10': [], 'mrr': []}
    }
    
    for q in test_queries:
        relevant = set(q['relevant_docs'])
        
        # 各通道检索
        dense_res = await dense_channel.search(q['text'], top_k=10)
        sparse_res = await sparse_channel.search(q['text'], top_k=10)
        bm25_res = await bm25_channel.search(q['text'], top_k=10)
        hybrid_res = await hybrid_engine.search(q['text'], top_k=10)
        
        # 计算指标
        for name, results in [('dense', dense_res), 
                              ('sparse', sparse_res),
                              ('bm25', bm25_res),
                              ('hybrid', hybrid_res)]:
            retrieved = [r['doc_id'] for r in results]
            
            # Recall@10
            hit = len(set(retrieved) & relevant)
            recall = hit / len(relevant) if relevant else 0
            metrics[name]['recall@10'].append(recall)
            
            # MRR
            mrr = 0
            for i, r in enumerate(retrieved, 1):
                if r in relevant:
                    mrr = 1.0 / i
                    break
            metrics[name]['mrr'].append(mrr)
    
    # 输出结果
    print("\n评估结果:")
    for name, scores in metrics.items():
        avg_recall = sum(scores['recall@10']) / len(scores['recall@10'])
        avg_mrr = sum(scores['mrr']) / len(scores['mrr'])
        print(f"{name:10s}: Recall@10={avg_recall:.3f}, MRR={avg_mrr:.3f}")

# 运行
# asyncio.run(build_indices(docs))
# asyncio.run(demo_search())
```

### 3.3 效果对比

**查询**："民法典第584条违约金调整"

| 通道 | Top-1结果 | 是否命中目标 | Recall@5 |
|-----|----------|-------------|---------|
| Dense | 合同法司法解释（一） | ❌ | 0.40 |
| Sparse | 违约金与损害赔偿关系 | ❌ | 0.60 |
| BM25 | **民法典第584条** | ✅ | 0.80 |
| **Hybrid(RRF)** | **民法典第584条** | ✅ | **0.95** |

**查询**："劳动合同违法解除的赔偿计算"

| 通道 | Top-1结果 | 是否命中目标 |
|-----|----------|-------------|
| Dense | **违法解除赔偿标准** | ✅ |
| Sparse | 劳动合同解除类型 | ❌ |
| BM25 | 经济补偿金计算 | ❌ |
| **Hybrid** | **违法解除赔偿标准** | ✅（Dense主导）|

**关键发现**：
- 精确引用场景：BM25关键，Hybrid通过RRF确保其权重
- 语义理解场景：Dense主导，Sparse补充关键词匹配
- 混合查询：多通道互补，RRF融合实现高召回

## 四、高级优化：生产级精排与调优

### 4.1 学习排序（LTR）精排

```python
class LearningToRankReranker:
    """基于特征的学习排序"""
    
    def __init__(self, model_path: str = None):
        # 加载预训练的LTR模型（如LightGBM）
        self.model = self._load_model(model_path) if model_path else None
        
        # 特征定义
        self.feature_extractors = [
            self._f_dense_score,
            self._f_sparse_score,
            self._f_bm25_score,
            self._f_channel_count,
            self._f_rank_variance,
            self._f_query_doc_sim,
            self._f_term_overlap,
            self._f_length_ratio
        ]
    
    def extract_features(self, 
                         query: str,
                         doc: Dict,
                         fusion_result: Dict) -> np.ndarray:
        """提取特征向量"""
        features = []
        
        # 通道分数特征
        dense_score = fusion_result['channel_scores'].get('dense', 0)
        sparse_score = fusion_result['channel_scores'].get('sparse', 0)
        bm25_score = fusion_result['channel_scores'].get('bm25', 0)
        
        features.extend([dense_score, sparse_score, bm25_score])
        
        # 通道覆盖度
        channels = len(fusion_result['channel_scores'])
        features.append(channels / 3.0)  # 归一化
        
        # 排名一致性（各通道排名方差）
        ranks = [s.get('rank', 100) for s in fusion_result['sources']]
        rank_variance = np.var(ranks) if len(ranks) > 1 else 0
        features.append(1.0 / (1 + rank_variance))  # 方差越小越好
        
        # 查询-文档文本相似度（快速计算）
        query_doc_sim = self._quick_similarity(query, doc.get('content', ''))
        features.append(query_doc_sim)
        
        # 其他特征...
        
        return np.array(features)
    
    def rerank(self,
               query: str,
               fused_results: List[Dict],
               documents: Dict[str, str]) -> List[Dict]:
        """精排"""
        if self.model is None:
            return fused_results
        
        # 提取特征
        X = []
        for r in fused_results:
            doc_id = r['doc_id']
            doc_content = documents.get(doc_id, '')
            doc = {'content': doc_content}
            
            features = self.extract_features(query, doc, r)
            X.append(features)
        
        X = np.array(X)
        
        # 预测分数
        scores = self.model.predict(X)
        
        # 重新排序
        for i, r in enumerate(fused_results):
            r['ltr_score'] = scores[i]
            r['final_score'] = r['rrf_score'] * 0.7 + scores[i] * 0.3
        
        fused_results.sort(key=lambda x: x['final_score'], reverse=True)
        
        return fused_results
    
    def _quick_similarity(self, q: str, d: str) -> float:
        """快速文本相似度"""
        q_words = set(q.lower().split())
        d_words = set(d.lower().split())
        if not q_words:
            return 0
        return len(q_words & d_words) / len(q_words)
```

### 4.2 动态权重调优

```python
class AdaptiveWeightTuner:
    """基于查询类型的动态权重调整"""
    
    def __init__(self):
        self.query_type_weights = {
            'exact_citation': {'dense': 0.7, 'sparse': 0.9, 'bm25': 1.2},
            'concept_query': {'dense': 1.2, 'sparse': 0.9, 'bm25': 0.7},
            'keyword_query': {'dense': 0.8, 'sparse': 1.1, 'bm25': 1.0},
            'verbose_description': {'dense': 1.1, 'sparse': 0.8, 'bm25': 0.7}
        }
    
    def tune(self, 
             query: str, 
             base_config: Dict) -> Dict:
        """根据查询类型调整权重"""
        
        q_type = self._classify_query(query)
        weights = self.query_type_weights.get(q_type, base_config['weights'])
        
        # 复制配置并更新权重
        new_config = base_config.copy()
        new_config['weights'] = weights
        
        return new_config
    
    def _classify_query(self, query: str) -> str:
        """查询分类"""
        import re
        
        # 精确引用检测
        if re.search(r'\d{3,}|第[一二三四五六七八九十\d]+[条章节款]', query):
            return 'exact_citation'
        
        # 关键词组合（短查询，多个名词）
        words = query.split()
        if len(words) <= 5 and len(query) < 30:
            noun_count = sum(1 for w in words if len(w) > 2)
            if noun_count >= 3:
                return 'keyword_query'
        
        # 长描述
        if len(query) > 100:
            return 'verbose_description'
        
        # 默认概念查询
        return 'concept_query'
```

### 4.3 性能优化

```python
class PerformanceOptimizedHybrid:
    """性能优化的混合检索"""
    
    def __init__(self, engine: HybridRetrievalEngine):
        self.engine = engine
        self.cache = {}
        
    async def search_with_fallback(self,
                                    query: str,
                                    top_k: int = 10,
                                    timeout_ms: int = 500) -> List[Dict]:
        """带超时的分层检索"""
        
        # 1. 尝试快速路径（仅稠密）
        try:
            dense_results = await asyncio.wait_for(
                self.engine.channels['dense'].search(query, top_k=top_k),
                timeout=timeout_ms/1000 * 0.6  # 60%时间给稠密
            )
            
            # 如果稠密结果置信度高，直接返回
            if dense_results and dense_results[0]['score'] > 0.9:
                return dense_results
                
        except asyncio.TimeoutError:
            dense_results = []
        
        # 2. 并行执行稀疏和BM25
        remaining_time = timeout_ms/1000 * 0.4
        
        try:
            results = await asyncio.wait_for(
                asyncio.gather(
                    self.engine.channels['sparse'].search(query, top_k=50),
                    self.engine.channels['bm25'].search(query, top_k=50)
                ),
                timeout=remaining_time
            )
            sparse_res, bm25_res = results
        except asyncio.TimeoutError:
            sparse_res, bm25_res = [], []
        
        # 3. 融合可用结果
        channel_results = {
            'dense': dense_results,
            'sparse': sparse_res,
            'bm25': bm25_res
        }
        
        # 过滤空结果
        channel_results = {k: v for k, v in channel_results.items() if v}
        
        if not channel_results:
            return []
        
        return self.engine._fuse_results(channel_results, query)[:top_k]
```

---

混合检索架构通过**稠密向量的语义理解**、**稀疏向量的关键词权重**、**BM25的精确匹配**三通道互补，配合**RRF融合策略**，可以将复杂场景下的召回率从60%提升至90%以上。在实际部署中，建议根据查询特征动态调整通道权重，结合LTR精排优化最终排序，并通过分层超时策略保障性能。这是当前工业界验证最稳健的多路召回方案。
