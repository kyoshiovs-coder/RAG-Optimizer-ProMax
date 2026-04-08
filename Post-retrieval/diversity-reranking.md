
# 多样性排序：避免10个答案一模一样——MMR算法实战

## 一、背景：当"精准"遇上"单调"

### 1.1 一个真实的推荐失败案例

某银行理财顾问使用RAG系统为客户推荐产品：

**客户查询**："稳健型理财产品，3年期"

**系统返回Top 10**：
1. 工银理财·鑫得利365天（R2风险，3.2%）
2. 工银理财·鑫得利370天（R2风险，3.25%）
3. 工银理财·鑫得利360天（R2风险，3.18%）
4. 工银理财·鑫得利368天（R2风险，3.22%）
5. 工银理财·鑫得利362天（R2风险，3.19%）
6. 建信理财·安鑫180天（R2风险，3.15%）
7. 工银理财·鑫得利366天（R2风险，3.21%）
8. 工银理财·鑫得利364天（R2风险，3.17%）
9. 工银理财·鑫得利369天（R2风险，3.24%）
10. 工银理财·鑫得利361天（R2风险，3.2%）

**客户反馈**："这些不是同一个产品吗？"

**问题诊断**：
- 向量检索高度"精准"：同一系列产品Embedding几乎相同
- 缺乏多样性：风险等级、期限结构、发行机构单一
- 信息茧房：客户看不到其他选择（如债基、混合类、不同银行）

**业务损失**：客户认为系统"不专业"，转人工服务，转化率下降35%。

### 1.2 多样性缺失的三种形态

| 形态 | 表现 | 危害 |
|:---|:---|:---|
| **同质化重复** | 同一产品的细微变体霸榜 | 用户体验差，信任度下降 |
| **信息茧房** | 只返回单一维度的结果 | 用户看不到其他可能性 |
| **冗余覆盖** | 不同文档表达相同信息 | 浪费Token，信息密度低 |

**核心洞察**：检索系统需要平衡两个目标——**相关性**（Relevance）和**多样性**（Diversity）。

## 二、MMR算法：最大边际相关性的数学优雅

### 2.1 问题建模

给定：
- 查询 $q$
- 候选文档集合 $D = \{d_1, d_2, ..., d_n\}$
- 相关性函数 $\text{Rel}(q, d)$：文档与查询的相关性
- 相似度函数 $\text{Sim}(d_i, d_j)$：文档间的相似度

目标：选择子集 $S \subseteq D$，$|S| = k$，最大化：
$$\text{MMR} = \lambda \cdot \text{Rel}(q, d) - (1-\lambda) \cdot \max_{d_j \in S} \text{Sim}(d_i, d_j)$$

**直观理解**：
- $\lambda = 1$：只考虑相关性（传统排序）
- $\lambda = 0$：只考虑多样性（最分散）
- $\lambda \in (0.5, 0.7)$：平衡，通常最优

### 2.2 贪心算法流程

```
初始化：已选集合 S = ∅，候选集 R = D

循环 k 次：
    1. 对每个 d ∈ R \ S，计算：
       MMR(d) = λ·Rel(q,d) - (1-λ)·max_{s∈S} Sim(d,s)
       
    2. 选择 MMR 最大的 d* 加入 S
    
    3. R = R \ {d*}

返回 S
```

**关键特性**：每步选择"既相关又新颖"的文档，已选文档影响后续选择。

## 三、完整实现：MMR重排序引擎

### 3.1 核心算法

```python
import numpy as np
from typing import List, Dict, Tuple, Callable
from dataclasses import dataclass
from sklearn.metrics.pairwise import cosine_similarity

@dataclass
class Document:
    """文档对象"""
    id: str
    content: str
    embedding: np.ndarray      # 语义向量
    metadata: Dict             # 业务属性（风险等级、期限、机构等）
    relevance_score: float     # 与查询的相关性

class MMRReranker:
    """MMR多样性重排序器"""
    
    def __init__(self,
                 lambda_param: float = 0.6,    # 相关性权重
                 similarity_threshold: float = 0.85,  # 去重阈值
                 diversity_dimensions: List[str] = None):  # 多样性维度
        
        self.lambda_param = lambda_param
        self.sim_threshold = similarity_threshold
        self.diversity_dims = diversity_dimensions or ['category', 'source']
        
    def rerank(self,
               query: str,
               documents: List[Document],
               top_k: int = 10) -> List[Dict]:
        """
        MMR重排序主入口
        
        Args:
            query: 查询文本
            documents: 候选文档列表（已按相关性预排序）
            top_k: 返回数量
        
        Returns:
            重排序后的文档列表，含MMR分数和多样性信息
        """
        if not documents:
            return []
        
        # 提取向量矩阵
        doc_vectors = np.stack([d.embedding for d in documents])
        
        # 预计算文档间相似度矩阵
        sim_matrix = cosine_similarity(doc_vectors)
        
        # MMR贪心选择
        selected_indices = self._mmr_select(
            documents=documents,
            sim_matrix=sim_matrix,
            top_k=top_k
        )
        
        # 组装结果
        results = []
        for rank, idx in enumerate(selected_indices, 1):
            doc = documents[idx]
            
            # 计算选中后的多样性贡献
            diversity_contrib = self._calculate_diversity_contribution(
                doc, results, sim_matrix, idx
            )
            
            results.append({
                'id': doc.id,
                'content': doc.content[:200],
                'original_rank': idx + 1,
                'relevance_score': doc.relevance_score,
                'mmr_score': self._compute_mmr_score(
                    doc, results, sim_matrix, idx
                ),
                'diversity_contribution': diversity_contrib,
                'metadata': doc.metadata,
                'rank': rank
            })
        
        return results
    
    def _mmr_select(self,
                    documents: List[Document],
                    sim_matrix: np.ndarray,
                    top_k: int) -> List[int]:
        """MMR贪心选择算法"""
        
        n = len(documents)
        selected = []
        remaining = set(range(n))
        
        # 第一步：选相关性最高的
        first_idx = max(remaining, key=lambda i: documents[i].relevance_score)
        selected.append(first_idx)
        remaining.remove(first_idx)
        
        # 后续步骤：MMR选择
        while len(selected) < top_k and remaining:
            best_mmr = -float('inf')
            best_idx = None
            
            for idx in remaining:
                doc = documents[idx]
                
                # 相关性项
                rel_score = doc.relevance_score
                
                # 多样性项：与已选文档的最大相似度
                max_sim = max(sim_matrix[idx][s] for s in selected)
                
                # MMR分数
                mmr_score = (self.lambda_param * rel_score - 
                            (1 - self.lambda_param) * max_sim)
                
                if mmr_score > best_mmr:
                    best_mmr = mmr_score
                    best_idx = idx
            
            if best_idx is not None:
                selected.append(best_idx)
                remaining.remove(best_idx)
        
        return selected
    
    def _compute_mmr_score(self,
                          doc: Document,
                          selected_results: List[Dict],
                          sim_matrix: np.ndarray,
                          doc_idx: int) -> float:
        """计算文档的MMR分数"""
        
        # 相关性
        rel = doc.relevance_score
        
        # 如果已有选中，计算最大相似度
        if not selected_results:
            diversity_penalty = 0
        else:
            # 从已选结果中找到对应的原始索引
            selected_indices = [r['original_rank'] - 1 for r in selected_results]
            max_sim = max(sim_matrix[doc_idx][s] for s in selected_indices)
            diversity_penalty = max_sim
        
        mmr = self.lambda_param * rel - (1 - self.lambda_param) * diversity_penalty
        
        return mmr
    
    def _calculate_diversity_contribution(self,
                                          doc: Document,
                                          selected: List[Dict],
                                          sim_matrix: np.ndarray,
                                          doc_idx: int) -> Dict:
        """计算文档的多样性贡献（多维度）"""
        
        contribution = {
            'semantic_diversity': 0,      # 语义新颖度
            'attribute_diversity': {}       # 属性维度新颖度
        }
        
        if not selected:
            # 第一个文档，多样性贡献最大
            contribution['semantic_diversity'] = 1.0
            for dim in self.diversity_dims:
                contribution['attribute_diversity'][dim] = 1.0
        else:
            # 语义新颖度：1 - 平均相似度
            selected_indices = [r['original_rank'] - 1 for r in selected]
            avg_sim = np.mean([sim_matrix[doc_idx][s] for s in selected_indices])
            contribution['semantic_diversity'] = 1 - avg_sim
            
            # 属性维度新颖度
            for dim in self.diversity_dims:
                existing_values = set(r['metadata'].get(dim) for r in selected)
                current_value = doc.metadata.get(dim)
                
                if current_value not in existing_values:
                    contribution['attribute_diversity'][dim] = 1.0  # 全新值
                else:
                    contribution['attribute_diversity'][dim] = 0.0   # 重复值
        
        return contribution

class MultiDimensionalDiversityReranker(MMRReranker):
    """多维多样性重排序器（MMR扩展）"""
    
    def __init__(self,
                 lambda_rel: float = 0.5,      # 相关性权重
                 lambda_div: float = 0.3,      # 语义多样性权重
                 lambda_attr: float = 0.2,     # 属性多样性权重
                 attribute_weights: Dict[str, float] = None):
        
        super().__init__(lambda_param=lambda_rel)
        
        self.lambda_rel = lambda_rel
        self.lambda_div = lambda_div
        self.lambda_attr = lambda_attr
        self.attr_weights = attribute_weights or {}
    
    def _compute_score(self,
                      doc: Document,
                      selected: List[Document],
                      sim_matrix: np.ndarray,
                      doc_idx: int) -> float:
        """多维评分函数"""
        
        # 相关性
        rel = doc.relevance_score
        
        # 语义多样性（MMR中的惩罚项）
        if not selected:
            div_semantic = 1.0
        else:
            selected_indices = [i for i, d in enumerate(selected)]
            max_sim = max(sim_matrix[doc_idx][s] for s in selected_indices)
            div_semantic = 1 - max_sim
        
        # 属性多样性
        div_attr = 0
        if not selected:
            div_attr = 1.0
        else:
            for dim, weight in self.attr_weights.items():
                existing = set(d.metadata.get(dim) for d in selected)
                if doc.metadata.get(dim) not in existing:
                    div_attr += weight
        
        # 综合分数
        score = (self.lambda_rel * rel + 
                self.lambda_div * div_semantic + 
                self.lambda_attr * div_attr)
        
        return score
```

### 3.2 金融场景实战：理财产品推荐

```python
# 模拟理财产品数据
def create_financial_products():
    """创建模拟理财产品"""
    
    products = [
        # 工银理财系列（高相似度）
        Document(
            id="GY001",
            content="工银理财·鑫得利365天，R2低风险，业绩基准3.2%，固收类",
            embedding=np.array([0.9, 0.1, 0.3, 0.8]),  # 模拟向量
            metadata={
                "issuer": "工银理财",
                "risk_level": "R2",
                "category": "固收类",
                "duration_days": 365,
                "expected_return": 3.2
            },
            relevance_score=0.95
        ),
        Document(
            id="GY002",
            content="工银理财·鑫得利370天，R2低风险，业绩基准3.25%，固收类",
            embedding=np.array([0.91, 0.11, 0.31, 0.81]),
            metadata={
                "issuer": "工银理财",
                "risk_level": "R2",
                "category": "固收类",
                "duration_days": 370,
                "expected_return": 3.25
            },
            relevance_score=0.94
        ),
        # 建信理财（不同机构）
        Document(
            id="JX001",
            content="建信理财·安鑫180天，R2低风险，业绩基准3.15%，固收类",
            embedding=np.array([0.7, 0.2, 0.4, 0.6]),
            metadata={
                "issuer": "建信理财",
                "risk_level": "R2",
                "category": "固收类",
                "duration_days": 180,
                "expected_return": 3.15
            },
            relevance_score=0.88
        ),
        # 混合类（不同类别）
        Document(
            id="ZG001",
            content="中银理财·稳富混合1年，R3中风险，业绩基准4.5%，固收+权益",
            embedding=np.array([0.5, 0.6, 0.7, 0.4]),
            metadata={
                "issuer": "中银理财",
                "risk_level": "R3",
                "category": "混合类",
                "duration_days": 365,
                "expected_return": 4.5
            },
            relevance_score=0.85
        ),
        # 债基（不同资产类型）
        Document(
            id="YD001",
            content="易方达纯债债券A，R2低风险，近一年收益4.2%，债券基金",
            embedding=np.array([0.6, 0.3, 0.5, 0.5]),
            metadata={
                "issuer": "易方达",
                "risk_level": "R2",
                "category": "债券基金",
                "duration_days": 730,  # 开放式
                "expected_return": 4.2
            },
            relevance_score=0.82
        ),
        # 更多产品...
    ]
    
    return products

# 运行MMR重排序
def demo_mmr_reranking():
    """MMR重排序演示"""
    
    products = create_financial_products()
    
    print("=" * 60)
    print("原始排序（仅按相关性）:")
    for i, p in enumerate(products[:5], 1):
        print(f"{i}. {p.id} | {p.metadata['issuer']} | "
              f"{p.metadata['category']} | 相关度: {p.relevance_score:.2f}")
    
    # MMR重排序
    reranker = MMRReranker(
        lambda_param=0.6,  # 相关性60%，多样性40%
        diversity_dimensions=['issuer', 'category', 'risk_level']
    )
    
    results = reranker.rerank(
        query="稳健型理财产品，3年期",
        documents=products,
        top_k=5
    )
    
    print("\n" + "=" * 60)
    print("MMR重排序后（λ=0.6）:")
    for r in results:
        print(f"{r['rank']}. {r['id']} | "
              f"相关度: {r['relevance_score']:.2f} | "
              f"MMR: {r['mmr_score']:.3f} | "
              f"多样性: {r['diversity_contribution']['semantic_diversity']:.2f}")
        print(f"   机构: {r['metadata']['issuer']} | "
              f"类别: {r['metadata']['category']} | "
              f"风险: {r['metadata']['risk_level']}")

if __name__ == "__main__":
    demo_mmr_reranking()
```

### 3.3 电子数据存证场景：通话记录检索

```python
class EvidenceChainDiversityReranker:
    """电子数据存证场景的多样性排序"""
    
    def __init__(self):
        self.reranker = MultiDimensionalDiversityReranker(
            lambda_rel=0.5,
            lambda_div=0.3,
            lambda_attr=0.2,
            attribute_weights={
                'call_type': 0.3,      # 通话类型（呼入/呼出/未接）
                'time_period': 0.3,    # 时间段（工作/非工作）
                'counterparty_type': 0.25,  # 对方类型（客户/内部/未知）
                'duration_bucket': 0.15  # 时长分段
            }
        )
    
    def rerank_evidence(self,
                        query: str,
                        call_records: List[Document],
                        top_k: int = 10) -> List[Dict]:
        """
        通话记录多样性排序
        
        避免：同一时间段、同一对方、同类型的记录扎堆
        """
        
        # 增强元数据
        for record in call_records:
            # 时间分段
            hour = record.metadata.get('hour', 12)
            if 9 <= hour <= 18:
                record.metadata['time_period'] = 'work_hours'
            else:
                record.metadata['time_period'] = 'off_hours'
            
            # 时长分段
            duration = record.metadata.get('duration_seconds', 0)
            if duration < 60:
                record.metadata['duration_bucket'] = 'short'
            elif duration < 300:
                record.metadata['duration_bucket'] = 'medium'
            else:
                record.metadata['duration_bucket'] = 'long'
        
        return self.reranker.rerank(query, call_records, top_k)

# 示例：客户投诉调查
"""
查询："客户投诉服务态度的通话"

原始结果（相关性排序）：
1. 2024-01-15 14:32 呼入 客户A 投诉服务态度 5分钟
2. 2024-01-15 15:10 呼入 客户A 投诉服务态度 8分钟  
3. 2024-01-15 16:45 呼入 客户A 询问业务 2分钟
4. 2024-01-16 09:20 呼入 客户B 投诉服务态度 6分钟
...

MMR重排序后：
1. 2024-01-15 14:32 客户A 投诉服务态度（工作时段）
2. 2024-01-16 20:15 客户C 投诉服务态度（非工作时段，不同客户）
3. 2024-01-14 11:00 客户A 业务咨询（同一客户，不同场景对比）
4. 2024-01-17 14:00 客户B 投诉处理结果（跟进记录）
5. 2024-01-18 10:30 内部质检 服务态度培训（内部视角）
"""
```

### 3.4 信贷场景：多源信息融合

```python
class CreditDecisionDiversityReranker:
    """信贷决策辅助的多样性排序"""
    
    def __init__(self):
        self.source_weights = {
            'credit_report': 0.35,    # 征信报告
            'bank_statement': 0.30,   # 银行流水
            'contract': 0.20,         # 合同信息
            'guarantee': 0.15         # 担保信息
        }
    
    def rerank_for_decision(self,
                            query: str,
                            evidence_docs: List[Document],
                            decision_focus: str = "comprehensive") -> List[Dict]:
        """
        信贷决策证据排序
        
        确保：不同信息来源、不同风险维度、不同时间段的证据都被覆盖
        """
        
        # 根据决策焦点调整多样性维度
        if decision_focus == "repayment_ability":
            dimensions = ['source_type', 'time_period', 'income_stability']
        elif decision_focus == "fraud_risk":
            dimensions = ['source_type', 'anomaly_type', 'counterparty_risk']
        else:  # comprehensive
            dimensions = ['source_type', 'risk_dimension', 'verification_status']
        
        # 增强文档元数据
        for doc in evidence_docs:
            # 自动识别来源类型
            content = doc.content.lower()
            if '征信' in content or '逾期' in content:
                doc.metadata['source_type'] = 'credit_report'
            elif '流水' in content or '收入' in content:
                doc.metadata['source_type'] = 'bank_statement'
            elif '合同' in content or '协议' in content:
                doc.metadata['source_type'] = 'contract'
            
            # 风险维度
            if '逾期' in content or '违约' in content:
                doc.metadata['risk_dimension'] = 'credit_risk'
            elif '负债' in content or '杠杆' in content:
                doc.metadata['risk_dimension'] = 'debt_risk'
            elif '收入' in content or '现金流' in content:
                doc.metadata['risk_dimension'] = 'liquidity_risk'
        
        reranker = MMRReranker(
            lambda_param=0.55,
            diversity_dimensions=dimensions
        )
        
        return reranker.rerank(query, evidence_docs, top_k=10)

# 示例输出
"""
查询："评估客户还款能力"

MMR重排序后Top 5：
1. [征信报告] 近24个月还款记录，无逾期（信用风险维度）
2. [银行流水] 近6个月工资流水，月收入稳定2.5万（流动性维度）
3. [合同信息] 劳动合同，无固定期限，已入职3年（稳定性维度）
4. [担保信息] 房产抵押评估，市值300万，抵押率50%（担保维度）
5. [征信报告] 当前负债明细，信用卡使用率30%（债务风险维度）

vs 纯相关性排序（可能全是征信报告的不同段落）
"""
```

## 四、高级策略与调优

### 4.1 动态λ调整

```python
class AdaptiveLambdaReranker:
    """根据查询特征动态调整λ参数"""
    
    def select_lambda(self, query: str, candidates: List[Document]) -> float:
        """动态选择λ值"""
        
        # 探索型查询：高多样性
        exploratory_keywords = ['推荐', '有哪些', '对比', '比较']
        if any(kw in query for kw in exploratory_keywords):
            return 0.4  # 多样性权重更高
        
        # 精确查找型查询：高相关性
        precise_patterns = ['编号', 'ID', '具体', '准确']
        if any(pat in query for pat in precise_patterns):
            return 0.8  # 相关性权重更高
        
        # 候选同质化程度高：强制多样性
        top_sim_matrix = self._compute_top_k_similarity(candidates[:10])
        avg_sim = np.mean(top_sim_matrix)
        if avg_sim > 0.9:  # 高度同质化
            return 0.3  # 强制高多样性
        
        return 0.6  # 默认平衡
```

### 4.2 多样性评估指标

```python
class DiversityEvaluator:
    """多样性效果评估"""
    
 def evaluate(self, results: List[Dict]) -> Dict:
        """计算多样性指标"""
        
        metrics = {}
        
        # 1. 语义多样性：Embedding平均距离
        embeddings = [r['embedding'] for r in results]
        if len(embeddings) > 1:
            sim_matrix = cosine_similarity(embeddings)
            # 上三角平均相似度
            upper_tri = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]
            metrics['semantic_diversity'] = 1 - np.mean(upper_tri)
        
        # 2. 属性覆盖率
        for dim in ['issuer', 'category', 'risk_level']:
            unique_values = set(r['metadata'].get(dim) for r in results)
            metrics[f'{dim}_coverage'] = len(unique_values) / len(results)
        
        # 3. 用户满意度代理指标
        # 假设：前3个结果覆盖≥3个不同机构 = 满意度高
        top3_issuers = set(r['metadata'].get('issuer') for r in results[:3])
        metrics['satisfaction_proxy'] = len(top3_issuers) >= 2
        
        return metrics
```

---

MMR算法通过数学优雅的贪心策略，在相关性和多样性之间找到动态平衡。在金融理财推荐、电子数据存证检索、信贷决策辅助等场景中，能有效打破"信息茧房"，提升用户信任度和决策质量。关键调参在于λ值的场景适配，以及多样性维度的业务化定义。
