# 数据增强与合成：让沉默的数据被"搜到"

## 一、背景：为什么你的RAG"搜不到"正确答案

### 1.1 一个令人困惑的现象

某企业知识库上线RAG系统后，出现了一个诡异现象：

**用户提问**："如何申请远程办公？"

**知识库原文**：
> "员工因特殊情况需在家办公的，应提前通过OA系统提交《弹性工作申请表》，经直属上级审批后报HR备案。审批通过后，需确保工作时间内保持企业微信在线，并按时参加线上会议。"

**检索结果**：系统返回了关于"考勤制度"、"加班申请"、"会议室预定"的文档，唯独没返回这条最相关的政策。

**人工检查**：该文档确实存在于知识库中，Embedding模型也没问题，向量相似度计算正常。

**问题根源**：**语义鸿沟**——用户问的是"远程办公"，文档写的是"在家办公"、"弹性工作"，它们的向量表征在空间中相距甚远。

### 1.2 语义鸿沟的三种形态

| 形态 | 用户查询 | 文档表述 | 相似度 | 结果 |
|-----|---------|---------|--------|------|
| **同义不同词** | 远程办公 | 在家办公、弹性工作、分布式办公 | 0.62 | ❌ 漏召回 |
| **概括与具体** | 怎么请假 | 年假申请流程、病假提交方式 | 0.58 | ❌ 漏召回 |
| **问题与答案** | 违约金多少 | 违约方应支付合同金额20%的违约金 | 0.55 | ❌ 漏召回 |
| **场景与条款** | 被辞退了怎么办 | 用人单位解除劳动合同的情形及补偿标准 | 0.48 | ❌ 漏召回 |

传统RAG假设"用户查询与相关文档的向量相似度最高"，但现实中：
- 用户不知道文档用什么术语
- 用户用口语提问，文档用书面语写作
- 用户问的是问题，文档写的是陈述

**数据增强与合成**的核心思想是：**在索引阶段，用LLM预先生成用户可能的各种问法，构建"查询-文档"的多对多映射，让无论用户怎么问，都能命中目标**。

## 二、核心策略：四维增强体系

### 2.1 策略矩阵

| 增强维度 | 生成内容 | 解决什么问题 | 技术复杂度 |
|---------|---------|-----------|-----------|
| **FAQ合成** | 为每个chunk生成3-5个常见问题 | 问题-答案对齐 | ⭐⭐ |
| **假设查询** | 生成用户可能搜索的变体查询 | 同义词、口语化 | ⭐⭐⭐ |
| **反向摘要** | 用一句话概括chunk核心 | 检索时快速匹配 | ⭐ |
| **多视角重写** | 从不同用户角色/场景重写 | 覆盖多样化需求 | ⭐⭐⭐⭐ |

### 2.2 完整架构

```
原始文档
    ↓
┌─────────────────────────────────────────┐
│           基础切分（语义分块）            │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  增强流水线（LLM驱动的多维度生成）        │
│  ├─ FAQ生成器：chunk → [Q1, Q2, Q3...] │
│  ├─ 查询变体器：chunk → [query_v1, v2...]│
│  ├─ 摘要生成器：chunk → one_line_summary │
│  └─ 视角重写器：chunk → [员工版, HR版, 高管版]│
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  索引构建（多向量+多字段）               │
│  ├─ 主向量：chunk原文                   │
│  ├─ FAQ向量：问题集合                   │
│  ├─ 查询向量：变体查询                  │
│  └─ 稀疏索引：关键词、实体、标签         │
└─────────────────────────────────────────┘
    ↓
用户Query → 多路召回（原文+FAQ+查询+关键词）→ 重排序 → 生成
```

## 三、完整实现：工业级数据增强系统

### 3.1 核心引擎

```python
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, AsyncIterator
from enum import Enum
import asyncio
import json
from collections import defaultdict

class EnhancementType(Enum):
    FAQ = "faq"                    # 常见问题
    HYPOTHETICAL_QUERY = "hypothetical_query"  # 假设查询
    BACKWARD_SUMMARY = "backward_summary"      # 反向摘要
    MULTI_VIEW = "multi_view"      # 多视角重写
    ENTITY_TAGS = "entity_tags"    # 实体标签
    KEYWORD_EXPANSION = "keyword_expansion"  # 关键词扩展

@dataclass
class EnhancedChunk:
    """增强后的数据块"""
    original_id: str
    original_content: str
    
    # 各类增强内容
    faqs: List[Dict] = field(default_factory=list)  # [{'question': '...', 'answer': '...'}]
    hypothetical_queries: List[str] = field(default_factory=list)
    backward_summary: str = ""
    multi_views: Dict[str, str] = field(default_factory=dict)  # {'employee': '...', 'manager': '...'}
    entity_tags: List[str] = field(default_factory=list)
    keyword_expansion: List[str] = field(default_factory=list)
    
    # 元数据
    enhancement_metadata: Dict = field(default_factory=dict)
    
    def get_all_searchable_texts(self) -> List[Dict]:
        """获取所有可用于检索的文本形式"""
        texts = []
        
        # 1. 原文
        texts.append({
            'type': 'original',
            'text': self.original_content,
            'weight': 1.0
        })
        
        # 2. FAQ问题
        for i, faq in enumerate(self.faqs):
            texts.append({
                'type': 'faq_question',
                'text': faq['question'],
                'reference_answer': faq['answer'],
                'weight': 0.9
            })
        
        # 3. 假设查询
        for i, query in enumerate(self.hypothetical_queries):
            texts.append({
                'type': 'hypothetical_query',
                'text': query,
                'weight': 0.85
            })
        
        # 4. 反向摘要
        if self.backward_summary:
            texts.append({
                'type': 'backward_summary',
                'text': self.backward_summary,
                'weight': 0.8
            })
        
        # 5. 多视角
        for perspective, content in self.multi_views.items():
            texts.append({
                'type': f'multi_view_{perspective}',
                'text': content,
                'weight': 0.75
            })
        
        return texts

class LLMEnhancer:
    """基于LLM的数据增强器"""
    
    def __init__(self, 
                 llm_client,
                 max_concurrency: int = 5,
                 cache_enabled: bool = True):
        self.llm = llm_client
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.cache = {} if cache_enabled else None
        self.generation_stats = defaultdict(int)
        
    async def enhance_chunk(self, 
                           chunk: Dict,
                           enhancement_types: List[EnhancementType] = None) -> EnhancedChunk:
        """对单个chunk执行全维度增强"""
        
        if enhancement_types is None:
            enhancement_types = list(EnhancementType)
        
        original_id = chunk.get('id', self._hash_content(chunk['content']))
        original_content = chunk['content']
        
        # 检查缓存
        cache_key = f"{original_id}:{','.join(t.value for t in enhancement_types)}"
        if self.cache and cache_key in self.cache:
            return self.cache[cache_key]
        
        # 并行执行各类增强
        tasks = []
        for etype in enhancement_types:
            if etype == EnhancementType.FAQ:
                tasks.append(self._generate_faqs(original_content))
            elif etype == EnhancementType.HYPOTHETICAL_QUERY:
                tasks.append(self._generate_hypothetical_queries(original_content))
            elif etype == EnhancementType.BACKWARD_SUMMARY:
                tasks.append(self._generate_backward_summary(original_content))
            elif etype == EnhancementType.MULTI_VIEW:
                tasks.append(self._generate_multi_views(original_content, chunk.get('metadata', {})))
            elif etype == EnhancementType.ENTITY_TAGS:
                tasks.append(self._extract_entities(original_content))
            elif etype == EnhancementType.KEYWORD_EXPANSION:
                tasks.append(self._expand_keywords(original_content))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 组装结果
        enhanced = EnhancedChunk(
            original_id=original_id,
            original_content=original_content
        )
        
        for etype, result in zip(enhancement_types, results):
            if isinstance(result, Exception):
                print(f"Enhancement failed for {etype}: {result}")
                continue
                
            if etype == EnhancementType.FAQ:
                enhanced.faqs = result
            elif etype == EnhancementType.HYPOTHETICAL_QUERY:
                enhanced.hypothetical_queries = result
            elif etype == EnhancementType.BACKWARD_SUMMARY:
                enhanced.backward_summary = result
            elif etype == EnhancementType.MULTI_VIEW:
                enhanced.multi_views = result
            elif etype == EnhancementType.ENTITY_TAGS:
                enhanced.entity_tags = result
            elif etype == EnhancementType.KEYWORD_EXPANSION:
                enhanced.keyword_expansion = result
        
        # 记录统计
        self.generation_stats['total_chunks'] += 1
        self.generation_stats['total_faqs'] += len(enhanced.faqs)
        self.generation_stats['total_queries'] += len(enhanced.hypothetical_queries)
        
        # 缓存
        if self.cache is not None:
            self.cache[cache_key] = enhanced
            
        return enhanced
    
    async def _generate_faqs(self, content: str, num_faqs: int = 3) -> List[Dict]:
        """生成FAQ对"""
        
        prompt = f"""基于以下文档内容，生成{num_faqs}个用户可能问的问题及答案。

要求：
1. 问题要覆盖不同角度（是什么、为什么、怎么做、有什么限制）
2. 问题要用用户视角，口语化
3. 答案必须严格基于文档内容，不添加外部信息
4. 输出JSON格式

文档内容：
{content[:1500]}

输出格式：
[
  {{"question": "...", "answer": "...", "question_type": "what/how/why/condition"}},
  ...
]"""

        async with self.semaphore:
            response = await self.llm.acomplete(prompt, temperature=0.7)
            
        try:
            faqs = json.loads(response)
            # 验证格式
            for faq in faqs:
                if 'question' not in faq or 'answer' not in faq:
                    raise ValueError("Invalid FAQ format")
            return faqs[:num_faqs]
        except json.JSONDecodeError:
            #  fallback：用正则提取
            return self._parse_faq_from_text(response, content)
    
    async def _generate_hypothetical_queries(self, 
                                             content: str, 
                                             num_queries: int = 5) -> List[str]:
        """生成假设查询（用户搜索的多种表述）"""
        
        prompt = f"""用户可能会用哪些不同的说法来搜索以下信息？生成{num_queries}个不同的查询。

要求：
1. 包含同义词替换（如"远程办公"→"在家工作"、"弹性工作"）
2. 包含不同详细程度（概括 vs 具体）
3. 包含不同句式（疑问句、短语、关键词组合）
4. 包含口语化和书面语
5. 只输出查询文本，每行一个

文档内容：
{content[:1500]}

示例输出：
远程办公怎么申请？
在家工作的流程是什么？
弹性工作制度
如何申请居家办公？
WFH政策"""

        async with self.semaphore:
            response = await self.llm.acomplete(prompt, temperature=0.8)
        
        queries = [q.strip() for q in response.strip().split('\n') if q.strip()]
        return queries[:num_queries]
    
    async def _generate_backward_summary(self, content: str) -> str:
        """生成反向摘要（一句话概括核心）"""
        
        prompt = f"""用一句话概括以下文档的核心内容，要求：
1. 包含关键实体（制度名称、数字、条件）
2. 使用概括性词汇，便于检索匹配
3. 不超过50字

文档内容：
{content[:1000]}

输出格式：该文档主要介绍了[核心内容]，适用于[适用对象]，关键要求包括[关键要点]。"""

        async with self.semaphore:
            response = await self.llm.acomplete(prompt, temperature=0.3)
        
        return response.strip()[:100]
    
    async def _generate_multi_views(self, 
                                    content: str, 
                                    metadata: Dict) -> Dict[str, str]:
        """从不同视角重写内容"""
        
        # 自动推断相关视角
        perspectives = self._infer_perspectives(metadata, content)
        
        views = {}
        for perspective in perspectives:
            prompt = f"""从{perspective}的视角，重写以下文档内容。保持事实准确，但调整侧重点和表述方式。

原文：
{content[:1000]}

要求：
- {perspective}最关心什么？
- 他们需要知道什么行动要点？
- 用他们能理解的语言表述

输出："""

            async with self.semaphore:
                response = await self.llm.acomplete(prompt, temperature=0.5)
            views[perspective] = response.strip()
        
        return views
    
    def _infer_perspectives(self, metadata: Dict, content: str) -> List[str]:
        """推断适用的视角"""
        perspectives = ['普通员工']  # 默认
        
        if any(kw in content for kw in ['审批', '审核', '报备', '上级']):
            perspectives.extend(['管理者', 'HR'])
        if any(kw in content for kw in ['预算', '成本', '采购', '供应商']):
            perspectives.extend(['财务', '采购部门'])
        if any(kw in content for kw in ['技术', '系统', '开发', '部署']):
            perspectives.extend(['技术人员', '管理员'])
        if any(kw in content for kw in ['合规', '风险', '法律']):
            perspectives.extend(['法务', '合规部门'])
            
        return list(set(perspectives))[:4]  # 最多4个视角
    
    async def _extract_entities(self, content: str) -> List[str]:
        """提取关键实体"""
        
        prompt = f"""从以下文档中提取关键实体（制度名、部门、数字、条件、状态等），用于标签索引。

文档：
{content[:800]}

输出JSON格式：{{"entities": ["实体1", "实体2", ...]}}"""

        async with self.semaphore:
            response = await self.llm.acomplete(prompt, temperature=0.1)
        
        try:
            data = json.loads(response)
            return data.get('entities', [])[:10]
        except:
            # fallback：简单提取
            import re
            # 引号内容、数字+单位、大写术语
            entities = []
            entities.extend(re.findall(r'《(.+?)》', content))
            entities.extend(re.findall(r'\d+\.?\d*\s*[个位条款项元年月日]%?', content))
            return list(set(entities))[:10]
    
    async def _expand_keywords(self, content: str) -> List[str]:
        """关键词扩展（同义词、上下位词）"""
        
        prompt = f"""为以下文档扩展关键词，包括：
1. 同义词（如"请假"="休假"="事假"）
2. 上位词（如"年假"→"假期"→"福利"）
3. 下位词（如"远程办公"→"居家办公"、"分布式办公"）
4. 相关场景词

文档：
{content[:800]}

输出：关键词1, 关键词2, ...（逗号分隔）"""

        async with self.semaphore:
            response = await self.llm.acomplete(prompt, temperature=0.5)
        
        keywords = [k.strip() for k in response.split(',') if k.strip()]
        return keywords[:15]
    
    def _hash_content(self, content: str) -> str:
        """生成内容哈希作为ID"""
        import hashlib
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def _parse_faq_from_text(self, text: str, original: str) -> List[Dict]:
        """从非结构化文本解析FAQ"""
        import re
        
        faqs = []
        # 尝试匹配 Q: ... A: ... 模式
        pattern = r'[Q问]:\s*(.+?)\s*[A答]:\s*(.+?)(?=[Q问]:|$)'
        matches = re.findall(pattern, text, re.DOTALL)
        
        for q, a in matches:
            faqs.append({
                'question': q.strip(),
                'answer': a.strip(),
                'question_type': 'unknown'
            })
        
        # 如果解析失败，生成一个默认FAQ
        if not faqs:
            faqs = [{
                'question': f"该文档的主要内容是什么？",
                'answer': original[:200],
                'question_type': 'what'
            }]
        
        return faqs

class DataAugmentationPipeline:
    """完整的数据增强流水线"""
    
    def __init__(self,
                 llm_client,
                 embedding_model,
                 vector_store,
                 config: Dict = None):
        
        self.enhancer = LLMEnhancer(llm_client)
        self.embedder = embedding_model
        self.vector_store = vector_store
        self.config = config or {
            'batch_size': 10,
            'enhancement_types': list(EnhancementType),
            'index_original': True,
            'index_faqs': True,
            'index_hypothetical': True,
            'index_multi_view': False,  # 可选，数据量大时关闭
            'deduplicate_queries': True
        }
        
    async def process_documents(self, chunks: List[Dict]) -> Dict:
        """处理文档批次"""
        
        all_enhanced = []
        
        # 批量增强
        for i in range(0, len(chunks), self.config['batch_size']):
            batch = chunks[i:i + self.config['batch_size']]
            
            # 并行处理批次
            tasks = [self.enhancer.enhance_chunk(c, self.config['enhancement_types']) 
                    for c in batch]
            enhanced_batch = await asyncio.gather(*tasks)
            
            all_enhanced.extend(enhanced_batch)
            print(f"Processed {min(i + self.config['batch_size'], len(chunks))}/{len(chunks)}")
        
        # 去重（针对假设查询）
        if self.config['deduplicate_queries']:
            all_enhanced = self._deduplicate_queries(all_enhanced)
        
        # 构建索引
        index_stats = await self._build_index(all_enhanced)
        
        return {
            'enhanced_chunks': all_enhanced,
            'index_stats': index_stats,
            'generation_stats': dict(self.enhancer.generation_stats)
        }
    
    def _deduplicate_queries(self, enhanced_chunks: List[EnhancedChunk]) -> List[EnhancedChunk]:
        """全局去重假设查询"""
        seen_queries = set()
        
        for chunk in enhanced_chunks:
            unique_queries = []
            for q in chunk.hypothetical_queries:
                q_normalized = q.lower().replace(' ', '')
                if q_normalized not in seen_queries:
                    seen_queries.add(q_normalized)
                    unique_queries.append(q)
            chunk.hypothetical_queries = unique_queries
        
        return enhanced_chunks
    
    async def _build_index(self, enhanced_chunks: List[EnhancedChunk]) -> Dict:
        """构建多维度索引"""
        
        documents = []
        embeddings = []
        metadatas = []
        ids = []
        
        for chunk in enhanced_chunks:
            searchable_texts = chunk.get_all_searchable_texts()
            
            for item in searchable_texts:
                # 生成embedding
                vector = self.embedder.encode(item['text'])
                
                doc_id = f"{chunk.original_id}_{item['type']}"
                
                documents.append(item['text'])
                embeddings.append(vector)
                metadatas.append({
                    'original_id': chunk.original_id,
                    'original_content': chunk.original_content,
                    'search_type': item['type'],
                    'weight': item['weight'],
                    'reference_answer': item.get('reference_answer', ''),
                    'entity_tags': chunk.entity_tags,
                    'keywords': chunk.keyword_expansion
                })
                ids.append(doc_id)
        
        # 批量写入向量库
        self.vector_store.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        
        return {
            'total_vectors': len(ids),
            'original_chunks': len(enhanced_chunks),
            'expansion_ratio': len(ids) / len(enhanced_chunks)
        }
```

### 3.2 增强检索引擎

```python
class AugmentedRetriever:
    """利用增强数据的检索引擎"""
    
    def __init__(self, 
                 vector_store,
                 embedding_model,
                 config: Dict = None):
        self.vector_store = vector_store
        self.embedder = embedding_model
        self.config = config or {
            'multi_stage_retrieval': True,
            'faq_boost': 1.2,           # FAQ匹配加分
            'original_weight': 1.0,
            'query_expansion_weight': 0.9,
            'rerank_top_k': 20
        }
    
    async def retrieve(self, 
                       query: str,
                       filters: Dict = None,
                       top_k: int = 5) -> List[Dict]:
        """多阶段检索"""
        
        # 阶段1：查询扩展（用LLM生成同义查询）
        expanded_queries = await self._expand_user_query(query)
        
        # 阶段2：多查询并行检索
        all_results = []
        for q, weight in expanded_queries:
            results = self._vector_search(q, filters, top_k=self.config['rerank_top_k'])
            for r in results:
                r['query_source'] = q
                r['query_weight'] = weight
                all_results.append(r)
        
        # 阶段3：融合与去重
        fused = self._fuse_results(all_results)
        
        # 阶段4：重排序（考虑增强类型）
        reranked = self._rerank_by_type(fused, query)
        
        # 阶段5：组装最终上下文
        final_results = self._assemble_context(reranked[:top_k])
        
        return final_results
    
    async def _expand_user_query(self, query: str) -> List[Tuple[str, float]]:
        """扩展用户查询"""
        expansions = [(query, 1.0)]  # 原始查询
        
        # 简单的同义词扩展（可用LLM增强）
        synonyms = {
            '远程办公': ['在家工作', '弹性工作', '居家办公', 'WFH'],
            '请假': ['休假', '事假', '病假', '申请假期'],
            '报销': ['费用报销', '差旅报销', '发票报销'],
        }
        
        for term, alts in synonyms.items():
            if term in query:
                for alt in alts:
                    new_query = query.replace(term, alt)
                    expansions.append((new_query, 0.8))
        
        # 生成疑问句变体
        if not query.endswith('?') and not query.endswith('？'):
            expansions.append((f"如何{query}？", 0.7))
            expansions.append((f"{query}怎么办？", 0.7))
        
        return expansions[:5]  # 最多5个变体
    
    def _vector_search(self, 
                       query: str, 
                       filters: Dict,
                       top_k: int) -> List[Dict]:
        """向量检索"""
        query_vec = self.embedder.encode(query)
        
        results = self.vector_store.search(
            query_vector=query_vec,
            filter=filters,
            top_k=top_k
        )
        return results
    
    def _fuse_results(self, results: List[Dict]) -> List[Dict]:
        """融合多查询结果"""
        # 按原始chunk ID分组
        groups = defaultdict(list)
        
        for r in results:
            original_id = r['metadata']['original_id']
            groups[original_id].append(r)
        
        # 合并分数
        fused = []
        for orig_id, group in groups.items():
            # 取最高分为代表
            best = max(group, key=lambda x: x['score'] * x.get('query_weight', 1.0))
            
            # 聚合信息
            all_types = [g['metadata']['search_type'] for g in group]
            all_queries = list(set(g.get('query_source', '') for g in group))
            
            fused.append({
                'original_id': orig_id,
                'score': best['score'],
                'content': best['metadata']['original_content'],
                'matched_types': all_types,  # 哪些增强类型匹配上了
                'matched_queries': all_queries,
                'best_match': best
            })
        
        # 按分数排序
        fused.sort(key=lambda x: x['score'], reverse=True)
        return fused
    
    def _rerank_by_type(self, 
                        results: List[Dict], 
                        original_query: str) -> List[Dict]:
        """根据匹配类型重排序"""
        
        for r in results:
            boost = 1.0
            
            # FAQ匹配给予高权重
            if 'faq_question' in r['matched_types']:
                boost *= self.config['faq_boost']
                
                # 检查FAQ是否与查询更接近
                faq_match = self._find_best_faq_match(r, original_query)
                if faq_match:
                    r['best_faq'] = faq_match
            
            # 原始内容匹配给予基础权重
            if 'original' in r['matched_types']:
                boost *= self.config['original_weight']
            
            # 假设查询匹配
            if 'hypothetical_query' in r['matched_types']:
                boost *= self.config['query_expansion_weight']
            
            r['final_score'] = r['score'] * boost
        
        results.sort(key=lambda x: x['final_score'], reverse=True)
        return results
    
    def _find_best_faq_match(self, result: Dict, query: str) -> Optional[Dict]:
        """找到最佳匹配的FAQ"""
        # 从metadata中恢复FAQ信息
        best_match = result.get('best_match', {})
        if 'reference_answer' in best_match.get('metadata', {}):
            return {
                'question': best_match['document'],  # FAQ问题
                'answer': best_match['metadata']['reference_answer']
            }
        return None
    
    def _assemble_context(self, results: List[Dict]) -> List[Dict]:
        """组装最终上下文"""
        assembled = []
        
        for r in results:
            context = {
                'content': r['content'],
                'score': r['final_score'],
                'match_info': {
                    'types': r['matched_types'],
                    'queries': r['matched_queries']
                }
            }
            
            # 如果有匹配的FAQ，优先使用FAQ答案
            if 'best_faq' in r:
                context['faq_match'] = r['best_faq']
                # 可选：用FAQ答案增强或替代原文
                context['enhanced_content'] = (
                    f"【常见问题】{r['best_faq']['question']}\n"
                    f"【官方解答】{r['best_faq']['answer']}\n"
                    f"【详细政策】{r['content'][:300]}..."
                )
            
            assembled.append(context)
        
        return assembled
```

## 四、实战案例：企业HR政策问答

### 4.1 原始数据

**政策原文**（节选）：

```
关于弹性工作制的管理规定

为提升员工工作满意度，兼顾业务运营需要，公司特制定弹性工作制。

适用对象：入职满6个月的正式员工，且所在部门支持远程协作。

申请流程：
1. 提前3个工作日通过OA提交《弹性工作申请表》
2. 直属上级审批（1个工作日内）
3. HR备案并开通VPN权限

工作规范：
- 核心工作时间（10:00-16:00）需保持在线
- 周报需在周五18:00前提交
- 紧急情况需2小时内响应

违规处理：
累计3次未按时提交周报，取消当月弹性工作资格。
```

### 4.2 增强过程

**FAQ生成**：

```json
[
  {
    "question": "哪些人可以申请弹性工作制？",
    "answer": "入职满6个月的正式员工，且所在部门支持远程协作。",
    "question_type": "condition"
  },
  {
    "question": "申请弹性工作需要提前多久？",
    "answer": "需要提前3个工作日通过OA提交申请。",
    "question_type": "how"
  },
  {
    "question": "弹性工作期间需要一直在线吗？",
    "answer": "核心工作时间（10:00-16:00）需保持在线，其他时间灵活安排。",
    "question_type": "what"
  },
  {
    "question": "违反弹性工作规定有什么后果？",
    "answer": "累计3次未按时提交周报，取消当月弹性工作资格。",
    "question_type": "consequence"
  }
]
```

**假设查询生成**：

```
远程办公怎么申请？
在家工作的条件是什么？
弹性工作制度适用范围
如何开通VPN权限？
周报提交截止时间
弹性工作会被取消吗？
WFH政策详解
```

**反向摘要**：

```
该文档规定了入职满6个月正式员工的弹性工作制申请条件、OA审批流程、核心工作时间要求（10-16点在线）及违规处罚措施（3次未交周报取消资格）。
```

**多视角重写**：

```python
{
    "普通员工": "入职半年后就能申请在家办公！记得提前3天在OA提交申请，等领导审批就行。工作时间比较灵活，但上午10点到下午4点必须在线，周五别忘了交周报。",
    
    "管理者": "团队成员申请弹性工作需满足：入职6个月+部门支持远程。审批时请评估其工作性质是否适合，并确保核心工作时间（10-16点）能联系到人。HR会协助开通VPN。",
    
    "HR": "弹性工作制管理要点：1）资格审核（6个月+正式员工+部门支持）；2）OA流程配置（3天提前量+上级审批）；3）VPN权限开通；4）月度合规检查（周报提交率）。"
}
```

### 4.3 检索效果对比

**查询**："在家办公需要什么条件？"

| 策略 | 召回内容 | 相似度 | 是否命中 |
|-----|---------|--------|---------|
| 基础RAG（原文） | "为提升员工工作满意度..."（开头） | 0.61 | ❌ |
| 基础RAG（原文） | "申请流程：1. 提前3个工作日..." | 0.58 | ❌ |
| **增强RAG（FAQ）** | **"哪些人可以申请弹性工作制？"** | **0.89** | ✅ |
| **增强RAG（假设查询）** | **"在家工作的条件是什么？"** | **0.92** | ✅ |
| **增强RAG（多视角）** | **"入职半年后就能申请在家办公！"** | **0.85** | ✅ |

**查询**："迟到周报会怎样？"

| 策略 | 召回内容 | 结果 |
|-----|---------|------|
| 基础RAG | 未召回"违规处理"段落 | ❌ 无法回答 |
| **增强RAG（FAQ）** | "违反弹性工作规定有什么后果？" | ✅ 直接命中 |
| **增强RAG（关键词）** | 关键词"周报"+"处罚"匹配 | ✅ 命中 |

### 4.4 完整代码实现

```python
# 初始化组件
llm = OpenAIClient(model="gpt-4-turbo")
embedder = SentenceTransformer('BAAI/bge-large-zh-v1.5')
vector_store = ChromaDB(collection_name="hr_policies_enhanced")

# 创建流水线
pipeline = DataAugmentationPipeline(
    llm_client=llm,
    embedding_model=embedder,
    vector_store=vector_store,
    config={
        'enhancement_types': [
            EnhancementType.FAQ,
            EnhancementType.HYPOTHETICAL_QUERY,
            EnhancementType.BACKWARD_SUMMARY,
            EnhancementType.ENTITY_TAGS,
            EnhancementType.KEYWORD_EXPANSION
        ],
        'index_multi_view': True  # HR场景需要多视角
    }
)

# 处理文档
chunks = [
    {
        'id': 'policy_001_elastic_work',
        'content': policy_text,
        'metadata': {'category': 'work_policy', 'department': 'HR'}
    }
]

result = asyncio.run(pipeline.process_documents(chunks))

print(f"增强完成：")
print(f"- 原始chunks: {result['index_stats']['original_chunks']}")
print(f"- 总向量数: {result['index_stats']['total_vectors']}")
print(f"- 扩展倍数: {result['index_stats']['expansion_ratio']:.1f}x")

# 检索测试
retriever = AugmentedRetriever(vector_store, embedder)

test_queries = [
    "在家办公需要什么条件？",
    "迟到周报会怎样？",
    "怎么申请VPN？",
    "弹性工作时间要求"
]

for q in test_queries:
    results = asyncio.run(retriever.retrieve(q, top_k=3))
    print(f"\n查询: {q}")
    print(f"命中类型: {results[0]['match_info']['types']}")
    print(f"匹配FAQ: {results[0].get('faq_match', {}).get('question', 'N/A')}")
    print(f"分数: {results[0]['score']:.3f}")
```

## 五、高级优化：质量与效率平衡

### 5.1 智能增强筛选

不是所有chunk都需要全维度增强：

```python
class SmartEnhancementSelector:
    def should_enhance(self, chunk: Dict) -> List[EnhancementType]:
        """智能选择增强策略"""
        content = chunk['content']
        types = []
        
        # 信息密度判断
        info_density = self._calculate_info_density(content)
        
        if info_density > 0.7:  # 高密度信息（政策、流程）
            types.extend([
                EnhancementType.FAQ,
                EnhancementType.HYPOTHETICAL_QUERY,
                EnhancementType.BACKWARD_SUMMARY
            ])
        elif info_density > 0.4:  # 中密度（说明、背景）
            types.extend([
                EnhancementType.BACKWARD_SUMMARY,
                EnhancementType.ENTITY_TAGS
            ])
        else:  # 低密度（过渡段落）
            types.append(EnhancementType.BACKWARD_SUMMARY)
        
        # 根据内容类型调整
        if self._is_procedure(content):  # 流程类
            types.append(EnhancementType.MULTI_VIEW)
        
        if self._has_numbers(content):  # 含数字
            types.append(EnhancementType.KEYWORD_EXPANSION)
        
        return list(set(types))
    
    def _calculate_info_density(self, text: str) -> float:
        """计算信息密度（实体数/总字数）"""
        import re
        # 提取数字、专有名词、动词
        info_units = len(re.findall(r'\d+|[A-Z][a-z]+|[\u4e00-\u9fa5]{2,}(?:是|有|需|应)', text))
        return min(info_units / len(text) * 10, 1.0)
```

### 5.2 增量增强

文档更新时只处理变更部分：

```python
class IncrementalEnhancement:
    def __init__(self, version_store):
        self.version_store = version_store
        
    async def process_update(self, 
                            new_chunks: List[Dict],
                            doc_id: str) -> List[EnhancedChunk]:
        """增量处理"""
        
        # 获取历史版本
        old_chunks = self.version_store.get_chunks(doc_id)
        old_hashes = {c['hash'] for c in old_chunks}
        
        to_enhance = []
        unchanged = []
        
        for chunk in new_chunks:
            chunk_hash = hashlib.md5(chunk['content'].encode()).hexdigest()
            
            if chunk_hash in old_hashes:
                # 未变更，复用历史增强结果
                old_enhanced = self.version_store.get_enhanced(chunk_hash)
                unchanged.append(old_enhanced)
            else:
                to_enhance.append(chunk)
        
        # 只增强变更部分
        new_enhanced = await self.enhancer.process_documents(to_enhance)
        
        # 合并保存
        all_enhanced = unchanged + new_enhanced
        self.version_store.save_version(doc_id, new_chunks, all_enhanced)
        
        return all_enhanced
```

### 5.3 成本优化

LLM调用成本优化策略：

```python
class CostOptimizedEnhancement:
    def __init__(self):
        self.local_embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        
    async def tiered_enhancement(self, chunk: Dict) -> EnhancedChunk:
        """分层增强：简单任务用本地模型，复杂任务用LLM"""
        
        # Tier 1: 本地模型生成摘要和关键词
        summary = self._local_summarize(chunk['content'])
        keywords = self._local_keywords(chunk['content'])
        
        # Tier 2: 检查是否需要LLM增强
        needs_llm = self._assess_complexity(chunk['content'])
        
        if not needs_llm:
            return EnhancedChunk(
                original_id=chunk['id'],
                original_content=chunk['content'],
                backward_summary=summary,
                keyword_expansion=keywords,
                faqs=[],  # 无FAQ
                hypothetical_queries=keywords[:3]  # 用关键词代替
            )
        
        # Tier 3: LLM生成高质量FAQ和查询
        return await self.enhancer.enhance_chunk(chunk)
    
    def _local_summarize(self, text: str) -> str:
        """抽取式摘要（无LLM）"""
        sentences = text.split('。')
        # 取前两句作为摘要
        return '。'.join(sentences[:2])[:100]
    
    def _local_keywords(self, text: str) -> List[str]:
        """TF-IDF提取关键词"""
        # 简化实现：高频词+停用词过滤
        import jieba
        from collections import Counter
        
        words = [w for w in jieba.cut(text) if len(w) > 1]
        stopwords = set(['的', '了', '是', '在', '和', '与', '及', '或'])
        words = [w for w in words if w not in stopwords]
        
        return [w for w, _ in Counter(words).most_common(10)]
    
    def _assess_complexity(self, text: str) -> bool:
        """评估内容复杂度，决定是否用LLM"""
        # 条件：长度>500字 或 包含条件从句 或 包含流程步骤
        has_conditions = any(w in text for w in ['如果', '若', '除非', '除...外'])
        has_procedures = any(w in text for w in ['步骤', '流程', '首先', '然后', '最后'])
        
        return len(text) > 500 or has_conditions or has_procedures
```

## 六、效果评估与监控

### 6.1 评估指标

```python
class EnhancementEvaluator:
    def evaluate_retrieval_improvement(self, 
                                        test_queries: List[Dict],
                                        baseline_retriever,
                                        enhanced_retriever) -> Dict:
        """评估增强效果"""
        
        baseline_metrics = {'recall@5': [], 'mrr': [], 'precision@1': []}
        enhanced_metrics = {'recall@5': [], 'mrr': [], 'precision@1': []}
        
        for q in test_queries:
            # 基线检索
            base_results = baseline_retriever.retrieve(q['text'])
            base_ids = [r['id'] for r in base_results]
            
            # 增强检索
            enhanced_results = enhanced_retriever.retrieve(q['text'])
            enhanced_ids = [r['original_id'] for r in enhanced_results]
            
            # 计算指标
            relevant = q['relevant_ids']
            
            # Recall@5
            base_recall = len(set(base_ids[:5]) & set(relevant)) / len(relevant)
            enh_recall = len(set(enhanced_ids[:5]) & set(relevant)) / len(relevant)
            
            baseline_metrics['recall@5'].append(base_recall)
            enhanced_metrics['recall@5'].append(enh_recall)
            
            # MRR
            base_mrr = self._calculate_mrr(base_ids, relevant)
            enh_mrr = self._calculate_mrr(enhanced_ids, relevant)
            
            baseline_metrics['mrr'].append(base_mrr)
            enhanced_metrics['mrr'].append(enh_mrr)
        
        return {
            'baseline': {k: np.mean(v) for k, v in baseline_metrics.items()},
            'enhanced': {k: np.mean(v) for k, v in enhanced_metrics.items()},
            'improvement': {
                k: (np.mean(enhanced_metrics[k]) - np.mean(baseline_metrics[k])) / 
                   np.mean(baseline_metrics[k]) * 100
                for k in baseline_metrics.keys()
            }
        }
```

### 6.2 生产监控

```python
class EnhancementMonitor:
    def __init__(self):
        self.stats = {
            'query_type_distribution': defaultdict(int),
            'match_type_distribution': defaultdict(int),
            'fallback_to_original': 0,
            'faq_hit_rate': []
        }
    
    def log_retrieval(self, query: str, results: List[Dict]):
        """记录检索日志"""
        
        # 分析查询类型
        q_type = self._classify_query_type(query)
        self.stats['query_type_distribution'][q_type] += 1
        
        # 记录匹配类型
        for r in results[:3]:
            for t in r.get('matched_types', []):
                self.stats['match_type_distribution'][t] += 1
        
        # 是否依赖原文 fallback
        if all('original' in r.get('matched_types', []) for r in results[:1]):
            self.stats['fallback_to_original'] += 1
        
        # FAQ命中率
        if any('faq_question' in r.get('matched_types', []) for r in results):
            self.stats['faq_hit_rate'].append(1)
        else:
            self.stats['faq_hit_rate'].append(0)
    
    def generate_report(self) -> Dict:
        """生成监控报告"""
        total = sum(self.stats['query_type_distribution'].values())
        
        return {
            'query_type_dist': dict(self.stats['query_type_distribution']),
            'match_type_dist': dict(self.stats['match_type_distribution']),
            'faq_hit_rate': np.mean(self.stats['faq_hit_rate']),
            'original_fallback_rate': self.stats['fallback_to_original'] / total if total else 0,
            'recommendations': self._generate_recommendations()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """基于数据生成优化建议"""
        recs = []
        
        if self.stats['match_type_distribution'].get('faq_question', 0) < 100:
            recs.append("FAQ匹配次数较少，建议增加FAQ生成数量或优化FAQ质量")
        
        if np.mean(self.stats['faq_hit_rate']) < 0.3:
            recs.append("FAQ命中率低于30%，建议分析未命中查询，补充对应FAQ")
        
        return recs
```

---

数据增强与合成是RAG系统的"预训练"阶段——它用LLM的生成能力弥补用户查询与文档表述之间的语义鸿沟。通过FAQ合成、假设查询、反向摘要等多维度增强，可以将**召回率提升30-50%**，特别是在术语不一致、口语化查询、问题-答案对齐等场景下效果显著。在实际部署中，建议结合智能筛选、增量更新和成本优化策略，在保证效果的同时控制计算开销。
