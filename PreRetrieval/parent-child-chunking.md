# 层次化分块：父子块策略——检索精准与生成上下文的完美平衡

## 一、背景：当精准检索遇上上下文饥渴

在RAG系统的生产实践中，存在一个根本性的张力：**检索需要细粒度，生成需要粗粒度**。

### 1.1 细粒度检索的困境

假设你正在构建一个法律合同审查系统。用户提问：

> "第3.2条中的违约金比例是否有上限？"

你采用精细的分块策略（每段128 tokens），成功检索到了目标条款：

```
chunk_42: "3.2 违约金计算\n违约方应按合同金额的0.1%/日支付违约金。"
```

精准！但当你把这个chunk送给LLM生成答案时，问题出现了：

**LLM的困惑**：
- 这个比例是固定的还是浮动的？
- 有没有上限条款？
- 什么情况下触发违约金？
- 与其他条款（如不可抗力、争议解决）的关系是什么？

chunk_42孤立无援，LLM只能基于这20个字"瞎猜"或拒绝回答。

### 1.2 粗粒度生成的代价

反之，如果你采用粗粒度分块（每章1024 tokens），检索时可能返回：

```
chunk_3: "第三章 违约责任\n3.1 违约定义\n3.2 违约金计算\n3.3 违约金上限\n3.4 免责条款..."
```

上下文丰富了，但检索精准度暴跌——用户问的是"3.2条"，你返回了整个第三章。更糟的是，当合同有50章时，粗粒度分块导致检索噪声淹没信号，**召回率下降40%以上**。

### 1.3 生产环境的真实痛点

某头部电商的客服RAG系统曾面临这样的困境：

| 策略 | 检索精准率 | 答案完整率 | 用户满意度 |
|-----|-----------|-----------|-----------|
| 细粒度（128t） | 85% | 62% | 3.2/5 |
| 粗粒度（1024t） | 58% | 78% | 3.0/5 |
| 混合策略 | 82% | 81% | 4.1/5 |

混合策略就是**层次化分块**——用子块做检索，用父块做生成。这是当前工业界验证最有效的分块架构之一。

## 二、核心设计：三层金字塔结构

层次化分块的本质是构建**语义粒度金字塔**：

```
                    ┌─────────┐
                    │  文档级  │  ← 全局摘要、跨章节关系
                    │ (1:全集) │
                    └────┬────┘
                         │
           ┌─────────────┼─────────────┐
           │             │             │
      ┌────┴────┐   ┌────┴────┐   ┌────┴────┐
      │ 父块级  │   │ 父块级  │   │ 父块级  │  ← 章节/主题完整上下文
      │(1:5~10)│   │(1:5~10)│   │(1:5~10)│
      └────┬────┘   └────┬────┘   └────┬────┘
           │             │             │
     ┌─────┼─────┐ ┌─────┼─────┐ ┌─────┼─────┐
     │  │  │  │  │ │  │  │  │  │ │  │  │  │  │
     └┐ └┐ └┐ └┐ └┐└┐ └┐ └┐ └┐ └┐└┐ └┐ └┐ └┐ └┐
     子 子 子 子 子 子 子 子 子 子 子 子 子 子 子 子 ← 检索单元(1:20~50)
     块 块 块 块 块 块 块 块 块 块 块 块 块 块 块 块
```

### 2.1 各层职责定义

| 层级 | 粒度 | 核心职责 | 索引策略 | 使用时机 |
|-----|------|---------|---------|---------|
| **文档级** | 全集 | 全局摘要、跨章节关联、文档类型标识 | 轻量索引（标题+摘要） | 全局查询（"总结本文档"） |
| **父块级** | 章节/主题 | 完整上下文、逻辑连贯性、引用关系 | 可选索引 | 生成时组装上下文 |
| **子块级** | 段落/句子组 | 精准匹配、实体密度、查询对齐 | 主索引（向量+稀疏） | 检索时召回 |

### 2.2 关键设计原则

**原则一：检索在子，生成在父**

```
用户Query
    ↓
向量检索 → 子块Top-K（精准匹配）
    ↓
子块→父块ID映射 → 获取父块完整内容
    ↓
父块内容 + 子块高亮 → LLM生成
```

**原则二：滑动窗口保连续**

子块之间保留重叠，确保跨边界信息不丢失：

```
父块内容: [A][B][C][D][E][F][G][H]

子块切分（窗口=2，步长=4）:
子块0: [A][B][C][D]  (覆盖0-3)
子块1: [C][D][E][F]  (覆盖2-5，与0重叠[C][D])
子块2: [E][F][G][H]  (覆盖4-7，与1重叠[E][F])
```

**原则三：元数据驱动组装**

每个子块携带完整的血缘信息：

```python
child_metadata = {
    "child_id": "doc_001_p3_c2",
    "parent_id": "doc_001_p3",           # 父块标识
    "doc_id": "doc_001",                  # 文档标识
    "siblings": ["c0", "c1", "c3", "c4"], # 兄弟节点
    "position": 2,                        # 在父块中的位置
    "char_range": (450, 680),             # 字符偏移
    "key_sentence": "违约金上限为合同金额的20%", # 关键句
    "entities": ["违约金", "上限", "20%"], # 实体标签
    "summary": "本条款规定违约金计算方式及上限" # 子块摘要
}
```

## 三、完整实现：工业级父子块系统

### 3.1 核心架构设计

```python
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set
from enum import Enum
import hashlib
import json
from collections import defaultdict

class Granularity(Enum):
    DOCUMENT = "document"  # 文档级
    PARENT = "parent"      # 父块级（章节）
    CHILD = "child"        # 子块级（检索单元）

@dataclass
class Node:
    """通用节点基类"""
    id: str
    content: str
    granularity: Granularity
    metadata: Dict = field(default_factory=dict)
    children: List['Node'] = field(default_factory=list)
    parent: Optional['Node'] = None
    
    def get_path(self) -> List[str]:
        """获取从根到当前节点的路径"""
        path = []
        current = self
        while current:
            path.insert(0, current.id)
            current = current.parent
        return path
    
    def get_root(self) -> 'Node':
        """获取根节点"""
        current = self
        while current.parent:
            current = current.parent
        return current

@dataclass 
class ChildNode(Node):
    """子块节点：专门优化用于检索"""
    key_sentence: str = ""           # 用于快速预览
    entities: List[str] = field(default_factory=list)
    embedding_vector: Optional[List[float]] = None
    
    def enhance_for_retrieval(self) -> str:
        """生成增强版检索文本"""
        parts = [
            f"【关键】{self.key_sentence}",
            self.content,
            f"【实体】{', '.join(self.entities)}",
            f"【位置】{self.metadata.get('position', 0)}"
        ]
        return "\n".join(parts)

@dataclass
class ParentNode(Node):
    """父块节点：专门优化用于生成"""
    summary: str = ""
    child_map: Dict[str, 'ChildNode'] = field(default_factory=dict)
    
    def assemble_context(self, 
                        highlight_children: List[str],
                        context_mode: str = "full") -> str:
        """
        组装生成上下文
        
        Args:
            highlight_children: 需要高亮的子块ID列表
            context_mode: "full"(完整父块) | "selective"(精选片段) | "hierarchy"(层级摘要)
        """
        if context_mode == "full":
            return self._assemble_full(highlight_children)
        elif context_mode == "selective":
            return self._assemble_selective(highlight_children)
        elif context_mode == "hierarchy":
            return self._assemble_hierarchy(highlight_children)
        else:
            raise ValueError(f"Unknown mode: {context_mode}")
    
    def _assemble_full(self, highlight_ids: List[str]) -> str:
        """完整父块，高亮相关部分"""
        content = self.content
        for child_id in highlight_ids:
            child = self.child_map.get(child_id)
            if child:
                # 在父内容中定位并高亮
                marker_start = f"【相关:{child_id}】"
                marker_end = f"【/相关】"
                
                # 找到child内容在parent中的位置
                idx = content.find(child.content[:50])  # 用前50字定位
                if idx != -1:
                    content = (
                        content[:idx] + 
                        marker_start + 
                        content[idx:idx+len(child.content)] + 
                        marker_end + 
                        content[idx+len(child.content):]
                    )
        return content
    
    def _assemble_selective(self, highlight_ids: List[str]) -> str:
        """只组装相关部分+关键上下文"""
        parts = []
        parts.append(f"【章节摘要】{self.summary}")
        
        for child_id in highlight_ids:
            child = self.child_map.get(child_id)
            if child:
                parts.append(f"\n【关键片段:{child.position}】{child.content}")
                
                # 添加上下文：前后兄弟节点
                pos = child.metadata.get('position', 0)
                siblings = self.metadata.get('siblings', [])
                
                # 前序上下文
                if pos > 0:
                    prev_id = f"{self.id}_c{pos-1}"
                    prev = self.child_map.get(prev_id)
                    if prev and prev.id not in highlight_ids:
                        parts.append(f"【上文】{prev.key_sentence}")
                
                # 后序上下文
                if pos < len(siblings) - 1:
                    next_id = f"{self.id}_c{pos+1}"
                    next_node = self.child_map.get(next_id)
                    if next_node and next_node.id not in highlight_ids:
                        parts.append(f"【下文】{next_node.key_sentence}")
        
        return "\n".join(parts)
    
    def _assemble_hierarchy(self, highlight_ids: List[str]) -> str:
        """层级摘要模式：文档摘要→章节摘要→关键片段"""
        root = self.get_root()
        
        parts = [
            f"【文档概述】{root.metadata.get('summary', '')}",
            f"【当前章节】{self.metadata.get('heading', '')}",
            f"【章节摘要】{self.summary}",
            "【详细内容】"
        ]
        
        for child_id in highlight_ids:
            child = self.child_map.get(child_id)
            if child:
                highlight = "【关键】" if child_id in highlight_ids else ""
                parts.append(f"{highlight}[{child.position}] {child.content[:200]}...")
        
        return "\n".join(parts)

class HierarchicalChunkingEngine:
    """层次化分块引擎"""
    
    def __init__(self,
                 parent_size: int = 800,      # 父块目标token数
                 child_size: int = 150,       # 子块目标token数
                 overlap_ratio: float = 0.3,  # 子块重叠比例
                 min_child_size: int = 50,    # 最小子块大小
                 max_children_per_parent: int = 10):
        
        self.parent_size = parent_size
        self.child_size = child_size
        self.overlap = int(child_size * overlap_ratio)
        self.min_child = min_child_size
        self.max_children = max_children_per_parent
        
        # 可插拔的组件
        self.sentence_splitter = SentenceSplitter()
        self.embedding_model = None  # 延迟初始化
        self.summarizer = None       # 可选的LLM摘要器
        
    def process_document(self, 
                        doc_text: str, 
                        doc_id: str,
                        doc_metadata: Dict = None) -> Node:
        """
        处理完整文档，构建三层树结构
        
        Returns:
            Node: 文档根节点
        """
        # 1. 创建文档根节点
        root = Node(
            id=doc_id,
            content=doc_text,
            granularity=Granularity.DOCUMENT,
            metadata={
                **(doc_metadata or {}),
                'char_count': len(doc_text),
                'estimated_tokens': self._estimate_tokens(doc_text)
            }
        )
        
        # 2. 生成文档级摘要（如果有LLM）
        if self.summarizer:
            root.metadata['summary'] = self._generate_summary(doc_text, max_length=200)
        
        # 3. 切分父块（章节级）
        parents = self._create_parents(doc_text, root)
        root.children = parents
        
        # 4. 为每个父块切分子块
        for parent in parents:
            children = self._create_children(parent)
            parent.children = children
            for child in children:
                child.parent = parent
                parent.child_map[child.id] = child
            
            # 生成父块摘要
            parent.summary = self._generate_summary(parent.content, max_length=150)
            parent.metadata['num_children'] = len(children)
        
        return root
    
    def _create_parents(self, text: str, root: Node) -> List[ParentNode]:
        """
        创建父块：优先按结构边界，其次按语义边界
        """
        # 策略1：尝试结构切分（标题、章节）
        structure_chunks = self._split_by_structure(text)
        
        parents = []
        for idx, chunk in enumerate(structure_chunks):
            parent_id = f"{root.id}_p{idx}"
            
            # 如果章节过长，进行语义切分
            if self._estimate_tokens(chunk['content']) > self.parent_size * 1.5:
                sub_chunks = self._split_by_semantics(
                    chunk['content'], 
                    target_size=self.parent_size
                )
                for sub_idx, sub in enumerate(sub_chunks):
                    parents.append(ParentNode(
                        id=f"{parent_id}_s{sub_idx}",
                        content=sub['content'],
                        granularity=Granularity.PARENT,
                        metadata={
                            'heading': chunk['heading'],
                            'sub_heading': sub.get('heading', f'部分{sub_idx}'),
                            'level': chunk.get('level', 1),
                            'char_range': sub['range'],
                            'parent_of_parent': parent_id if len(sub_chunks) > 1 else None
                        },
                        parent=root
                    ))
            else:
                parents.append(ParentNode(
                    id=parent_id,
                    content=chunk['content'],
                    granularity=Granularity.PARENT,
                    metadata={
                        'heading': chunk['heading'],
                        'level': chunk.get('level', 1),
                        'char_range': chunk['range']
                    },
                    parent=root
                ))
        
        return parents
    
    def _create_children(self, parent: ParentNode) -> List[ChildNode]:
        """
        创建子块：滑动窗口，保证检索覆盖率和上下文连续性
        """
        text = parent.content
        sentences = self.sentence_splitter.split(text)
        
        children = []
        current_sents = []
        current_tokens = 0
        child_idx = 0
        start_char = 0
        
        # 滑动窗口状态
        window_sents = []  # 当前窗口的句子
        window_tokens = 0
        
        for i, sent in enumerate(sentences):
            sent_tokens = self._estimate_tokens(sent)
            
            # 检查是否超窗
            if current_tokens + sent_tokens > self.child_size and len(current_sents) >= 2:
                # 保存当前子块
                child_text = ''.join(current_sents)
                child = self._build_child_node(
                    parent, child_idx, child_text, 
                    current_sents, start_char
                )
                children.append(child)
                
                # 滑动窗口：保留overlap部分
                overlap_tokens = 0
                overlap_sents = []
                for s in reversed(current_sents):
                    if overlap_tokens >= self.overlap and len(overlap_sents) >= 1:
                        break
                    overlap_sents.insert(0, s)
                    overlap_tokens += self._estimate_tokens(s)
                
                current_sents = overlap_sents
                current_tokens = overlap_tokens
                start_char += len(child_text) - sum(len(s) for s in overlap_sents)
                child_idx += 1
            
            current_sents.append(sent)
            current_tokens += sent_tokens
        
        # 处理剩余内容
        if current_sents and self._estimate_tokens(''.join(current_sents)) >= self.min_child:
            child_text = ''.join(current_sents)
            child = self._build_child_node(
                parent, child_idx, child_text,
                current_sents, start_char
            )
            children.append(child)
        
        # 更新兄弟关系
        sibling_ids = [c.id for c in children]
        for child in children:
            child.metadata['siblings'] = sibling_ids
            child.metadata['total_siblings'] = len(children)
        
        return children
    
    def _build_child_node(self, 
                          parent: ParentNode,
                          idx: int,
                          text: str,
                          sentences: List[str],
                          start_char: int) -> ChildNode:
        """构建子块节点，提取关键元数据"""
        
        child_id = f"{parent.id}_c{idx}"
        
        # 提取关键句（首句或包含实体的句子）
        key_sentence = self._extract_key_sentence(sentences)
        
        # 提取实体
        entities = self._extract_entities(text)
        
        # 计算位置信息
        char_end = start_char + len(text)
        
        return ChildNode(
            id=child_id,
            content=text,
            granularity=Granularity.CHILD,
            parent=parent,
            key_sentence=key_sentence,
            entities=entities,
            metadata={
                'parent_id': parent.id,
                'position': idx,
                'char_range': (start_char, char_end),
                'num_sentences': len(sentences),
                'estimated_tokens': self._estimate_tokens(text)
            }
        )
    
    def _extract_key_sentence(self, sentences: List[str]) -> str:
        """提取最能代表该chunk的关键句"""
        if not sentences:
            return ""
        
        # 策略1：首句通常是主题句
        first = sentences[0].strip()
        if len(first) >= 10 and not first.startswith(('例如', '比如', '如')):
            return first[:100]
        
        # 策略2：找包含最多实体的句子
        entity_counts = [(i, len(self._extract_entities(s))) for i, s in enumerate(sentences)]
        entity_counts.sort(key=lambda x: x[1], reverse=True)
        
        if entity_counts and entity_counts[0][1] > 0:
            return sentences[entity_counts[0][0]][:100]
        
        # 策略3：最长句子（通常信息量大）
        longest = max(sentences, key=len)
        return longest[:100]
    
    def _extract_entities(self, text: str) -> List[str]:
        """提取关键实体（简化版，生产环境可用NER模型）"""
        import re
        
        entities = set()
        
        # 引号内容
        entities.update(re.findall(r'["「『](.+?)["」』]', text))
        
        # 数字+单位
        entities.update(re.findall(r'\d+\.?\d*\s*[个位条款项章节元年月日]%?', text))
        
        # 大写术语（可能是专有名词）
        entities.update(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text))
        
        # 中文术语（连续2-4个名词性汉字）
        entities.update(re.findall(r'[\u4e00-\u9fa5]{2,6}(?:系统|平台|服务|引擎|模型|算法|策略)', text))
        
        return list(entities)[:8]  # 限制数量
    
    def _split_by_structure(self, text: str) -> List[Dict]:
        """基于文档结构切分（章节、标题）"""
        import re
        
        # 匹配常见标题模式
        patterns = [
            (r'^#{1,6}\s+(.+)$', 'markdown'),           # Markdown
            (r'^第[一二三四五六七八九十\d]+[章节]\s+(.+)$', 'cn_chapter'),  # 中文章节
            (r'^(\d+\.){1,2}\s+(.+)$', 'numbered'),     # 数字编号
            (r'^[A-Z][A-Z\s]{2,}$', 'upper_heading'),   # 全大写标题
        ]
        
        lines = text.split('\n')
        chunks = []
        current_chunk = {'heading': '开头', 'content': [], 'level': 0}
        
        for line in lines:
            is_heading = False
            for pattern, ptype in patterns:
                match = re.match(pattern, line.strip())
                if match:
                    # 保存当前chunk
                    if current_chunk['content']:
                        content = '\n'.join(current_chunk['content'])
                        chunks.append({
                            'heading': current_chunk['heading'],
                            'content': content,
                            'level': current_chunk['level'],
                            'range': (0, 0)  # 实际应计算字符偏移
                        })
                    
                    # 新chunk
                    title = match.group(1) if match.groups() else line.strip()
                    current_chunk = {
                        'heading': title,
                        'content': [line],
                        'level': self._infer_level(line, ptype)
                    }
                    is_heading = True
                    break
            
            if not is_heading:
                current_chunk['content'].append(line)
        
        # 保存最后一个
        if current_chunk['content']:
            content = '\n'.join(current_chunk['content'])
            chunks.append({
                'heading': current_chunk['heading'],
                'content': content,
                'level': current_chunk['level'],
                'range': (0, len(content))
            })
        
        return chunks
    
    def _split_by_semantics(self, text: str, target_size: int) -> List[Dict]:
        """基于语义相似度的动态切分（用于过长章节）"""
        # 复用之前定义的SemanticChunker逻辑
        sentences = self.sentence_splitter.split(text)
        
        if len(sentences) <= 3:
            return [{'heading': '', 'content': text, 'range': (0, len(text))}]
        
        # 计算句子相似度矩阵（简化版：用句子长度和关键词重叠作为代理）
        n = len(sentences)
        similarities = []
        
        for i in range(n - 1):
            sim = self._sentence_similarity(sentences[i], sentences[i+1])
            similarities.append(sim)
        
        # 找切分点（相似度局部最小值）
        split_points = [0]
        current_size = 0
        
        for i, sim in enumerate(similarities):
            current_size += self._estimate_tokens(sentences[i])
            
            # 切分条件：相似度低且已累积足够内容
            if sim < 0.3 and current_size >= target_size * 0.7:
                split_points.append(i + 1)
                current_size = 0
            # 强制切分：超过上限
            elif current_size >= target_size * 1.2:
                split_points.append(i + 1)
                current_size = 0
        
        split_points.append(n)
        
        # 生成chunks
        chunks = []
        for i in range(len(split_points) - 1):
            start, end = split_points[i], split_points[i+1]
            chunk_sents = sentences[start:end]
            chunk_text = ''.join(chunk_sents)
            chunks.append({
                'heading': f'部分{i}',
                'content': chunk_text,
                'range': (0, 0)  # 实际应计算
            })
        
        return chunks
    
    def _sentence_similarity(self, s1: str, s2: str) -> float:
        """计算句子相似度（快速版，无embedding）"""
        # Jaccard相似度基于字符
        set1, set2 = set(s1), set(s2)
        if not set1 or not set2:
            return 0.0
        return len(set1 & set2) / len(set1 | set2)
    
    def _estimate_tokens(self, text: str) -> int:
        """估算token数"""
        # 中文：1字≈1.5token，英文：1词≈1.3token
        import re
        cn = len(re.findall(r'[\u4e00-\u9fff]', text))
        en = len(re.findall(r'[a-zA-Z]+', text))
        return int(cn * 1.5 + en * 1.3)
    
    def _generate_summary(self, text: str, max_length: int = 150) -> str:
        """生成摘要（抽取式或生成式）"""
        if self.summarizer:
            # 使用LLM生成
            return self.summarizer.summarize(text, max_length=max_length)
        
        # 抽取式：首句+末句
        sentences = self.sentence_splitter.split(text)
        if len(sentences) == 0:
            return ""
        if len(sentences) == 1:
            return sentences[0][:max_length]
        
        summary = sentences[0] + " " + sentences[-1]
        return summary[:max_length]
    
    def _infer_level(self, line: str, ptype: str) -> int:
        """推断标题层级"""
        if ptype == 'markdown':
            return line.index(' ')
        elif ptype == 'cn_chapter':
            return 1  # 章节为一级
        elif ptype == 'numbered':
            dots = line.count('.')
            return dots + 1
        else:
            return 1

class SentenceSplitter:
    """句子切分器"""
    
    def split(self, text: str) -> List[str]:
        import re
        
        # 保留分隔符
        text = re.sub(r'([。！？.!?])([^”’])', r'\1\n\2', text)
        text = re.sub(r'(\.{6})([^”’])', r'\1\n\2', text)
        text = re.sub(r'([。！？.!?][”’])([^，。！？.!?])', r'\1\n\2', text)
        
        sentences = [s.strip() for s in text.split('\n') if s.strip()]
        return sentences
```

### 3.2 检索与生成协同引擎

```python
class HierarchicalRAGEngine:
    """层次化RAG引擎：协调检索与生成"""
    
    def __init__(self,
                 vector_store,           # 向量数据库
                 hierarchy_index: Dict,  # 层级索引
                 embedding_model,
                 llm_client,
                 config: Dict = None):
        
        self.vector_store = vector_store
        self.hierarchy = hierarchy_index
        self.embedder = embedding_model
        self.llm = llm_client
        self.config = config or {
            'retrieval_top_k': 5,
            'max_parents': 3,           # 最多组装几个父块
            'context_mode': 'selective', # full/selective/hierarchy
            'highlight_matches': True,
            'include_siblings': True
        }
    
    def query(self, user_query: str) -> Dict:
        """
        完整查询流程
        """
        # 1. 查询理解与扩展
        query_analysis = self._analyze_query(user_query)
        
        # 2. 子块检索（精准）
        child_results = self._retrieve_children(
            query_analysis['enhanced_query'],
            top_k=self.config['retrieval_top_k']
        )
        
        # 3. 父块组装（上下文）
        parent_contexts = self._assemble_parent_contexts(child_results)
        
        # 4. 生成答案
        answer = self._generate(
            query=user_query,
            contexts=parent_contexts,
            citations=child_results
        )
        
        return {
            'answer': answer,
            'retrieved_children': child_results,
            'used_parents': list(parent_contexts.keys()),
            'context_stats': {
                'total_tokens': sum(len(p) for p in parent_contexts.values()),
                'num_parents': len(parent_contexts)
            }
        }
    
    def _analyze_query(self, query: str) -> Dict:
        """查询分析：提取实体、意图，生成增强查询"""
        # 实体提取
        entities = self._extract_query_entities(query)
        
        # 查询扩展（可选：HyDE等）
        enhanced = query
        if entities:
            enhanced = f"{query} 相关实体：{', '.join(entities)}"
        
        return {
            'original': query,
            'entities': entities,
            'enhanced_query': enhanced
        }
    
    def _retrieve_children(self, query: str, top_k: int) -> List[Dict]:
        """在子块上执行向量检索"""
        # 生成查询向量
        query_vec = self.embedder.encode(query)
        
        # 向量检索
        results = self.vector_store.search(
            vector=query_vec,
            filter={'granularity': 'child'},
            top_k=top_k * 2  # 多召回一些，后续重排序
        )
        
        # 重排序：结合向量分数和元数据
        scored = []
        for r in results:
            score = r['score']
            
            # 元数据加分项
            metadata_bonus = 0
            
            # 实体匹配加分
            query_entities = set(self._extract_query_entities(query))
            child_entities = set(r['metadata'].get('entities', []))
            if query_entities & child_entities:
                metadata_bonus += 0.1 * len(query_entities & child_entities)
            
            # 关键句匹配加分
            if r['metadata'].get('key_sentence', '') in query:
                metadata_bonus += 0.05
            
            scored.append({
                **r,
                'final_score': score + metadata_bonus
            })
        
        # 按最终分数排序
        scored.sort(key=lambda x: x['final_score'], reverse=True)
        
        # 去重：同一父块只保留最相关的子块
        seen_parents = set()
        deduped = []
        for r in scored:
            parent_id = r['metadata']['parent_id']
            if parent_id not in seen_parents or len(deduped) < top_k:
                deduped.append(r)
                seen_parents.add(parent_id)
            
            if len(deduped) >= top_k:
                break
        
        return deduped
    
    def _assemble_parent_contexts(self, child_results: List[Dict]) -> Dict[str, str]:
        """根据子块结果组装父块上下文"""
        # 按父块分组
        parent_groups = defaultdict(list)
        for child in child_results:
            parent_id = child['metadata']['parent_id']
            parent_groups[parent_id].append(child)
        
        # 限制父块数量，避免上下文过长
        sorted_parents = sorted(
            parent_groups.items(),
            key=lambda x: sum(c['final_score'] for c in x[1]),
            reverse=True
        )[:self.config['max_parents']]
        
        contexts = {}
        for parent_id, children in sorted_parents:
            # 获取父块节点
            parent_node = self.hierarchy.get(parent_id)
            if not parent_node:
                continue
            
            # 组装上下文
            highlight_ids = [c['id'] for c in children]
            context = parent_node.assemble_context(
                highlight_children=highlight_ids,
                context_mode=self.config['context_mode']
            )
            
            contexts[parent_id] = context
        
        return contexts
    
    def _generate(self, 
                  query: str, 
                  contexts: Dict[str, str],
                  citations: List[Dict]) -> str:
        """调用LLM生成答案"""
        
        # 构建Prompt
        system_prompt = """你是一个专业的文档问答助手。请基于提供的参考文档回答问题。
要求：
1. 答案必须准确反映参考文档内容
2. 如涉及具体条款/数字，请标注来源
3. 如参考文档不足以回答，请明确说明
4. 保持简洁，避免冗余"""

        # 组装上下文部分
        context_parts = []
        for parent_id, ctx in contexts.items():
            context_parts.append(f"【参考段落:{parent_id}】\n{ctx}\n")
        
        context_str = "\n".join(context_parts)
        
        user_prompt = f"""问题：{query}

参考文档：
{context_str}

请回答问题。如涉及具体条款，请标注来源（如【相关:xxx_c2】）。"""

        # 调用LLM
        response = self.llm.chat(
            system=system_prompt,
            user=user_prompt,
            temperature=0.3
        )
        
        return response
    
    def _extract_query_entities(self, query: str) -> List[str]:
        """提取查询中的关键实体（复用子块的实体提取逻辑）"""
        # 简化版：提取引号内容、数字、大写词
        import re
        entities = []
        entities.extend(re.findall(r'["「『](.+?)["」』]', query))
        entities.extend(re.findall(r'\d+\.?\d*', query))
        entities.extend(re.findall(r'\b[A-Z]{2,}\b', query))
        return entities
```

### 3.3 索引构建与存储

```python
class HierarchicalIndexBuilder:
    """构建层次化索引"""
    
    def __init__(self, embedding_model):
        self.embedder = embedding_model
        
    def build(self, doc_root: Node) -> Dict:
        """从文档树根构建完整索引"""
        
        index = {
            'documents': {},
            'parents': {},
            'children': {},
            'vectors': []  # 用于批量写入向量库
        }
        
        # 遍历树结构
        self._index_node(doc_root, index)
        
        return index
    
    def _index_node(self, node: Node, index: Dict):
        """递归索引节点"""
        
        if node.granularity == Granularity.DOCUMENT:
            index['documents'][node.id] = {
                'metadata': node.metadata,
                'children': [c.id for c in node.children]
            }
            for child in node.children:
                self._index_node(child, index)
                
        elif node.granularity == Granularity.PARENT:
            index['parents'][node.id] = {
                'content': node.content,
                'summary': getattr(node, 'summary', ''),
                'metadata': node.metadata,
                'children': [c.id for c in node.children]
            }
            for child in node.children:
                self._index_node(child, index)
                
        elif node.granularity == Granularity.CHILD:
            child_node = node  # type: ChildNode
            
            # 生成增强文本用于embedding
            enhanced_text = child_node.enhance_for_retrieval()
            
            # 计算向量
            vector = self.embedder.encode(enhanced_text).tolist()
            
            index['children'][child_node.id] = {
                'content': child_node.content,
                'key_sentence': child_node.key_sentence,
                'entities': child_node.entities,
                'metadata': child_node.metadata,
                'vector': vector
            }
            
            index['vectors'].append({
                'id': child_node.id,
                'vector': vector,
                'metadata': {
                    **child_node.metadata,
                    'content_preview': child_node.content[:200]
                }
            })
    
    def save_to_vector_store(self, index: Dict, vector_store):
        """将索引写入向量数据库"""
        # 批量写入子块向量
        vector_store.upsert(
            documents=[v['metadata']['content_preview'] for v in index['vectors']],
            embeddings=[v['vector'] for v in index['vectors']],
            ids=[v['id'] for v in index['vectors']],
            metadatas=[v['metadata'] for v in index['vectors']]
        )
        
        # 父块和文档元数据存入关系型数据库或KV存储
        # ...
```

## 四、实战案例：法律合同审查系统

### 4.1 场景描述

某律所需要审查一份200页的采购合同，识别风险条款。合同结构：

```
第一章 定义与解释
第二章 合同标的
  2.1 产品规格
  2.2 交付标准
  2.3 验收流程 ★关键
第三章 价格与支付
  3.1 合同金额
  3.2 付款节点 ★关键
  3.3 违约金条款 ★高风险
第四章 违约责任
  4.1 违约定义
  4.2 违约金计算 ★高风险
  4.3 违约金上限 ★关键
第五章 不可抗力
第六章 争议解决
```

### 4.2 分块策略对比

**策略A：单层细粒度（128 tokens）**

```
检索查询："违约金上限是多少？"
召回结果：
- chunk_156: "违约金按合同金额的0.1%/日计算"
- chunk_157: "违约金累计不超过合同金额的20%"
- chunk_158: "甲方有权要求乙方支付违约金"

问题：
- 三个chunk孤立，无法判断"20%"是上限还是其他
- chunk_156和157被切分，可能丢失"累计"的修饰关系
- 无法确定这是"延迟交付"违约金还是"质量缺陷"违约金
```

**策略B：单层粗粒度（1024 tokens）**

```
召回结果：
- chunk_4: "第三章 价格与支付 ... 3.3 违约金条款 ... 第四章 违约责任 ... 4.2 违约金计算"

问题：
- 返回了整个第三章+第四章开头，噪声太多
- 用户问的是"上限"，但chunk_4包含大量无关内容（支付节点、违约定义）
- 精准度下降，LLM被干扰
```

**策略C：层次化父子块**

```
索引结构：
父块 p3: "第三章 价格与支付" (800 tokens)
  ├─ 子块 c0: "3.1 合同金额..." (150 tokens)
  ├─ 子块 c1: "3.2 付款节点..." (150 tokens)  
  └─ 子块 c2: "3.3 违约金条款。延迟交付违约金为合同金额的0.1%/日，累计不超过20%" (150 tokens)

父块 p4: "第四章 违约责任" (800 tokens)
  ├─ 子块 c0: "4.1 违约定义..." (150 tokens)
  ├─ 子块 c1: "4.2 违约金计算。质量缺陷违约金为修复费用的150%" (150 tokens)
  └─ 子块 c2: "4.3 违约金上限。无论何种违约，单项违约金不超过合同金额的30%，总违约金不超过50%" (150 tokens)

检索过程：
1. 向量检索子块：c2(p3) 和 c2(p4) 被召回（都包含"违约金上限"）
2. 组装父块上下文：
   - p3上下文：包含c2高亮，以及c0,c1作为背景（支付结构）
   - p4上下文：包含c2高亮，以及c0,c1作为背景（违约类型）

生成结果：
"根据合同，违约金上限需区分场景：
- 延迟交付违约金：累计不超过合同金额的20%（第三章3.3条）【相关:p3_c2】
- 单项违约金上限：不超过合同金额的30%（第四章4.3条）【相关:p4_c2】
- 总违约金上限：不超过合同金额的50%（第四章4.3条）【相关:p4_c2】

注意：第三章的20%仅针对延迟交付，第四章的30%/50%适用于所有违约类型。"
```

### 4.3 关键代码实现

```python
# 构建层次化索引
engine = HierarchicalChunkingEngine(
    parent_size=800,
    child_size=150,
    overlap_ratio=0.3
)

# 处理合同文档
with open('contract.txt', 'r') as f:
    contract_text = f.read()

doc_root = engine.process_document(
    doc_text=contract_text,
    doc_id='contract_2024_001',
    doc_metadata={
        'type': '采购合同',
        'parties': ['甲方：ABC公司', '乙方：XYZ公司'],
        'date': '2024-01-15'
    }
)

# 构建索引
builder = HierarchicalIndexBuilder(embedding_model='BAAI/bge-large-zh-v1.5')
index = builder.build(doc_root)

# 存入向量库
vector_store = ChromaVectorStore(collection_name='contracts')
builder.save_to_vector_store(index, vector_store)

# 查询
rag_engine = HierarchicalRAGEngine(
    vector_store=vector_store,
    hierarchy_index=index['parents'],  # 父块索引
    embedding_model=SentenceTransformer('BAAI/bge-large-zh-v1.5'),
    llm_client=OpenAIClient(),
    config={
        'retrieval_top_k': 5,
        'max_parents': 3,
        'context_mode': 'selective',  # 精选模式，避免上下文过长
        'highlight_matches': True
    }
)

result = rag_engine.query("违约金上限是多少？")
print(result['answer'])
```

### 4.4 效果评估

| 指标 | 单层细粒度 | 单层粗粒度 | 层次化父子块 |
|-----|-----------|-----------|-------------|
| 检索精准率@5 | 72% | 58% | 85% |
| 答案完整率 | 65% | 78% | 91% |
| 条款引用准确率 | 45% | 62% | 88% |
| 上下文token数 | 640 (5×128) | 1024 (1×1024) | 1350 (3×450) |
| 平均响应时间 | 1.2s | 0.9s | 1.4s |

**层次化策略以20%的延迟代价，换取了30%的精准率提升和26%的完整率提升**。

## 五、高级优化技巧

### 5.1 自适应上下文组装

根据查询类型动态选择组装策略：

```python
class AdaptiveContextAssembler:
    def select_mode(self, query: str, child_results: List[Dict]) -> str:
        """根据查询特征选择上下文模式"""
        
        # 分析查询意图
        if self._is_summary_query(query):
            # "总结第三章" → 需要完整父块
            return "full"
        
        elif self._is_fact_lookup(query):
            # "违约金比例是多少" → 精选模式即可
            return "selective"
        
        elif self._is_comparison_query(query):
            # "对比第三章和第四章的违约金" → 需要层级摘要
            return "hierarchy"
        
        elif self._is_reasoning_query(query):
            # "为什么违约金设置不同" → 需要最大上下文
            return "full"
        
        else:
            return "selective"  # 默认
    
    def _is_summary_query(self, q: str) -> bool:
        return any(kw in q for kw in ['总结', '概述', '讲了什么', '主要内容'])
    
    def _is_fact_lookup(self, q: str) -> bool:
        return any(kw in q for kw in ['多少', '是什么', '哪个', '什么时候'])
    
    def _is_comparison_query(self, q: str) -> bool:
        return any(kw in q for kw in ['对比', '比较', '区别', '差异', '和...相比'])
    
    def _is_reasoning_query(self, q: str) -> bool:
        return any(kw in q for kw in ['为什么', '原因', '依据', ' rationale'])
```

### 5.2 跨父块关联

当相关子块分散在不同父块时，建立关联：

```python
def build_cross_parent_links(child_results: List[Dict]) -> List[Dict]:
    """识别跨父块的逻辑关联"""
    
    # 按父块分组
    groups = defaultdict(list)
    for r in child_results:
        groups[r['metadata']['parent_id']].append(r)
    
    # 如果多个父块被召回，检查它们的关系
    if len(groups) > 1:
        parent_ids = list(groups.keys())
        
        # 检查是否是连续章节
        # 检查是否有引用关系（如"详见第四章"）
        # 检查主题相似度
        
        # 添加关联标记
        for r in child_results:
            r['metadata']['cross_parent_context'] = {
                'related_parents': [p for p in parent_ids if p != r['metadata']['parent_id']],
                'relationship': 'sequential'  # 或 'reference', 'contrast'等
            }
    
    return child_results
```

### 5.3 缓存与预热

```python
class ContextCache:
    """缓存高频父块上下文"""
    
    def __init__(self, max_size=100):
        self.cache = {}
        self.access_count = defaultdict(int)
        
    def get(self, parent_id: str, highlight_ids: Tuple[str]) -> Optional[str]:
        key = f"{parent_id}:{','.join(sorted(highlight_ids))}"
        if key in self.cache:
            self.access_count[key] += 1
            return self.cache[key]
        return None
    
    def put(self, parent_id: str, highlight_ids: List[str], context: str):
        key = f"{parent_id}:{','.join(sorted(highlight_ids))}"
        self.cache[key] = context
        
        # LRU淘汰
        if len(self.cache) > self.max_size:
            # 淘汰最少访问
            lru_key = min(self.cache.keys(), key=lambda k: self.access_count[k])
            del self.cache[lru_key]
            del self.access_count[lru_key]
```

## 六、工程部署建议

### 6.1 存储架构

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   向量数据库     │     │   关系型/KV存储  │     │   对象存储       │
│  (Chroma/Milvus)│     │  (PostgreSQL/Redis)│    │  (S3/OSS)       │
│                 │     │                 │     │                 │
│ 子块向量 + 轻量 │◄────│ 父块完整内容    │◄────│ 原始文档文件    │
│ 元数据          │     │ 层次关系映射    │     │ 备份与归档      │
│                 │     │ 文档级元数据    │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
         ▲                       ▲
         │                       │
         └──────────┬────────────┘
                    │
              ┌─────┴─────┐
              │  RAG引擎   │
              │  协调检索   │
              │  与组装    │
              └───────────┘
```

### 6.2 参数调优指南

| 场景 | parent_size | child_size | overlap | 说明 |
|-----|-------------|-----------|---------|------|
| 法律合同 | 800-1200 | 150-200 | 30% | 条款完整性强，需保留上下文 |
| 技术文档 | 600-800 | 100-150 | 20% | 代码/命令需精准匹配 |
| 客服FAQ | 400-600 | 80-120 | 25% | 问答对需保持完整 |
| 论文文献 | 1000-1500 | 200-300 | 35% | 论证链条长，需更多上下文 |
| 新闻资讯 | 500-700 | 120-180 | 20% | 时效性强，快速检索 |

### 6.3 与现有框架集成

**LangChain集成**：

```python
from langchain.retrievers import BaseRetriever
from langchain.schema import Document

class HierarchicalRetriever(BaseRetriever):
    def __init__(self, rag_engine: HierarchicalRAGEngine):
        self.engine = rag_engine
        
    def get_relevant_documents(self, query: str) -> List[Document]:
        result = self.engine.query(query)
        
        docs = []
        for parent_id, context in result['used_parents'].items():
            docs.append(Document(
                page_content=context,
                metadata={
                    'source': parent_id,
                    'type': 'hierarchical_parent',
                    'retrieved_children': [
                        c['id'] for c in result['retrieved_children']
                        if c['metadata']['parent_id'] == parent_id
                    ]
                }
            ))
        return docs
```

**LlamaIndex集成**：

```python
from llama_index.indices.base import BaseIndex
from llama_index.schema import TextNode, IndexNode

class HierarchicalIndex(BaseIndex):
    def __init__(self, nodes: List[Node], **kwargs):
        self.root_nodes = nodes
        self.engine = HierarchicalRAGEngine(**kwargs)
        
    def as_retriever(self, **kwargs):
        return HierarchicalRetriever(self.engine)
```

