# 语义感知的智能分块：让RAG告别"半截话"灾难

## 一、背景：一个真实的生产事故

2024年初，某金融科技公司的智能客服系统上线了新功能——基于内部知识库的员工福利问答。上线第一周，HR部门收到了大量投诉。

**用户提问**："公司的年假政策是什么？"

**系统回答**："根据《员工手册》第三章，所有正式员工每年享有15天带薪年假。"

看起来完美无缺。但问题在于，知识库中完整的原文是：

> "所有正式员工每年享有15天带薪年假。**但销售部门员工因实行弹性工作制，年假天数为10天，且需在淡季申请。**"

固定长度分块（512 tokens）把这段话切成了两部分。检索时只召回前半段，后半段的"但书"被彻底遗漏。销售部门的员工拿着系统截图申请15天年假，引发了严重的内部纠纷。

这就是**语义完整性被破坏**的典型代价。传统基于固定token长度的分块策略，本质上是在不了解内容的情况下"盲切"。当切分点落在关键修饰语、条件从句或否定词上时，语义被肢解，RAG系统就成了"断章取义"的专家。

## 二、问题剖析：为什么固定长度分块不够

### 2.1 语义完整性破坏的三种模式

**模式一：句子截断**

```
原文： "用户在使用信用卡进行境外消费时，若单笔交易金额超过等值5000美元，
       需提前24小时通过手机银行或客服热线进行报备，否则交易可能被拒绝。"

切块1： "用户在使用信用卡进行境外消费时，若单笔交易金额超过等值5000美元，
        需提前24小时通过手机银行"  [截断]

切块2： "或客服热线进行报备，否则交易可能被拒绝。"  [失去主语和条件]
```

检索"境外消费需要报备吗？"时，切块1的语义是"需要提前24小时通过手机银行"——手机银行做什么？语义不完整。切块2更是主语缺失，无法理解。

**模式二：指代消解失败**

```
原文： "Transformer架构由Vaswani等人在2017年提出。它彻底改变了NLP领域。"

切块1： "Transformer架构由Vaswani等人在2017年提出。"
切块2： "它彻底改变了NLP领域。"
```

"它"指代什么？单独看切块2完全无法解析。当用户问"什么技术改变了NLP领域？"时，切块2的向量表征与查询的相似度会很低，导致关键信息漏召回。

**模式三：逻辑关系断裂**

```
原文： "本产品适用于18-65周岁人群。但孕妇、高血压患者及心脏病患者禁用。"

切块1： "本产品适用于18-65周岁人群。"  [正面描述]
切块2： "但孕妇、高血压患者及心脏病患者禁用。"  [负面限制]
```

检索"60岁高血压老人能用吗？"时，切块1被召回，系统回答"适用"，完全忽略了切块2的禁用条件。这在医疗、法律场景中是致命错误。

### 2.2 向量表征的扭曲

Embedding模型（如BGE、OpenAI text-embedding-3）在编码时，高度依赖**局部上下文**。当文本被强行截断：

- **边界token的表征失真**：句子开头和结尾的token通常承载关键语义角色（主语、宾语、否定词），截断导致这些token失去上下文支撑
- **全局语义结构破坏**：长距离依赖（如"虽然...但是..."）被切断后，chunk的向量方向偏离原始语义
- **噪声引入**：半截句子可能被模型理解为完全不同的含义

实验数据显示，在NQ（Natural Questions）数据集上，固定长度分块（512 tokens）相比语义边界分块，**召回率下降12-18%**，**答案准确率下降23%**。

## 三、解决方案：语义感知的智能分块体系

语义感知分块的核心思想是：**分块的边界应该是语义边界，而非token边界**。这需要从文本结构、语义连贯性、任务需求三个维度设计策略。

### 3.1 基础层：基于文档结构的边界识别

#### 3.1.1 显式结构标记

大多数业务文档具有明确的结构层级：

```python
import re
from typing import List, Dict
from dataclasses import dataclass

@dataclass
class Chunk:
    content: str
    metadata: Dict
    level: int  # 结构层级：0=文档, 1=章节, 2=小节, 3=段落
    
class StructureAwareChunker:
    def __init__(self):
        # 匹配Markdown/HTML/Word导出的常见标题格式
        self.heading_patterns = [
            r'^#{1,6}\s+(.+)$',           # Markdown
            r'^<h[1-6][^>]*>(.+)</h[1-6]>',  # HTML
            r'^(\d+\.)+\s+(.+)$',         # 数字编号 1. 1.1 1.1.1
            r'^[一二三四五六七八九十]+[、.]\s*(.+)$',  # 中文编号
        ]
        
    def extract_structure(self, text: str) -> List[Dict]:
        """提取文档的结构树"""
        lines = text.split('\n')
        structure = []
        current_path = []  # 当前层级路径，用于构建父子关系
        
        for i, line in enumerate(lines):
            for pattern in self.heading_patterns:
                match = re.match(pattern, line.strip())
                if match:
                    level = self._infer_level(line, match)
                    node = {
                        'type': 'heading',
                        'level': level,
                        'title': match.group(1) if len(match.groups()) > 0 else match.group(0),
                        'line_num': i,
                        'content_start': i + 1
                    }
                    # 维护层级路径
                    while len(current_path) >= level:
                        current_path.pop()
                    current_path.append(node)
                    node['path'] = [n['title'] for n in current_path]
                    structure.append(node)
                    break
                    
        return structure
    
    def _infer_level(self, line: str, match) -> int:
        """推断标题层级"""
        if line.startswith('#'):
            return line.index(' ')  # Markdown: #数量即层级
        elif line.startswith('<'):
            return int(re.search(r'h([1-6])', line).group(1))
        else:
            # 根据编号深度判断：1. -> 1, 1.1 -> 2, 1.1.1 -> 3
            dots = line.count('.')
            return dots + 1
            
    def chunk_by_structure(self, text: str, max_tokens: int = 1024) -> List[Chunk]:
        """基于结构的分块，优先保持章节完整性"""
        structure = self.extract_structure(text)
        lines = text.split('\n')
        chunks = []
        
        for i, node in enumerate(structure):
            # 确定当前章节的文本范围
            start_line = node['content_start']
            end_line = structure[i+1]['line_num'] if i+1 < len(structure) else len(lines)
            
            section_content = '\n'.join(lines[start_line:end_line])
            
            # 如果章节过长，进行内部切分
            if self._estimate_tokens(section_content) > max_tokens:
                sub_chunks = self._split_section(
                    section_content, 
                    node['path'],
                    max_tokens
                )
                chunks.extend(sub_chunks)
            else:
                chunks.append(Chunk(
                    content=section_content,
                    metadata={
                        'path': node['path'],
                        'level': node['level'],
                        'heading': node['title']
                    },
                    level=node['level']
                ))
                
        return chunks
    
    def _split_section(self, content: str, path: List[str], max_tokens: int) -> List[Chunk]:
        """对过长章节进行内部切分，优先按段落边界"""
        paragraphs = content.split('\n\n')
        current_chunk = []
        current_tokens = 0
        chunks = []
        
        for para in paragraphs:
            para_tokens = self._estimate_tokens(para)
            
            if current_tokens + para_tokens > max_tokens and current_chunk:
                # 保存当前chunk
                chunks.append(Chunk(
                    content='\n\n'.join(current_chunk),
                    metadata={
                        'path': path,
                        'section_heading': path[-1] if path else '',
                        'chunk_index': len(chunks)
                    },
                    level=len(path)
                ))
                current_chunk = [para]
                current_tokens = para_tokens
            else:
                current_chunk.append(para)
                current_tokens += para_tokens
                
        if current_chunk:
            chunks.append(Chunk(
                content='\n\n'.join(current_chunk),
                metadata={
                    'path': path,
                    'section_heading': path[-1] if path else '',
                    'chunk_index': len(chunks)
                },
                level=len(path)
            ))
            
        return chunks
    
    def _estimate_tokens(self, text: str) -> int:
        """粗略估计token数量（中文1字≈1.5token，英文1词≈1.3token）"""
        import re
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        english_words = len(re.findall(r'[a-zA-Z]+', text))
        return int(chinese_chars * 1.5 + english_words * 1.3)
```

**关键设计点**：

1. **层级继承**：每个chunk携带`path`元数据，记录从根到当前节点的完整路径。这使得检索时可以重建上下文，也支持按层级过滤（如"只查第三章的内容"）。

2. **章节优先**：尽量保持章节完整性。只有当单个章节超过`max_tokens`时才内部切分，且切分点优先选择段落边界（`\n\n`）。

3. **元数据丰富**：`level`、`heading`、`chunk_index`等元数据支持后续的过滤、排序和上下文组装。

#### 3.1.2 针对特定文档格式的解析器

不同格式需要专门的解析策略：

**PDF/Word文档**：

```python
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

class DocumentParser:
    def parse_pdf_with_layout(self, file_path: str) -> List[Dict]:
        """保留布局信息的PDF解析"""
        import fitz  # PyMuPDF
        
        doc = fitz.open(file_path)
        elements = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # 提取文本块（带坐标信息）
            blocks = page.get_text("dict")["blocks"]
            
            for block in blocks:
                if "lines" in block:
                    # 根据y坐标判断是否为标题（通常字体更大或在页面顶部）
                    text = "".join([span["text"] for line in block["lines"] 
                                   for span in line["spans"]])
                    
                    # 推断元素类型
                    element_type = self._classify_element(block, text)
                    
                    elements.append({
                        'text': text,
                        'type': element_type,  # 'heading', 'paragraph', 'table', 'list'
                        'page': page_num + 1,
                        'bbox': block['bbox'],  # 边界框坐标
                        'font_size': max([span["size"] for line in block["lines"] 
                                         for span in line["spans"]])
                    })
                    
        # 按阅读顺序（从上到下，从左到右）排序
        elements.sort(key=lambda x: (x['page'], x['bbox'][1], x['bbox'][0]))
        
        return self._elements_to_chunks(elements)
    
    def _classify_element(self, block: Dict, text: str) -> str:
        """基于视觉特征分类元素"""
        spans = [span for line in block["lines"] for span in line["spans"]]
        avg_font_size = sum(s["size"] for s in spans) / len(spans)
        is_bold = any("Bold" in s["font"] for s in spans)
        
        # 启发式规则
        if avg_font_size > 14 or (avg_font_size > 12 and is_bold):
            if len(text) < 100:  # 标题通常较短
                return 'heading'
        
        if text.strip().startswith(('•', '-', '*', '1.', '2.', '(1)', '①')):
            return 'list'
            
        if '\t' in text or '  ' in text:  # 多列或表格特征
            # 进一步检查是否表格
            lines = text.split('\n')
            if len(lines) > 2 and all('|' in l or '\t' in l for l in lines[:3]):
                return 'table'
                
        return 'paragraph'
    
    def _elements_to_chunks(self, elements: List[Dict]) -> List[Chunk]:
        """将元素序列转换为语义chunk"""
        chunks = []
        current_section = []
        current_heading = "文档开头"
        
        for elem in elements:
            if elem['type'] == 'heading':
                # 保存当前section
                if current_section:
                    content = '\n'.join([e['text'] for e in current_section])
                    chunks.append(Chunk(
                        content=content,
                        metadata={
                            'heading': current_heading,
                            'page_range': (current_section[0]['page'], 
                                        current_section[-1]['page']),
                            'types': list(set(e['type'] for e in current_section))
                        },
                        level=1
                    ))
                current_heading = elem['text']
                current_section = [elem]
            else:
                current_section.append(elem)
                
        # 处理最后一个section
        if current_section:
            content = '\n'.join([e['text'] for e in current_section])
            chunks.append(Chunk(
                content=content,
                metadata={
                    'heading': current_heading,
                    'page_range': (current_section[0]['page'], 
                                current_section[-1]['page']),
                    'types': list(set(e['type'] for e in current_section))
                },
                level=1
            ))
            
        return chunks
```

**表格处理**：

表格是RAG的难点。直接转成文本会丢失结构，需要特殊处理：

```python
class TableChunker:
    def __init__(self):
        self.llm = None  # 可选：用于生成表格描述
        
    def table_to_structured_chunk(self, table_data: List[List[str]], 
                                   caption: str = "",
                                   surrounding_text: str = "") -> Chunk:
        """将表格转换为结构化的chunk"""
        
        # 策略1：Markdown格式保留结构
        md_table = self._to_markdown(table_data)
        
        # 策略2：生成自然语言描述（用于语义检索）
        description = self._generate_description(table_data, caption)
        
        # 策略3：提取关键事实（三元组形式）
        facts = self._extract_facts(table_data)
        
        content = f"""【表格】{caption}

结构数据：
{md_table}

自然语言描述：
{description}

关键事实：
{facts}

上下文：
{surrounding_text}
"""
        return Chunk(
            content=content,
            metadata={
                'type': 'table',
                'caption': caption,
                'rows': len(table_data),
                'cols': len(table_data[0]) if table_data else 0,
                'facts': facts  # 可用于图谱构建
            },
            level=2
        )
    
    def _to_markdown(self, table: List[List[str]]) -> str:
        """转换为Markdown表格"""
        if not table:
            return ""
        
        md = []
        md.append('| ' + ' | '.join(table[0]) + ' |')
        md.append('|' + '|'.join(['---' for _ in table[0]]) + '|')
        
        for row in table[1:]:
            md.append('| ' + ' | '.join(row) + ' |')
            
        return '\n'.join(md)
    
    def _generate_description(self, table: List[List[str]], caption: str) -> str:
        """生成表格的自然语言描述（可用于向量检索）"""
        if not table or len(table) < 2:
            return caption
            
        headers = table[0]
        sample_rows = table[1:4]  # 取前3行作为示例
        
        description = f"这是一个关于{caption}的表格，包含{len(table)-1}行数据。"
        description += f"列包括：{', '.join(headers)}。"
        
        # 提取统计信息（如果是数值列）
        for col_idx, header in enumerate(headers):
            values = [row[col_idx] for row in table[1:] if len(row) > col_idx]
            numeric_values = self._extract_numbers(values)
            
            if numeric_values:
                description += f"{header}的范围是{min(numeric_values)}到{max(numeric_values)}。"
                
        return description
    
    def _extract_facts(self, table: List[List[str]]) -> List[str]:
        """将表格行转换为事实陈述（用于精确检索）"""
        if len(table) < 2:
            return []
            
        headers = table[0]
        facts = []
        
        for row in table[1:]:
            if len(row) >= 2:
                # 主语通常是第一列
                subject = row[0]
                for i, value in enumerate(row[1:], 1):
                    if i < len(headers):
                        fact = f"{subject}的{headers[i]}是{value}"
                        facts.append(fact)
                        
        return facts
    
    def _extract_numbers(self, values: List[str]) -> List[float]:
        """从字符串列表中提取数值"""
        import re
        numbers = []
        for v in values:
            matches = re.findall(r'\d+\.?\d*', v.replace(',', ''))
            numbers.extend([float(m) for m in matches])
        return numbers
```

### 3.2 进阶层：基于语义相似度的动态切分

当文档缺乏显式结构（如聊天记录、访谈录音转写、非结构化网页），需要基于语义连贯性进行切分。

#### 3.2.1 句子级Embedding相似度切分

核心思想：**相邻句子的语义相似度骤降处，通常是话题转换点**。

```python
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Tuple
import torch

class SemanticChunker:
    def __init__(self, 
                 embedding_model: str = 'BAAI/bge-large-zh-v1.5',
                 similarity_threshold: float = 0.7,
                 min_chunk_size: int = 3,
                 max_chunk_size: int = 10):
        """
        Args:
            similarity_threshold: 相似度低于此值视为边界
            min_chunk_size: 每个chunk最少句子数（避免过碎）
            max_chunk_size: 每个chunk最多句子数（避免过长）
        """
        self.model = SentenceTransformer(embedding_model)
        self.threshold = similarity_threshold
        self.min_size = min_chunk_size
        self.max_size = max_chunk_size
        
    def split(self, text: str) -> List[Chunk]:
        """主入口：将文本切分为语义连贯的chunk"""
        # 1. 句子切分
        sentences = self._split_sentences(text)
        
        if len(sentences) <= self.min_size:
            return [Chunk(
                content=text,
                metadata={'method': 'semantic', 'num_sentences': len(sentences)},
                level=1
            )]
        
        # 2. 计算句子embedding
        embeddings = self.model.encode(sentences, convert_to_tensor=True)
        
        # 3. 计算相邻句子相似度
        similarities = self._compute_similarities(embeddings)
        
        # 4. 动态规划找最优切分点
        boundaries = self._find_boundaries_dp(sentences, similarities)
        
        # 5. 生成chunks
        chunks = []
        start = 0
        for end in boundaries + [len(sentences)]:
            chunk_sentences = sentences[start:end]
            chunk_text = ''.join(chunk_sentences)  # 中文无需空格
            
            # 计算chunk的统计信息
            avg_similarity = np.mean(similarities[start:end-1]) if end-start > 1 else 1.0
            
            chunks.append(Chunk(
                content=chunk_text,
                metadata={
                    'method': 'semantic',
                    'num_sentences': len(chunk_sentences),
                    'sentence_range': (start, end),
                    'avg_internal_similarity': float(avg_similarity),
                    'boundary_similarity': float(similarities[end-1]) if end < len(sentences) else None
                },
                level=1
            ))
            start = end
            
        return chunks
    
    def _split_sentences(self, text: str) -> List[str]:
        """中文/英文句子切分"""
        import re
        
        # 保留分隔符
        pattern = r'([。！？.!?])([^”’])'
        text = re.sub(pattern, r'\1\n\2', text)
        text = re.sub(r'(\.{6})([^”’])', r'\1\n\2', text)  # 省略号
        text = re.sub(r'([。！？.!?][”’])([^，。！？.!?])', r'\1\n\2', text)
        
        sentences = [s.strip() for s in text.split('\n') if s.strip()]
        return sentences
    
    def _compute_similarities(self, embeddings: torch.Tensor) -> np.ndarray:
        """计算相邻句子的余弦相似度"""
        # 归一化
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        # 计算相邻相似度
        similarities = torch.sum(embeddings[:-1] * embeddings[1:], dim=1)
        
        return similarities.cpu().numpy()
    
    def _find_boundaries_dp(self, 
                           sentences: List[str], 
                           similarities: np.ndarray) -> List[int]:
        """动态规划寻找最优切分边界
        
        目标：在满足min/max约束的前提下，最小化chunk内部差异，最大化chunk间差异
        """
        n = len(sentences)
        
        # cost[i][j] = 将句子i到j-1作为一个chunk的代价（内部不相似度）
        cost = np.zeros((n, n+1))
        for i in range(n):
            for j in range(i+1, min(i+self.max_size+1, n+1)):
                if j - i == 1:
                    cost[i][j] = 0
                else:
                    # 使用平均相似度的负值作为代价（越相似代价越低）
                    seg_sim = similarities[i:j-1]
                    cost[i][j] = 1.0 - np.mean(seg_sim)
        
        # dp[i] = 切分到第i个句子的最小总代价
        dp = [float('inf')] * (n + 1)
        parent = [-1] * (n + 1)
        dp[0] = 0
        
        for i in range(1, n + 1):
            # 尝试所有可能的切分点
            for j in range(max(0, i - self.max_size), 
                          max(0, i - self.min_size + 1)):
                if dp[j] + cost[j][i] < dp[i]:
                    dp[i] = dp[j] + cost[j][i]
                    parent[i] = j
        
        # 回溯得到切分点
        boundaries = []
        cur = n
        while parent[cur] != -1:
            boundaries.append(cur)
            cur = parent[cur]
            
        return sorted(boundaries[:-1])  # 去掉最后一个（即n）
    
    def visualize_boundaries(self, text: str) -> None:
        """可视化展示切分边界（用于调试）"""
        import matplotlib.pyplot as plt
        
        sentences = self._split_sentences(text)
        embeddings = self.model.encode(sentences, convert_to_tensor=True)
        similarities = self._compute_similarities(embeddings)
        boundaries = self._find_boundaries_dp(sentences, similarities)
        
        plt.figure(figsize=(14, 4))
        plt.plot(range(len(similarities)), similarities, 'b-', label='Similarity')
        plt.axhline(y=self.threshold, color='r', linestyle='--', label='Threshold')
        
        for b in boundaries:
            plt.axvline(x=b-0.5, color='g', linestyle='-', alpha=0.7, label='Boundary' if b == boundaries[0] else "")
            
        plt.xlabel('Sentence Index')
        plt.ylabel('Cosine Similarity')
        plt.title('Semantic Boundaries Detection')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('/mnt/kimi/output/semantic_boundaries.png', dpi=150)
        plt.show()
        
        print(f"检测到 {len(boundaries)} 个边界，生成 {len(boundaries)+1} 个chunks")
```

**关键优化点**：

1. **动态规划替代贪心**：贪心策略（遇到低相似度就切）可能导致碎片化。DP在保证min/max长度约束下，全局优化切分质量。

2. **自适应阈值**：固定阈值对长文档和短文档效果不同。可以基于相似度分布动态调整：
   ```python
   def adaptive_threshold(similarities: np.ndarray) -> float:
       """基于统计分布计算自适应阈值"""
       mean = np.mean(similarities)
       std = np.std(similarities)
       # 取均值减1个标准差作为阈值
       return max(0.5, mean - std)
   ```

3. **滑动窗口平滑**：单句相似度可能受噪声影响，使用3句窗口平均：
   ```python
   smoothed = np.convolve(similarities, np.ones(3)/3, mode='same')
   ```

#### 3.2.2 基于主题模型的切分

对于长文档，可以使用更粗粒度的主题检测：

```python
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

class TopicAwareChunker:
    def __init__(self, n_topics: int = 5, window_size: int = 3):
        self.n_topics = n_topics
        self.window_size = window_size
        
    def split(self, paragraphs: List[str]) -> List[Chunk]:
        """基于主题转换的段落级切分"""
        # 1. 训练LDA模型
        vectorizer = CountVectorizer(max_df=0.7, min_df=2, stop_words='english')
        doc_term_matrix = vectorizer.fit_transform(paragraphs)
        
        lda = LatentDirichletAllocation(
            n_components=self.n_topics,
            random_state=42,
            max_iter=10
        )
        topic_distributions = lda.fit_transform(doc_term_matrix)
        
        # 2. 检测主题转换点
        boundaries = []
        for i in range(1, len(paragraphs)):
            # 计算相邻窗口的主题分布差异（JS散度）
            prev_dist = np.mean(topic_distributions[max(0,i-self.window_size):i], axis=0)
            next_dist = np.mean(topic_distributions[i:min(len(paragraphs),i+self.window_size)], axis=0)
            
            divergence = self._js_divergence(prev_dist, next_dist)
            
            if divergence > 0.3:  # 主题显著变化
                boundaries.append(i)
        
        # 3. 生成chunks
        chunks = []
        start = 0
        for end in boundaries + [len(paragraphs)]:
            content = '\n\n'.join(paragraphs[start:end])
            dominant_topic = int(np.argmax(np.mean(topic_distributions[start:end], axis=0)))
            
            chunks.append(Chunk(
                content=content,
                metadata={
                    'method': 'topic',
                    'dominant_topic': dominant_topic,
                    'topic_mixture': topic_distributions[start:end].mean(axis=0).tolist(),
                    'num_paragraphs': end - start
                },
                level=1
            ))
            start = end
            
        return chunks
    
    def _js_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """计算JS散度"""
        m = 0.5 * (p + q)
        kl_pm = np.sum(p * np.log(p / m + 1e-10))
        kl_qm = np.sum(q * np.log(q / m + 1e-10))
        return 0.5 * (kl_pm + kl_qm)
```

### 3.3 高级层：层次化分块策略（Parent-Child Chunking）

这是生产环境中最有效的策略：**检索时用细粒度单元，生成时用粗粒度上下文**。

```python
from typing import Optional
import hashlib

class HierarchicalChunker:
    def __init__(self,
                 parent_chunk_size: int = 1024,  # 父块：用于生成上下文
                 child_chunk_size: int = 256,    # 子块：用于检索
                 overlap_ratio: float = 0.2):     # 子块间重叠比例
        self.parent_size = parent_chunk_size
        self.child_size = child_chunk_size
        self.overlap = int(child_chunk_size * overlap_ratio)
        
    def create_hierarchy(self, text: str, doc_id: str) -> Dict:
        """创建三层结构：文档 -> 父块 -> 子块"""
        
        # 第一层：父块（按语义/结构切分）
        parent_chunks = self._create_parents(text)
        
        hierarchy = {
            'doc_id': doc_id,
            'doc_summary': self._generate_summary(text),
            'parents': []
        }
        
        for parent_idx, parent in enumerate(parent_chunks):
            parent_id = f"{doc_id}_p{parent_idx}"
            
            # 第二层：子块（滑动窗口切分）
            children = self._create_children(
                parent['content'], 
                parent_id,
                start_offset=parent['start_char']
            )
            
            parent_node = {
                'parent_id': parent_id,
                'content': parent['content'],
                'metadata': {
                    **parent['metadata'],
                    'level': 'parent',
                    'child_ids': [c['child_id'] for c in children],
                    'summary': self._generate_summary(parent['content'])
                },
                'children': children
            }
            
            hierarchy['parents'].append(parent_node)
            
        return hierarchy
    
    def _create_parents(self, text: str) -> List[Dict]:
        """创建父块：优先保持语义完整性"""
        # 使用之前定义的StructureAwareChunker或SemanticChunker
        chunker = StructureAwareChunker()
        chunks = chunker.chunk_by_structure(text, max_tokens=self.parent_size)
        
        return [{
            'content': c.content,
            'metadata': c.metadata,
            'start_char': 0,  # 实际需要计算字符偏移
        } for c in chunks]
    
    def _create_children(self, 
                        parent_text: str, 
                        parent_id: str,
                        start_offset: int = 0) -> List[Dict]:
        """创建子块：滑动窗口，保证检索覆盖率"""
        # 按句子切分
        sentences = self._split_sentences(parent_text)
        
        children = []
        current_sentences = []
        current_tokens = 0
        child_idx = 0
        
        for i, sent in enumerate(sentences):
            sent_tokens = len(sent)  # 粗略估计
            
            if current_tokens + sent_tokens > self.child_size and current_sentences:
                # 保存当前子块
                child_text = ''.join(current_sentences)
                child_id = f"{parent_id}_c{child_idx}"
                
                children.append({
                    'child_id': child_id,
                    'content': child_text,
                    'metadata': {
                        'parent_id': parent_id,
                        'level': 'child',
                        'child_index': child_idx,
                        'char_range': (
                            start_offset + parent_text.find(child_text),
                            start_offset + parent_text.find(child_text) + len(child_text)
                        ),
                        'key_sentence': self._extract_key_sentence(current_sentences),
                        'entities': self._extract_entities(child_text)
                    }
                })
                
                # 滑动窗口：保留最后overlap部分
                overlap_tokens = 0
                overlap_sentences = []
                for s in reversed(current_sentences):
                    if overlap_tokens >= self.overlap:
                        break
                    overlap_sentences.insert(0, s)
                    overlap_tokens += len(s)
                
                current_sentences = overlap_sentences
                current_tokens = overlap_tokens
                child_idx += 1
            
            current_sentences.append(sent)
            current_tokens += sent_tokens
        
        # 处理最后一个子块
        if current_sentences:
            child_text = ''.join(current_sentences)
            child_id = f"{parent_id}_c{child_idx}"
            children.append({
                'child_id': child_id,
                'content': child_text,
                'metadata': {
                    'parent_id': parent_id,
                    'level': 'child',
                    'child_index': child_idx,
                    'char_range': (
                        start_offset + parent_text.find(child_text),
                        start_offset + parent_text.find(child_text) + len(child_text)
                    ),
                    'key_sentence': self._extract_key_sentence(current_sentences),
                    'entities': self._extract_entities(child_text)
                }
            })
        
        return children
    
    def _extract_key_sentence(self, sentences: List[str]) -> str:
        """提取关键句（通常是第一句或包含核心实体的句子）"""
        if not sentences:
            return ""
        
        # 简单策略：返回第一句，如果太短则合并第二句
        key = sentences[0]
        if len(key) < 20 and len(sentences) > 1:
            key += sentences[1]
        return key[:100]  # 限制长度
    
    def _extract_entities(self, text: str) -> List[str]:
        """提取关键实体（可用NER模型）"""
        # 简化版：提取引号内容、大写短语、数字
        import re
        entities = []
        
        # 引号内容
        entities.extend(re.findall(r'["「『](.+?)["」』]', text))
        
        # 英文大写短语（可能是专有名词）
        entities.extend(re.findall(r'\b[A-Z][A-Z\s]{2,}\b', text))
        
        # 数字+单位
        entities.extend(re.findall(r'\d+\.?\d*\s*[个位只件条人元年]%?', text))
        
        return list(set(entities))[:5]  # 去重，限制数量
    
    def _generate_summary(self, text: str, max_length: int = 100) -> str:
        """生成文本摘要（可用LLM或抽取式）"""
        # 简化版：取前两句
        sentences = self._split_sentences(text)
        summary = ''.join(sentences[:2])
        return summary[:max_length]
    
    def _split_sentences(self, text: str) -> List[str]:
        """句子切分（同前）"""
        import re
        pattern = r'([。！？.!?])([^”’])'
        text = re.sub(pattern, r'\1\n\2', text)
        text = re.sub(r'(\.{6})([^”’])', r'\1\n\2', text)
        text = re.sub(r'([。！？.!?][”’])([^，。！？.!?])', r'\1\n\2', text)
        return [s.strip() for s in text.split('\n') if s.strip()]
    
    def build_index_structure(self, hierarchy: Dict) -> Dict:
        """构建用于索引的数据结构"""
        index_docs = []
        
        for parent in hierarchy['parents']:
            # 父块用于生成上下文（可选索引，用于全局查询）
            index_docs.append({
                'id': parent['parent_id'],
                'text': parent['content'],
                'metadata': {k: v for k, v in parent['metadata'].items() 
                           if k != 'children'},
                'type': 'parent'
            })
            
            # 子块用于检索（主要索引）
            for child in parent['children']:
                # 增强子块文本：关键句 + 内容 + 实体标签
                enhanced_text = f"{child['metadata']['key_sentence']}。{child['content']} 相关实体：{', '.join(child['metadata']['entities'])}"
                
                index_docs.append({
                    'id': child['child_id'],
                    'text': enhanced_text,
                    'metadata': child['metadata'],
                    'type': 'child'
                })
        
        return {
            'doc_id': hierarchy['doc_id'],
            'index_documents': index_docs,
            'hierarchy_map': self._build_hierarchy_map(hierarchy)
        }
    
    def _build_hierarchy_map(self, hierarchy: Dict) -> Dict:
        """构建ID映射关系，用于检索后重建上下文"""
        mapping = {}
        for parent in hierarchy['parents']:
            for child in parent['children']:
                mapping[child['child_id']] = {
                    'parent_id': parent['parent_id'],
                    'parent_content': parent['content'],
                    'siblings': [c['child_id'] for c in parent['children']],
                    'doc_summary': hierarchy['doc_summary']
                }
        return mapping
```

**层次化检索流程**：

```python
class HierarchicalRetriever:
    def __init__(self, vector_store, hierarchy_map: Dict):
        self.vector_store = vector_store
        self.hierarchy_map = hierarchy_map
        
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """层次化检索：子块召回 -> 父块组装"""
        
        # 1. 在子块上检索（精准）
        child_results = self.vector_store.search(
            query=query,
            filter={'type': 'child'},
            top_k=top_k
        )
        
        # 2. 按父块分组
        parent_groups = {}
        for res in child_results:
            parent_id = res['metadata']['parent_id']
            if parent_id not in parent_groups:
                parent_groups[parent_id] = {
                    'parent_content': self.hierarchy_map[res['id']]['parent_content'],
                    'children': [],
                    'scores': []
                }
            parent_groups[parent_id]['children'].append(res)
            parent_groups[parent_id]['scores'].append(res['score'])
        
        # 3. 组装结果：父块内容 + 高亮相关子块
        final_results = []
        for parent_id, group in parent_groups.items():
            # 计算父块整体相关性（子块分数的加权平均）
            avg_score = np.mean(group['scores'])
            
            # 标记相关片段
            parent_content = group['parent_content']
            for child in group['children']:
                # 在父内容中定位子内容并标记
                child_text = child['metadata'].get('key_sentence', child['text'][:50])
                parent_content = parent_content.replace(
                    child_text, 
                    f"【相关】{child_text}【/相关】"
                )
            
            final_results.append({
                'content': parent_content,
                'score': avg_score,
                'source': parent_id,
                'matched_children': len(group['children']),
                'metadata': {
                    'hierarchy': 'parent',
                    'child_details': group['children']
                }
            })
        
        # 按分数排序
        final_results.sort(key=lambda x: x['score'], reverse=True)
        return final_results[:top_k]
```

### 3.4 特殊场景：对话与多模态内容

#### 3.4.1 对话内容的分块

聊天记录具有明确的轮次结构，不能简单按token切：

```python
class ConversationChunker:
    def __init__(self, 
                 max_turns_per_chunk: int = 5,
                 context_window_turns: int = 2):
        self.max_turns = max_turns_per_chunk
        self.context_window = context_window_turns
        
    def chunk_conversation(self, messages: List[Dict]) -> List[Chunk]:
        """
        messages: [{'role': 'user'/'assistant', 'content': str, 'timestamp': int}, ...]
        """
        chunks = []
        
        # 按轮次分组（用户+助手为一轮）
        turns = self._group_into_turns(messages)
        
        # 滑动窗口切分，保留上下文
        for i in range(0, len(turns), self.max_turns - self.context_window):
            start_turn = max(0, i - self.context_window)
            end_turn = min(i + self.max_turns, len(turns))
            
            chunk_turns = turns[start_turn:end_turn]
            content = self._format_turns(chunk_turns)
            
            # 提取对话主题（可用LLM或关键词）
            topic = self._extract_topic(chunk_turns)
            
            chunks.append(Chunk(
                content=content,
                metadata={
                    'type': 'conversation',
                    'turn_range': (start_turn, end_turn),
                    'num_turns': len(chunk_turns),
                    'topic': topic,
                    'participants': list(set(
                        m['role'] for turn in chunk_turns for m in turn
                    )),
                    'time_range': (
                        chunk_turns[0][0]['timestamp'],
                        chunk_turns[-1][-1]['timestamp']
                    ) if chunk_turns else None
                },
                level=1
            ))
            
        return chunks
    
    def _group_into_turns(self, messages: List[Dict]) -> List[List[Dict]]:
        """将消息列表按轮次分组"""
        turns = []
        current_turn = []
        
        for msg in messages:
            if msg['role'] == 'user' and current_turn:
                # 新轮次开始
                turns.append(current_turn)
                current_turn = [msg]
            else:
                current_turn.append(msg)
        
        if current_turn:
            turns.append(current_turn)
            
        return turns
    
    def _format_turns(self, turns: List[List[Dict]]) -> str:
        """格式化轮次为可读文本"""
        lines = []
        for i, turn in enumerate(turns):
            for msg in turn:
                role = "用户" if msg['role'] == 'user' else "助手"
                lines.append(f"[{role}] {msg['content']}")
            if i < len(turns) - 1:
                lines.append("---")
        return '\n'.join(lines)
    
    def _extract_topic(self, turns: List[List[Dict]]) -> str:
        """提取对话主题（简化版：取第一轮用户问题的前10个字）"""
        for turn in turns:
            for msg in turn:
                if msg['role'] == 'user':
                    return msg['content'][:20]
        return "未命名对话"
```

## 四、完整实战示例

### 4.1 场景：产品手册智能问答

**原始文档**（节选）：

```
第一章 产品概述

1.1 产品简介
本产品是一款面向企业级用户的智能数据分析平台，支持多源数据接入、实时计算和可视化展示。适用于金融、零售、制造等行业。

1.2 系统要求
- 操作系统：Linux CentOS 7.6+ 或 Ubuntu 18.04+
- 内存：最低16GB，推荐32GB
- 存储：系统盘100GB SSD，数据盘根据数据量配置

第二章 安装部署

2.1 单机部署
适合测试环境和小规模生产环境。部署步骤如下：
1. 下载安装包并解压
2. 运行 install.sh 脚本
3. 配置数据库连接
注意：单机部署不支持高可用，若需高可用请参考2.2节集群部署。

2.2 集群部署
适合大规模生产环境，支持水平扩展和故障转移...
```

### 4.2 不同分块策略对比

**策略A：固定长度（512 tokens）**

```python
# 切分结果
chunk1: "第一章 产品概述 1.1 产品简介 本产品是一款面向企业级用户的智能数据分析平台，支持多源数据接入、实时计算和可视化展示。适用于金融、零售、制造等行业。 1.2 系统要求 - 操作系统：Linux CentOS 7.6+ 或 Ubuntu 18.04+ - 内存：最低16GB，推荐32GB - 存储：系统盘100GB SSD，数据盘根据"
chunk2: "数据量配置 第二章 安装部署 2.1 单机部署 适合测试环境和小规模生产环境。部署步骤如下： 1. 下载安装包并解压 2. 运行 install.sh 脚本 3. 配置数据库连接 注意：单机部署不支持高可用，若需高可用请参考2.2节集群部署。 2.2 集群部署 适合大规模生产环境，支持水平扩展和故障转"
```

**问题**：
- chunk1在"数据盘根据"处截断，"数据量配置"被分到chunk2，导致"存储要求"信息不完整
- "注意：单机部署不支持高可用"这一关键警告被截断在chunk2开头，可能丢失上下文

**策略B：语义感知分块（StructureAwareChunker）**

```python
chunks = [
    {
        "content": "1.1 产品简介\n本产品是一款面向企业级用户的智能数据分析平台，支持多源数据接入、实时计算和可视化展示。适用于金融、零售、制造等行业。",
        "metadata": {
            "path": ["第一章 产品概述", "1.1 产品简介"],
            "level": 2
        }
    },
    {
        "content": "1.2 系统要求\n- 操作系统：Linux CentOS 7.6+ 或 Ubuntu 18.04+\n- 内存：最低16GB，推荐32GB\n- 存储：系统盘100GB SSD，数据盘根据数据量配置",
        "metadata": {
            "path": ["第一章 产品概述", "1.2 系统要求"],
            "level": 2
        }
    },
    {
        "content": "2.1 单机部署\n适合测试环境和小规模生产环境。部署步骤如下：\n1. 下载安装包并解压\n2. 运行 install.sh 脚本\n3. 配置数据库连接\n注意：单机部署不支持高可用，若需高可用请参考2.2节集群部署。",
        "metadata": {
            "path": ["第二章 安装部署", "2.1 单机部署"],
            "level": 2
        }
    }
]
```

**优势**：
- 每个chunk语义完整，"注意"警告与所属章节绑定
- 元数据包含完整路径，检索时可重建上下文

**策略C：层次化分块（HierarchicalChunker）**

```python
hierarchy = {
    "parent_0": {
        "content": "第一章 产品概述\n\n1.1 产品简介\n本产品是一款面向企业级用户的智能数据分析平台...\n\n1.2 系统要求\n- 操作系统：Linux CentOS 7.6+...",
        "children": [
            {
                "child_id": "p0_c0",
                "content": "1.1 产品简介 本产品是一款面向企业级用户的智能数据分析平台，支持多源数据接入、实时计算和可视化展示。",
                "metadata": {
                    "key_sentence": "本产品是一款面向企业级用户的智能数据分析平台",
                    "entities": ["智能数据分析平台", "多源数据接入", "实时计算", "可视化展示"]
                }
            },
            {
                "child_id": "p0_c1", 
                "content": "1.2 系统要求 - 操作系统：Linux CentOS 7.6+ 或 Ubuntu 18.04+ - 内存：最低16GB，推荐32GB",
                "metadata": {
                    "key_sentence": "系统要求包括操作系统和内存配置",
                    "entities": ["Linux CentOS 7.6", "Ubuntu 18.04", "16GB", "32GB"]
                }
            }
        ]
    }
}
```

**检索效果对比**：

| 查询 | 固定长度 | 语义分块 | 层次化分块 |
|-----|---------|---------|-----------|
| "内存最低要求" | 召回chunk1（包含内存信息，但截断了存储部分） | 召回"1.2 系统要求"完整chunk | 召回child_c1，并自动组装parent_0提供完整上下文 |
| "单机部署注意事项" | 召回chunk2（"注意"在开头，可能丢失） | 召回"2.1 单机部署"完整chunk | 召回对应child，高亮"注意"部分，提供完整部署步骤 |

### 4.2 评估指标与测试

```python
class ChunkingEvaluator:
    def __init__(self):
        self.metrics = {
            'boundary_accuracy': [],  # 边界准确性
            'semantic_coherence': [],  # 语义连贯性
            'retrieval_recall': [],  # 检索召回率
            'answer_accuracy': []  # 最终答案准确率
        }
    
    def evaluate_boundary_accuracy(self, 
                                    chunks: List[Chunk], 
                                    gold_boundaries: List[int]) -> float:
        """评估切分边界与人工标注的匹配度"""
        predicted = set()
        for chunk in chunks:
            if 'sentence_range' in chunk.metadata:
                predicted.add(chunk.metadata['sentence_range'][1])
        
        gold = set(gold_boundaries)
        
        precision = len(predicted & gold) / len(predicted) if predicted else 0
        recall = len(predicted & gold) / len(gold) if gold else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
        
        return f1
    
    def evaluate_semantic_coherence(self, chunks: List[Chunk]) -> float:
        """评估chunk内部语义连贯性（平均内部相似度）"""
        scores = []
        model = SentenceTransformer('BAAI/bge-large-zh-v1.5')
        
        for chunk in chunks:
            sentences = self._split_sentences(chunk.content)
            if len(sentences) < 2:
                continue
                
            embeddings = model.encode(sentences)
            similarities = cosine_similarity(embeddings)
            
            # 计算平均相似度（上三角矩阵）
            avg_sim = np.mean(similarities[np.triu_indices_from(similarities, k=1)])
            scores.append(avg_sim)
        
        return np.mean(scores) if scores else 0.0
    
    def run_ablation_test(self, 
                         corpus: List[str], 
                         queries: List[Dict],
                         strategies: List[Tuple[str, callable]]) -> pd.DataFrame:
        """对比不同策略的效果"""
        results = []
        
        for strategy_name, chunker in strategies:
            for doc in corpus:
                chunks = chunker(doc)
                
                # 模拟检索和生成
                for query in queries:
                    # 检索最相关的chunk
                    retrieved = self._simulate_retrieval(chunks, query['text'])
                    
                    # 评估
                    recall = self._calculate_recall(retrieved, query['relevant_chunks'])
                    
                    results.append({
                        'strategy': strategy_name,
                        'query': query['text'],
                        'recall': recall,
                        'num_chunks': len(chunks),
                        'avg_chunk_size': np.mean([len(c.content) for c in chunks])
                    })
        
        return pd.DataFrame(results)
```

## 五、工程实践建议

### 5.1 策略选择决策树

```
开始
  │
  ├─ 文档是否有明确结构（标题、章节）？
  │    ├─ 是 → 使用 StructureAwareChunker
  │    └─ 否 → 继续判断
  │
  ├─ 内容是否为连续文本（论文、小说）？
  │    ├─ 是 → 使用 SemanticChunker（相似度切分）
  │    └─ 否 → 继续判断
  │
  ├─ 是否为对话/聊天记录？
  │    ├─ 是 → 使用 ConversationChunker
  │    └─ 否 → 使用通用 SemanticChunker
  │
  └─ 是否需要最高检索质量？
       ├─ 是 → 使用 HierarchicalChunker（层次化）
       └─ 否 → 使用 SemanticChunker
```

### 5.2 性能优化

1. **Embedding缓存**：对长文档切分句子后，缓存句子级embedding，避免重复计算
2. **并行处理**：文档级并行 + batch embedding
3. **增量更新**：只重新处理修改的章节，而非全量重建

### 5.3 与现有框架集成

**LangChain集成**：

```python
from langchain.text_splitter import TextSplitter
from langchain.schema import Document

class SemanticTextSplitter(TextSplitter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.chunker = SemanticChunker()
        
    def split_text(self, text: str) -> List[str]:
        chunks = self.chunker.split(text)
        return [c.content for c in chunks]
    
    def create_documents(self, texts: List[str], metadatas=None):
        documents = []
        for i, text in enumerate(texts):
            chunks = self.chunker.split(text)
            for chunk in chunks:
                doc = Document(
                    page_content=chunk.content,
                    metadata={**(metadatas[i] if metadatas else {}), **chunk.metadata}
                )
                documents.append(doc)
        return documents
```

**LlamaIndex集成**：

```python
from llama_index.node_parser import NodeParser
from llama_index.schema import TextNode

class SemanticNodeParser(NodeParser):
    def __init__(self):
        self.chunker = HierarchicalChunker()
        
    def _parse_nodes(self, nodes, **kwargs):
        all_nodes = []
        for node in nodes:
            hierarchy = self.chunker.create_hierarchy(
                node.text, 
                doc_id=node.id_
            )
            
            # 创建父节点（可选）
            for parent in hierarchy['parents']:
                parent_node = TextNode(
                    text=parent['content'],
                    metadata=parent['metadata'],
                    id_=parent['parent_id']
                )
                all_nodes.append(parent_node)
                
                # 创建子节点
                for child in parent['children']:
                    child_node = TextNode(
                        text=child['content'],
                        metadata=child['metadata'],
                        id_=child['child_id']
                    )
                    all_nodes.append(child_node)
        
        return all_nodes
```
