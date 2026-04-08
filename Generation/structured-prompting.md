# 结构化Prompt模板：从"玄学调参"到"工程化设计"

## 一、背景：Prompt工程的"野生时代"

### 1.1 一个典型的混乱案例

某团队的AI助手项目，3个工程师写了3版Prompt：

**工程师A的版本**：
```
帮我写个邮件，要专业一点，发给客户的，关于那个延期的事情。
```

**工程师B的版本**：
```
你是一位资深客户经理，现在需要给客户写一封正式的延期通知邮件。语气要诚恳但专业，解释清楚原因，给出补救方案，最后表达歉意。不要太长，300字以内。
```

**工程师C的版本**：
```
写邮件，延期，客户，急。
```

**结果**：
- 同一功能，输出风格天差地别
- 新人接手无从维护
- 调优全靠"语感"，无法量化迭代

这就是**非结构化Prompt**的痛点——像写散文，靠灵感，不可复现。

### 1.2 为什么需要结构化

| 问题 | 非结构化 | 结构化 |
|-----|---------|--------|
| 可维护性 | 改一处，崩三处 | 模块化，独立迭代 |
| 可交接 | 口头传承 | 文档即规范 |
| A/B测试 | 无法对比 | 控制变量，精准优化 |
| 版本管理 | 文本地狱 | 字段化，diff清晰 |
| 多语言适配 | 重写 | 替换字段即可 |

**核心洞察**：Prompt不是"提示语"，是**接口定义**——输入、处理、输出的契约。

## 二、核心框架：RTCOE模型

### 2.1 五要素定义

```
┌─────────────────────────────────────────┐
│           RTCOE 结构化框架               │
├─────────────────────────────────────────┤
│  R - Role（角色）                        │
│     你是谁？决定知识边界、语气、价值观       │
├─────────────────────────────────────────┤
│  T - Task（任务）                        │
│     要做什么？明确动作和交付物             │
├─────────────────────────────────────────┤
│  C - Context（上下文）                    │
│     背景信息？约束条件？输入数据？          │
├─────────────────────────────────────────┤
│  O - Output Format（输出格式）            │
│     长什么样？结构、类型、限制              │
├─────────────────────────────────────────┤
│  E - Example（示例）                      │
│     给样板。少样本学习，降低歧义            │
└─────────────────────────────────────────┘
```

### 2.2 对比：混乱 vs 结构化

**混乱版本**：
```
帮我分析这段用户反馈，看看有什么问题，然后给我一些建议，要具体一点，不要太笼统。
```

**结构化版本（RTCOE）**：

```markdown
## Role
你是一位资深用户体验分析师，擅长从用户反馈中识别痛点并提出可落地的改进方案。

## Task
分析给定的用户反馈文本，完成以下动作：
1. 情绪识别：判断用户整体情绪（正面/负面/中性）
2. 问题提取：列出具体抱怨点，标注严重程度（高/中/低）
3. 根因分析：推断问题背后的系统性原因
4. 改进建议：针对每个问题给出2-3条具体改进措施

## Context
- 产品类型：B2B SaaS项目管理工具
- 用户角色：项目经理（非技术背景）
- 反馈渠道：应用内弹窗调研

## Output Format
输出JSON格式：
{
  "sentiment": "负面",
  "confidence": 0.92,
  "issues": [
    {
      "description": "问题描述",
      "severity": "高",
      "category": "功能缺失|性能问题|交互设计|文档支持",
      "root_cause": "根因分析",
      "suggestions": ["建议1", "建议2"]
    }
  ],
  "priority_action": "最紧急的1项改进建议"
}

## Example
输入："这个甘特图功能太难用了，我找了半天不知道怎么导出PDF，最后只能截图发邮件，太尴尬了。"

输出：
{
  "sentiment": "负面",
  "confidence": 0.88,
  "issues": [
    {
      "description": "甘特图导出PDF功能入口不明显",
      "severity": "中",
      "category": "交互设计",
      "root_cause": "功能可见性不足，未遵循常见设计模式",
      "suggestions": [
        "在甘特图工具栏添加显式'导出'按钮",
        "支持右键菜单导出选项",
        "添加导出功能的引导提示"
      ]
    }
  ],
  "priority_action": "在甘特图界面添加显式导出按钮"
}
```

## 三、各要素设计详解

### 3.1 Role：角色定义的三层

**错误示范**：
```
你是一位专家。
```
（太泛，无边界）

**正确示范**：

```markdown
## Role
你是一位{具体领域}的{经验级别}{角色类型}，{风格特质}。

- 专业知识：{领域知识范围}
- 决策倾向：{保守/激进/平衡}
- 表达风格：{正式/亲切/技术/商业}
- 价值观：{用户优先/效率优先/安全优先}
```

**案例对比**：

| 场景 | 弱Role | 强Role |
|-----|--------|--------|
| 医疗咨询 | "你是医生" | "你是一位有10年临床经验的内科主治医师，擅长用通俗语言解释复杂病情，倾向于保守治疗，注重患者心理感受" |
| 代码审查 | "你是程序员" | "你是一位资深Python后端工程师，注重代码可读性和边界情况处理，习惯用PEP8规范，审查时会指出潜在性能陷阱" |
| 创意写作 | "你是作家" | "你是一位擅长悬疑短篇的作家，风格类似欧·亨利，喜欢在结尾反转，善用细节暗示，避免直白叙述" |

### 3.2 Task：任务描述的SMART原则

```markdown
## Task
{动作动词} + {交付物} + {成功标准} + {约束条件}

动作类型：
- 分析类：识别、提取、分类、归因、预测
- 生成类：撰写、改写、扩展、摘要、翻译
- 决策类：评估、排序、选择、优化、推荐
- 交互类：问答、澄清、确认、引导、拒绝

约束条件：
- 长度限制：{字数/Token/条目数}
- 质量要求：{准确性/完整性/创造性}
- 禁止事项：{不做XX/不提及XX}
```

**案例**：

```markdown
## Task
对输入的产品需求文档（PRD）进行技术可行性预审，输出评估报告。

具体动作：
1. 【提取】识别PRD中的功能点列表
2. 【评估】对每个功能点给出技术难度（简单/中等/复杂）
3. 【风险】标记潜在的技术风险点（性能/安全/兼容性）
4. 【建议】对复杂功能给出简化替代方案

成功标准：
- 覆盖PRD中90%以上的功能点
- 每个评估有明确的判断依据
- 风险点必须有缓解建议

约束：
- 不评估UI/UX设计合理性
- 不涉及资源排期和人力估算
- 总输出不超过2000字
```

### 3.3 Context：上下文的分层管理

```markdown
## Context

### 固定上下文（每次相同）
- 产品定位：面向中小企业的轻量级CRM
- 技术栈：Python/Django + PostgreSQL + Vue3
- 用户画像：销售团队主管，日均使用2小时，非技术背景

### 动态上下文（每次输入）
- 当前功能模块：{module_name}
- 相关历史决策：{decision_history}
- 用户原始输入：{user_input}

### 隐式上下文（系统注入）
- 当前日期：{current_date}
- 用户ID：{user_id}（用于个性化）
- 会话历史：{last_3_turns}
```

**上下文注入技巧**：

| 技巧 | 适用场景 | 示例 |
|-----|---------|------|
| 摘要注入 | 长历史 | "前文摘要：用户已确认需求A，正在讨论方案B" |
| 关键词标签 | 快速定位 | "[标签:退款][标签:VIP用户]" |
| 结构化数据 | 精确计算 | "订单数据：{amount: 1500, currency: 'CNY', status: 'paid'}" |
| 状态机 | 多轮交互 | "当前状态：awaiting_confirmation, 上一步：price_quoted" |

### 3.4 Output Format：格式的确定性设计

**常见格式模板**：

```markdown
## Output Format

### 选项1：结构化数据（JSON/XML/YAML）
```json
{
  "field1": {
    "type": "string|number|boolean|array|object",
    "description": "字段说明",
    "constraints": "长度/范围/枚举值",
    "example": "示例值"
  }
}
```

### 选项2：Markdown文档
```markdown
# {标题}

## 章节1：{章节名}
- 要点1
- 要点2

## 章节2：{章节名}
| 表头1 | 表头2 |
|-------|-------|
| 内容  | 内容  |
```

### 选项3：纯文本（带标记）
[标签:开始]
{内容}
[标签:结束]

### 选项4：代码块（指定语言）
```python
# 函数说明：{description}
def function_name(param: type) -> return_type:
    """
    Docstring with constraints
    """
    # 实现逻辑
    pass
```
```

**格式约束的粒度**：

| 粒度 | 控制点 | 示例 |
|-----|--------|------|
| 宏观 | 整体结构 | "先结论后论据"、"总分总" |
| 中观 | 段落组织 | "每段不超过3句话"、"用 bullet 而非 paragraph" |
| 微观 | 句式用词 | "禁止用'非常'、'很'等程度副词"、"用主动语态" |

### 3.5 Example：少样本设计的艺术

**原则：质量 > 数量，多样性 > 重复**

```markdown
## Example

### 示例1：标准情况（覆盖主要流程）
输入：{典型输入}
输出：{规范输出}
说明：此示例展示标准处理流程

### 示例2：边界情况（展示容错能力）
输入：{异常/模糊/不完整输入}
输出：{合理处理/澄清请求/降级输出}
说明：此示例展示对边界情况的处理

### 示例3：负面示例（明确禁止行为）
输入：{诱导错误输出的输入}
错误输出：{不期望的输出}
修正输出：{正确的处理方式}
说明：此示例明确禁止某种输出模式
```

**少样本设计的陷阱**：

| 陷阱 | 表现 | 修正 |
|-----|------|------|
| 过度拟合 | 只学示例字面意思，不懂泛化 | 增加示例多样性，明确抽象规则 |
| 位置偏见 | 总是模仿最后一个示例的风格 | 随机打乱示例顺序，或明确说明"不模仿风格" |
| 长度偏见 | 输出长度向示例看齐 | 明确长度约束，提供不同长度的示例 |
| 格式偏见 | 只输出示例中的字段 | 在格式定义中明确"必填/可选/动态" |

## 四、工程化实践：模板管理系统

### 4.1 模板版本管理

```yaml
# prompt_template.yaml
template_id: user_feedback_analyzer
version: 2.3.1
last_updated: 2024-01-15
author: team-ai-platform

rtcoe:
  role:
    base: "资深用户体验分析师"
    expertise: ["SaaS产品", "B2B用户行为", "NPS方法论"]
    tone: "专业但易懂，数据驱动"
    
  task:
    action: analyze_feedback
    deliverables: [sentiment, issues, suggestions]
    constraints:
      max_issues: 5
      suggestion_per_issue: 2-3
      
  context:
    static:
      product_type: "B2B SaaS"
      user_segments: ["admin", "editor", "viewer"]
    dynamic:
      - feedback_text
      - user_tier
      - submission_channel
      
  output_format:
    type: json
    schema: |
      {
        "sentiment": {"type": "string", "enum": ["positive", "negative", "neutral"]},
        "issues": {
          "type": "array",
          "items": {
            "description": "string",
            "severity": {"enum": ["high", "medium", "low"]}
          }
        }
      }
      
  examples:
    - id: standard_case
      input: "..."
      output: "..."
    - id: edge_case
      input: "..."
      output: "..."

performance:
  baseline_metrics:
    sentiment_accuracy: 0.89
    issue_recall: 0.76
  ab_test_history:
    - version: 2.3.0
      change: "添加severity字段"
      impact: "+5% issue_recall"
```

### 4.2 动态模板渲染

```python
from jinja2 import Template

class PromptRenderer:
    """模板渲染器"""
    
    def __init__(self, template_yaml: str):
        self.template = yaml.safe_load(template_yaml)
        
    def render(self, dynamic_context: Dict) -> str:
        """渲染完整Prompt"""
        
        # 组装RTCOE
        rtcoe = self.template['rtcoe']
        
        prompt_parts = []
        
        # Role
        role_text = self._render_role(rtcoe['role'])
        prompt_parts.append(f"## Role\n{role_text}")
        
        # Task
        task_text = self._render_task(rtcoe['task'])
        prompt_parts.append(f"## Task\n{task_text}")
        
        # Context（合并静态+动态）
        context_text = self._render_context(
            rtcoe['context']['static'],
            rtcoe['context']['dynamic'],
            dynamic_context
        )
        prompt_parts.append(f"## Context\n{context_text}")
        
        # Output Format
        prompt_parts.append(f"## Output Format\n{rtcoe['output_format']['schema']}")
        
        # Examples
        examples_text = self._render_examples(rtcoe['examples'])
        prompt_parts.append(f"## Example\n{examples_text}")
        
        return "\n\n".join(prompt_parts)
    
    def _render_role(self, role_config: Dict) -> str:
        """渲染角色部分"""
        template_str = """
        你是一位{{ base }}，擅长{{ expertise | join('、') }}。
        你的分析风格是：{{ tone }}。
        """
        return Template(template_str).render(role_config)
    
    def _render_task(self, task_config: Dict) -> str:
        """渲染任务部分"""
        lines = [
            f"执行动作：{task_config['action']}",
            f"交付物：{', '.join(task_config['deliverables'])}",
        ]
        if 'constraints' in task_config:
            lines.append("约束条件：")
            for k, v in task_config['constraints'].items():
                lines.append(f"  - {k}: {v}")
        return "\n".join(lines)
    
    def _render_context(self,
                        static: Dict,
                        dynamic_fields: List[str],
                        dynamic_values: Dict) -> str:
        """渲染上下文"""
        lines = ["### 固定背景"]
        for k, v in static.items():
            lines.append(f"- {k}: {v}")
        
        lines.append("\n### 当前输入")
        for field in dynamic_fields:
            value = dynamic_values.get(field, "[未提供]")
            lines.append(f"- {field}: {value}")
        
        return "\n".join(lines)
    
    def _render_examples(self, examples: List[Dict]) -> str:
        """渲染示例"""
        lines = []
        for ex in examples:
            lines.append(f"### {ex['id']}")
            lines.append(f"输入：{ex['input']}")
            lines.append(f"输出：{ex['output']}")
        return "\n\n".join(lines)
```

### 4.3 A/B测试框架

```python
class PromptABTest:
    """Prompt版本A/B测试"""
    
    def __init__(self, experiment_id: str):
        self.experiment_id = experiment_id
        self.variants = {}
        
    def register_variant(self,
                         name: str,
                         template: PromptRenderer,
                         traffic: float):
        """注册变体"""
        self.variants[name] = {
            'template': template,
            'traffic': traffic,
            'metrics': {
                'requests': 0,
                'latency': [],
                'quality_score': [],
                'error_rate': []
            }
        }
    
    def route(self, user_id: str) -> str:
        """按流量分配变体"""
        # 哈希分桶
        bucket = hash(f"{self.experiment_id}:{user_id}") % 100
        
        cumulative = 0
        for name, config in self.variants.items():
            cumulative += config['traffic'] * 100
            if bucket < cumulative:
                return name
        
        return list(self.variants.keys())[0]
    
    def evaluate(self,
                 variant_name: str,
                 prompt: str,
                 output: str,
                 ground_truth: str = None):
        """评估变体效果"""
        metrics = self._compute_metrics(output, ground_truth)
        
        v = self.variants[variant_name]
        v['metrics']['requests'] += 1
        v['metrics']['quality_score'].append(metrics['quality'])
        
        return metrics
    
    def _compute_metrics(self,
                         output: str,
                         ground_truth: str = None) -> Dict:
        """计算评估指标"""
        metrics = {}
        
        # 格式合规性（JSON是否可解析）
        metrics['format_valid'] = self._check_format(output)
        
        # 长度合规性
        metrics['length_ok'] = len(output) < 2000
        
        # 如果有标准答案，计算相似度
        if ground_truth:
            metrics['similarity'] = self._text_similarity(output, ground_truth)
        
        # 综合质量分
        metrics['quality'] = (
            metrics.get('format_valid', 0) * 0.3 +
            metrics.get('length_ok', 0) * 0.2 +
            metrics.get('similarity', 0.5) * 0.5
        )
        
        return metrics
```

## 五、实战案例：客服意图识别系统

### 5.1 业务场景

**需求**：识别用户咨询意图，路由到对应处理流程

**意图类别**：退款、技术支持、账户问题、投诉建议、合作咨询

### 5.2 结构化Prompt设计

```markdown
## Role
你是一位智能客服路由助手，擅长从用户咨询中快速识别核心意图。
你熟悉电商平台的业务逻辑，能区分表面诉求和真实意图。

决策风格：
- 优先理解用户情绪，再判断意图
- 对模糊表述主动澄清，不盲目分类
- 对紧急问题（投诉、安全）提高优先级

## Task
分析用户输入消息，完成意图识别和初步处理建议。

具体动作：
1. 【情绪识别】判断用户情绪状态（平静/着急/愤怒/满意）
2. 【意图分类】从预定义类别中选择最匹配的意图
3. 【置信度评估】给出分类置信度（0-1）
4. 【关键信息提取】提取订单号、商品名、时间等实体
5. 【路由建议】推荐处理流程（自动回复/人工介入/升级处理）
6. 【澄清需求】如信息不足，列出需澄清的问题（最多2个）

约束：
- 意图必须属于预定义列表，不得自创
- 置信度低于0.7时必须要求澄清
- 检测到辱骂或威胁时，标记为"紧急升级"

## Context

### 预定义意图类别
1. 退款退货：涉及退款、退货、换货、差价补偿
2. 技术支持：功能使用、系统故障、操作指导
3. 账户问题：登录、密码、信息修改、权限
4. 投诉建议：服务质量投诉、功能建议、体验反馈
5. 合作咨询：商务合作、入驻、API对接
6. 其他：不属于以上类别

### 当前会话背景
- 用户等级：{user_tier}（普通/VIP/企业）
- 历史订单数：{order_count}
- 近30天咨询次数：{recent_tickets}
- 上一条消息：{last_message}

### 用户输入
{user_input}

## Output Format
输出严格JSON，不要Markdown代码块标记：

{
  "emotion": "calm|urgent|angry|satisfied",
  "emotion_confidence": 0.0-1.0,
  "intent": "refund|tech_support|account|complaint|partnership|other",
  "intent_confidence": 0.0-1.0,
  "entities": {
    "order_id": "字符串或null",
    "product_name": "字符串或null",
    "amount": "数字或null",
    "time_reference": "字符串或null"
  },
  "routing": "auto|human|escalate",
  "clarification_needed": true|false,
  "clarification_questions": ["问题1", "问题2"] // 最多2个
}

## Example

### 示例1：明确退款意图
输入："我昨天买的iPhone 15有问题，屏幕有亮点，想退货，订单号是TB20240115001"

输出：
{
  "emotion": "calm",
  "emotion_confidence": 0.85,
  "intent": "refund",
  "intent_confidence": 0.95,
  "entities": {
    "order_id": "TB20240115001",
    "product_name": "iPhone 15",
    "amount": null,
    "time_reference": "昨天"
  },
  "routing": "auto",
  "clarification_needed": false,
  "clarification_questions": []
}

### 示例2：模糊意图需澄清
输入："你们这个系统怎么回事"

输出：
{
  "emotion": "urgent",
  "emotion_confidence": 0.72,
  "intent": "other",
  "intent_confidence": 0.45,
  "entities": {
    "order_id": null,
    "product_name": null,
    "amount": null,
    "time_reference": null
  },
  "routing": "human",
  "clarification_needed": true,
  "clarification_questions": [
    "您遇到了什么问题？例如：无法登录、功能报错、页面加载慢等",
    "这个问题影响您完成什么操作？"
  ]
}

### 示例3：表面咨询实际投诉
输入："我想了解一下你们的发票政策"

输出：
{
  "emotion": "calm",
  "emotion_confidence": 0.60,
  "intent": "complaint",
  "intent_confidence": 0.78,
  "entities": {
    "order_id": null,
    "product_name": null,
    "amount": null,
    "time_reference": null
  },
  "routing": "human",
  "clarification_needed": false,
  "clarification_questions": [],
  "note": "用户用词委婉但上下文暗示对之前发票处理不满，建议人工介入主动关怀"
}
```

### 5.3 迭代优化记录

| 版本 | 改动 | 效果 |
|-----|------|------|
| v1.0 | 基础RTCOE，无示例 | 意图准确率72%，格式错误率15% |
| v1.1 | 添加2个示例 | 准确率提升至81%，格式错误率5% |
| v1.2 | 增加"note"字段说明隐性意图 | 投诉识别率从45%提升至78% |
| v2.0 | 拆分emotion和intent置信度 | 澄清准确率提升，减少误路由 |
| v2.1 | 添加上下文（用户等级、历史） | VIP用户满意度+12% |

## 六、常见模式与反模式

### 6.1 推荐模式

| 模式 | 描述 | 适用场景 |
|-----|------|---------|
| **洋葱式** | 从核心到外围逐层展开 | 复杂分析任务 |
| **清单式** | 明确的检查项列表 | 审查、评估任务 |
| **对话式** | 模拟角色对话 | 创意生成、咨询 |
| **代码式** | 类函数定义（输入/处理/输出/异常） | 结构化数据处理 |

### 6.2 反模式警示

| 反模式 | 表现 | 后果 |
|-------|------|------|
| **Prompt膨胀** | 超过2000字，要素堆砌 | 重点淹没，LLM忽略后半部分 |
| **矛盾约束** | "要详细"且"要简短" | 输出质量不稳定 |
| **过度指定** | 每个字都要控制 | 扼杀创造性，输出僵硬 |
| **示例失衡** | 3个示例都是正面案例 | 负面情况处理能力弱 |
| **动态缺失** | 静态Prompt处理动态场景 | 上下文脱节，答非所问 |

---

结构化Prompt不是"写得更长"，而是"设计得更清晰"。RTCOE框架将Prompt从"艺术"转化为"工程"，实现可维护、可测试、可迭代的AI应用开发。记住：**好的Prompt模板，是团队的知识资产，不是个人的语感秘籍**。
