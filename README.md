# E-Commerce AI Agent Service

一个面向业务编排场景的 Agent Harness Engineering 框架示例。项目基于 Spring Boot 和 Spring AI，将自然语言请求拆解为可验证、可执行、可扩展的工作流，而不是把所有判断都压进一次大模型调用里。

它默认落地在电商场景，但项目本质上并不是“电商问答机器人”，而是一个可复用的后端编排框架:

- 你可以替换各个 Agent 的 prompt，适配自己的业务语义
- 你可以新增或替换 `CapabilityHandler`，接入自己的后端接口或本地能力
- 你可以保留现有状态机、验证链路、记忆层和失败降级机制，快速搭出自己的 agent service

## Why This Project

很多 AI Agent Demo 的实现方式是:

- 一个 prompt
- 一个模型
- 一次输出
- 模型顺带决定计划、调用、判断和最终回复

这种方式启动快，但很快会遇到几个问题:

- 难以约束模型输出
- 失败原因不清晰
- 调试成本高
- 工具接入容易失控
- 多轮对话中的上下文污染严重
- 项目越来越难演进成稳定框架

这个项目采用另一种思路: 把大模型放进一个明确的 harness 里，让模型参与决策，但不让模型直接拥有系统控制权。

## What Is Harness Engineering Here

这个项目里，`harness engineering` 的核心不是“换更强的 prompt”，而是为模型包一层稳定的执行外壳。它主要体现在下面几件事上:

### 1. 把能力边界从 Prompt 中抽出来

模型不直接自由调用外部系统，只能在受控协议内工作。

- `planAgent` 只做意图级规划
- `executorAgent` 只把规划翻译成受支持的执行动作
- 真正调用接口的是 Java 里的 `CapabilityHandler`
- `verifyAgent` 专门检查计划和执行结果，而不是兼做回答

也就是说，模型负责“决定做什么”，Java 负责“真正怎么做”。

### 2. 用结构化协议替代自由文本串联

项目把关键阶段都建成了显式对象，而不是让模型前后靠自然语言硬拼:

- `PlanDraft`
- `ExecutionPlan`
- `ActionSpec`
- `ExecutionResult`
- `VerificationResult`
- `WorkflowState`

这让系统具备几个关键好处:

- 每个阶段都可单独调试
- 每个阶段都可校验
- 失败可以被归因，而不是只看到“模型答错了”
- 后续可以替换模型、替换 handler、替换下游服务，而不必重写整条链路

### 3. 用状态机管理 Agent 生命周期

整个请求不是“来一条消息就回一句话”，而是经过显式阶段推进:

- `INTAKE`
- `EXPLORE`
- `PLAN`
- `PLAN_VERIFY`
- `EXECUTE`
- `EXECUTION_VERIFY`
- `RESPOND`
- `FAILSAFE`

这相当于给 Agent 系统加了一层 workflow runtime。模型是工作流中的参与者，不是工作流本身。

### 4. 用验证环节替代盲信模型

项目把验证拆成两个阶段:

- 规划验证: 这个计划是否完整、是否合理、是否缺信息
- 执行验证: 执行结果是否足以回答用户、是否存在失败、是否需要降级

这是一种很典型的 harness 设计思想: 不把模型输出视为事实，而把它视为待审查的中间产物。

### 5. 用记忆分层避免上下文混乱

项目没有把“全部历史”粗暴塞进一个 prompt，而是区分了不同类型的记忆:

- Recent conversation window
- Workflow memory
- User profile
- User long-term memory
- System operation memory

不同 Agent 只拿自己需要的上下文，减少 prompt 污染和跨阶段噪音。

## Multi-Agent Orchestration Design

这个项目是一个多角色协作的 Agent Orchestration Pipeline。

更准确地说，它当前实现的是“角色分离式多智能体编排”，而不是“并行 swarm / worker 群协作”。也就是说:

- 有多个职责明确的 agent
- 它们按阶段串行协同
- 每个 agent 的输入上下文不同
- 最终由 orchestrator 汇总结果

当前实际存在的 Agent 角色有 4 个:

### 1. `orchestratorAgent`

职责:

- 处理极简单请求的直接回复
- 在执行验证通过后，组织最终用户回复
- 不直接访问下游服务

### 2. `planAgent`

职责:

- 把自然语言请求转成意图级 `PlanDraft`
- 输出目标、缺失信息、假设、计划步骤、是否需要执行
- 不负责选 handler，不负责编排具体参数

### 3. `executorAgent`

职责:

- 把意图级计划转成可执行的 `ExecutionPlan`
- 只能选择系统已支持的 capability
- 只能使用已定义参数
- 缺必要信息时返回 `blocked=true`

### 4. `verifyAgent`

职责:

- 验证规划是否可执行
- 验证执行结果是否足以回答用户
- 输出 `VerificationResult`
- 不负责生成最终回复

## High-Level Flow

```text
User Request
  -> WorkflowController
  -> WorkflowOrchestratorService
  -> planAgent
  -> verifyAgent (plan verification)
  -> executorAgent
  -> CapabilityDispatcher
  -> CapabilityHandler
  -> verifyAgent (execution verification)
  -> orchestratorAgent
  -> ChatResponse
```

## Core Architecture

### 1. API Layer

入口接口:

- `POST /api/workflows/chat`

对应文件:

- `src/main/java/com/JJLin/aiagent/controller/WorkflowController.java`

### 2. Orchestration Layer

核心类:

- `src/main/java/com/JJLin/aiagent/service/WorkflowOrchestratorService.java`

负责:

- 初始化工作流状态
- 决定是否走直接回复
- 驱动规划、验证、执行、再验证、回复
- 维护阶段历史、失败原因、修复项、执行结果
- 在失败时进入 `FAILSAFE`

### 3. Agent Configuration Layer

核心类:

- `src/main/java/com/JJLin/aiagent/config/AgentClientConfig.java`

负责:

- 创建各类 `ChatClient`
- 为不同角色配置独立 system prompt
- 初始化下游 HTTP client、本地文件 client、浏览器 client

### 4. Capability Layer

核心类:

- `src/main/java/com/JJLin/aiagent/service/CapabilityDispatcher.java`
- `src/main/java/com/JJLin/aiagent/capability/CapabilityRegistry.java`
- `src/main/java/com/JJLin/aiagent/capability/CapabilityHandler.java`

负责:

- 注册所有可执行能力
- 描述能力 schema
- 校验 `actionType` 和参数
- 把动作路由到对应 handler

### 5. Memory Layer

核心类:

- `src/main/java/com/JJLin/aiagent/service/WorkflowMemoryService.java`
- `src/main/java/com/JJLin/aiagent/service/UserProfileService.java`
- `src/main/java/com/JJLin/aiagent/service/UserLongTermMemoryService.java`
- `src/main/java/com/JJLin/aiagent/service/SystemOperationMemoryService.java`
- `src/main/java/com/JJLin/aiagent/service/MemoryContextAssembler.java`

负责:

- 保存工作流状态摘要
- 抽取用户画像
- 维护向量化长期记忆
- 记录系统操作失败经验
- 为不同 Agent 组装差异化上下文

## Memory Strategy

项目当前采用分层记忆设计，而不是单一会话历史:

### 1. Conversation Memory

- 实现: `MessageWindowChatMemory`
- 当前窗口大小: 20 条消息
- 用途: 保留最近对话上下文

### 2. Workflow Memory

- 记录当前 `chatId` 的计划摘要、执行摘要、开放问题、已解析事实、失败标签
- 用于支持跨轮次状态延续

### 3. User Profile

- 从用户消息中异步抽取结构化画像
- 持久化到 MySQL

### 4. User Long-Term Memory

- 从用户消息中异步抽取长期偏好和背景信息
- 当前基于 `SimpleVectorStore`
- 通过向量召回提供相关记忆

### 5. System Operation Memory

- 记录特定操作的失败经验
- 在后续同类任务中提供给验证阶段

## Capability Model

项目的一个关键设计点是: LLM 不直接调用任意工具，而是先产出一个受控动作，再由 Java 执行。

动作的核心协议包括:

- `actionType`
- `targetService`
- `params`
- `expectedOutput`

当前已实现的能力包括两类:

### E-commerce Capabilities

- `SEARCH_PRODUCTS`
- `RECOMMEND_PRODUCTS`
- `CHECK_ORDER`
- `CREATE_ORDER`

### Local / Utility Capabilities

- `LIST_FILES`
- `READ_FILE`
- `WRITE_FILE`
- `OPEN_BROWSER_URL`

这说明当前仓库虽然以电商为默认业务域，但底层框架已经在向通用 agent runtime 演进。

## Why This Is Reusable

你的描述“这是一个框架，其他人只需要改 prompt，并根据后端功能或接口实现更多 handler”基本符合项目现状，但更准确的表达建议写成:

> This project is a reusable agent orchestration framework with an e-commerce default implementation. Users can adapt it to new domains by changing agent prompts, adding or replacing capability handlers, and connecting their own backend services.

原因是:

- 它不只是 prompt framework，还包含状态机、验证链路、能力约束、记忆分层
- 它不只是 handler framework，还包含多 agent 分工和执行协议
- 它当前默认场景是电商，但并不被电商场景本身锁死

如果只写“改 prompt + 加 handler”会低估这个项目真正的工程价值。

## Repository Structure

```text
src/main/java/com/JJLin/aiagent
├── capability   # 能力定义、能力注册、能力执行
├── client       # 下游 HTTP / 本地文件 / 浏览器客户端
├── config       # Spring、Agent、Memory、VectorStore 配置
├── controller   # REST API 入口
├── entites      # 协议对象、状态对象、实体对象
├── enums        # 工作流与验证枚举
├── extractor    # 用户画像与长期记忆抽取
├── mapper       # MyBatis-Plus 持久化映射
├── memory       # 工作流记忆存储实现
├── service      # 编排、记忆、上下文组装等核心服务
└── vector       # Embedding 模型实现
```

## Tech Stack

- Java 17
- Spring Boot 3.5.0
- Spring AI 1.0.0
- DeepSeek Chat Model
- Spring Web
- Spring WebFlux
- Spring JDBC
- MyBatis-Plus
- MySQL
- Spring AI Vector Store
- Maven
- JUnit 5

## API

### Endpoint

`POST /api/workflows/chat`

### Example Request

```json
{
  "userId": "u-1001",
  "chatId": "chat-001",
  "content": "帮我查一下订单 123456 的状态"
}
```

### Example Response Shape

```json
{
  "stage": "RESPOND",
  "reply": "订单 123456 当前状态为已发货。",
  "planDraft": {},
  "executionPlan": {},
  "verificationResult": {},
  "executionResults": [],
  "workflowState": {},
  "memorySummary": "latestPlan=... | latestExecution=..."
}
```

## Local Run

### Requirements

- JDK 17
- Maven 3.9+
- MySQL 8+

### Database

默认数据库名:

- `work_memory`

项目会根据 `src/main/resources/schema.sql` 初始化表结构。

### Configuration

主配置文件:

- `src/main/resources/application.yml`

当前配置包括:

- 服务端口
- DeepSeek 模型接入
- MySQL 连接
- 下游能力服务地址
- Workspace root
- 浏览器协议白名单

建议在发布到 GitHub 前处理敏感信息，不要把真实密钥和账号密码直接保留在仓库里。更合理的方式是改为环境变量，例如:

- `MYSQL_HOST`
- `MYSQL_PORT`
- `MYSQL_USERNAME`
- `MYSQL_PASSWORD`
- `DEEPSEEK_API_KEY`
- `AI_CAPABILITY_BASE_URL`

### Start

```bash
mvn spring-boot:run
```

或:

```bash
mvn clean package
java -jar target/ecommerce-ai-agent-service-0.0.1-SNAPSHOT.jar
```

默认端口:

```text
8099
```

## How To Adapt This Framework To Your Own Domain

如果你想把这个项目改造成自己的 Agent Framework，最常见的做法是下面三步:

### 1. 改 Agent Prompt

调整位置:

- `src/main/java/com/JJLin/aiagent/config/AgentClientConfig.java`

你可以重写:

- orchestrator 的回复风格
- planner 的规划粒度
- executor 的动作选择规则
- verifier 的审查标准

### 2. 增加或替换 CapabilityHandler

扩展位置:

- `src/main/java/com/JJLin/aiagent/capability`

每个新能力只需要实现统一接口:

- `actionType()`
- `targetService()`
- `definition()`
- `execute(ActionSpec actionSpec)`

然后 Spring 会自动注册到 `CapabilityRegistry`。

### 3. 接入自己的后端服务

你可以把默认电商接口替换成:

- CRM
- ERP
- OA
- 工单系统
- 知识库
- 内部文件系统
- 浏览器自动化能力
- 任何 HTTP API

只要能力被包装成 `CapabilityHandler`，整个编排链路无需重写。

## Testing

当前测试覆盖重点不是 UI 或联调，而是工作流骨架本身:

- `WorkflowOrchestratorServiceTest`
  验证多阶段编排、失败降级、参数绑定、记忆注入、执行验证
- `WorkflowMemoryInfrastructureTest`
  验证会话窗口与 workflow memory 隔离
- `CapabilityContractTest`
  验证 capability schema 和参数约束
- `SystemOperationMemoryServiceTest`
  验证操作记忆写入与读取
- `UserMemoryExtractorServiceTest`
  验证长期记忆抽取逻辑

运行测试:

```bash
mvn test
```

## Current Strengths

- 多 Agent 分工明确
- 状态机驱动，链路清晰
- 计划、执行、验证职责分离
- tool / capability 调用受控
- 失败有标签，不是纯黑盒
- 记忆分层较完整
- 易于扩展新业务 handler
- 已经具备“框架化”而不是“单一 demo”特征

## Current Limitations

项目目前也有几个明确边界，README 中最好主动讲清楚:

- 当前多智能体是角色编排，不是并行 worker 编排
- 长期记忆基于 `SimpleVectorStore`，默认不是持久化向量数据库
- 下游能力仍以本地和电商接口为默认样例
- 真实生产级观测、审计、追踪能力还不完整
- prompt 与 schema 约束仍有继续强化空间
- 当前是单体后端服务，不是分布式 agent runtime

## Recommended Roadmap

如果继续往“可复用开源框架”方向推进，建议优先补这些能力:

- 将向量存储替换为持久化方案
- 引入更严格的 action schema 和 output schema
- 增加 capability 级别的审计日志与 tracing
- 增加真正的 clarify loop，而不只是 failsafe 提示
- 引入并行 worker / subtask 编排
- 为 handler 提供更标准的插件化加载方式
- 补齐 integration tests 和示例业务接入文档

## Suitable GitHub Description

可用作仓库简介:

```text
A reusable multi-agent orchestration framework built with Spring Boot and Spring AI, featuring harness engineering, structured planning, controlled capability execution, verification, and layered memory.
```

## License

发布开源仓库前建议补充 License。

如果希望简单直接，`MIT` 是比较合适的默认选择。
