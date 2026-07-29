```
flowchart TB
    User[用户 / 客户端]

    subgraph Backend["FastAPI 后端服务"]
        API[API接口层]
        Session[会话管理器]
        Orchestrator[Agent编排器]
        Intent[意图路由]
        State[状态管理]
        Dispatch[工具分发]
        Response[响应生成]
        RAGTool[RAG知识库检索]
        PlanTool[任务规划]
        Retrieval[检索链路]
        LLMPipeline[Prompt/LLM链路]
    end

    subgraph External["外部服务与数据层"]
        VectorDB[向量数据库 Qdrant]
        LLMService[LLM服务]
        KB[知识库/数据层]
    end

    User --> API
    API --> Session
    Session --> Orchestrator
    Orchestrator --> Intent
    Intent --> State
    State --> Dispatch
    Dispatch --> Response
    Orchestrator --> RAGTool
    Orchestrator --> PlanTool
    RAGTool --> Retrieval
    PlanTool --> LLMPipeline
    Retrieval --> VectorDB
    VectorDB --> KB
    LLMPipeline --> LLMService
```



```
graph TD
    A[Hello] --> B[World]
```

