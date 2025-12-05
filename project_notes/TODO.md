1. 改掉依赖qwen-agent框架的function call，变为基于`BaseTool`的verl式流程
2. 在不引入记忆操作的前提下，先实现一个多轮对话episode -> vLLM response -> reward -> verl GRPO的rollout
   1. 使用MemoryDiaglogueEpisodeRunner，但是把记忆操作留空
   2. 不在system prompt里注册memory
   3. reward = 最后几轮回答的EM/F1
3. 上述流程跑通之后再往里加记忆操作