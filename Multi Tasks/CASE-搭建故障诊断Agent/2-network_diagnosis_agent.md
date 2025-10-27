# 2-network_diagnosis_agent.py 代码逻辑说明

本文件实现了一个基于 LangChain 框架的网络故障诊断 Agent，能够模拟常见的网络诊断工具（如 Ping、DNS 查询、接口检查、日志分析），并通过大语言模型（通义千问）自动分析和诊断用户描述的网络问题。

---

## 1. 环境与依赖
- 依赖 `langchain`、`langchain_community.llms`、`dashscope` 等库。
- 通过环境变量或硬编码设置通义千问 API Key。

---

## 2. 自定义网络诊断工具类

### 2.1 PingTool
- 名称：网络连通性检查 (Ping)
- 功能：模拟 ping 命令，检查本机到目标主机的连通性。
- 逻辑：根据目标字符串内容，返回不同的模拟结果（如超时、成功、延迟等）。

### 2.2 DNSTool
- 名称：DNS解析查询
- 功能：模拟 DNS 查询，将主机名解析为 IP 地址。
- 逻辑：针对特定主机名返回固定 IP，未知主机名返回失败或通用 IP。

### 2.3 InterfaceCheckTool
- 名称：本地网络接口检查
- 功能：模拟检查本地网络接口（如以太网、Wi-Fi）的状态。
- 逻辑：指定接口名为 eth1 时返回关闭，否则返回启用及相关网络信息。

### 2.4 LogAnalysisTool
- 名称：网络日志分析
- 功能：模拟分析系统或应用日志，查找与网络相关的错误。
- 逻辑：根据关键词返回预设的日志条目摘要。

---

## 3. Agent 及工具链创建

### create_network_diagnosis_chain()
- 初始化上述四个工具类，并将其包装为 LangChain 的 Tool 对象。
- 使用通义千问大模型（Tongyi）作为 LLM。
- 通过 `initialize_agent` 创建 Zero-Shot ReAct Agent，支持多轮对话和记忆。
- 返回可直接调用的 agent 对象。

---

## 4. 网络诊断主流程

### diagnose_network_issue(issue_description)
- 输入：用户描述的网络问题（字符串）。
- 步骤：
  1. 调用 `create_network_diagnosis_chain()` 获取 agent。
  2. 通过 agent.invoke() 让大模型自动调用合适的工具链分析问题。
  3. 返回诊断结果（output 字段）。
- 异常处理：捕获并打印详细错误信息。

---

## 5. 主程序入口
- 提供多个示例任务（如无法访问网站、内部服务连接失败等）。
- 依次调用 `diagnose_network_issue`，输出诊断结果。
- 便于测试和演示 agent 的诊断能力。

---

## 6. 代码设计亮点
- 工具链高度模块化，便于扩展更多网络诊断工具。
- 支持多轮对话和上下文记忆，适合复杂诊断场景。
- 诊断流程自动化，用户只需描述问题即可获得智能分析。

---

## 7. 注意事项
- 该实现为模拟环境，未实际执行系统命令，仅用于演示和开发测试。
- 生产环境请妥善保护 API 密钥。 