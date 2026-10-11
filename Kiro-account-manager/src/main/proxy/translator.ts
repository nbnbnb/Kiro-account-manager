// OpenAI/Claude 格式与 Kiro 格式转换器
import { v4 as uuidv4 } from 'uuid'
import type {
  OpenAIChatRequest,
  OpenAIMessage,
  OpenAIContentPart,
  OpenAITool,
  OpenAIChatResponse,
  OpenAIStreamChunk,
  OpenAIResponsesRequest,
  OpenAIResponsesResponse,
  OpenAIResponseContentPart,
  OpenAIResponseOutputItem,
  OpenAIResponseTool,
  ClaudeRequest,
  ClaudeMessage,
  ClaudeResponse,
  ClaudeStreamEvent,
  ClaudeContentBlock,
  KiroPayload,
  KiroHistoryMessage,
  KiroToolWrapper,
  KiroToolResult,
  KiroImage,
  KiroDocument,
  KiroToolUse,
  KiroUserInputMessage,
  KiroCachePoint,
  KiroReasoningContent,
  KiroUsage
} from './types'
import { buildKiroPayload, mapModelId } from './kiroApi'
import { ToolNameRegistry } from './toolNameRegistry'

const KIRO_CACHE_POINT: KiroCachePoint = { type: 'default' }

/** 模型 thinking 能力元数据（由 proxyServer 从模型缓存中查询后传入） */
export interface ThinkingConfig {
  schemaPath: 'output_config' | 'reasoning'
  efforts: string[]
  defaultEffort?: string
}

/**
 * 将客户端 effort / thinking 参数 + 模型 schema path 映射为 Kiro additionalModelRequestFields。
 * 官方 Kiro IDE 的 kr() 函数逻辑：
 *   output_config → { thinking: { type: 'adaptive', display: 'summarized' }, output_config: { effort } }
 *   reasoning     → { reasoning: { effort } }
 */
function buildThinkingFields(
  thinkingConfig: ThinkingConfig | undefined,
  clientThinking?: { type: string; budget_tokens?: number; display?: string },
  clientReasoningEffort?: string
): Record<string, unknown> | undefined {
  // 客户端明确关闭 thinking
  if (clientThinking?.type === 'disabled') return undefined
  // GPT-5.6 的 reasoning.effort=none 表示不启用推理，不能降级成 Kiro
  // 模型支持列表里的最高 effort。
  if (clientReasoningEffort?.toLowerCase() === 'none') return undefined

  // 没有模型元数据时，回退到旧逻辑：仅传 { thinking: { type: 'adaptive' } }
  if (!thinkingConfig) {
    if (clientThinking && clientThinking.type !== 'disabled') {
      return { thinking: { type: 'adaptive' } }
    }
    if (clientReasoningEffort) {
      return { thinking: { type: 'adaptive' } }
    }
    return undefined
  }

  // 客户端没请求 thinking 也没请求 reasoning_effort → 不启用
  const wantsThinking = !!(clientThinking && clientThinking.type !== 'disabled') || !!clientReasoningEffort
  if (!wantsThinking) return undefined

  // 映射 effort level（直接使用客户端值，不做强制转换）
  const mapEffort = (input: string): string => input.toLowerCase()

  let effort: string
  if (clientReasoningEffort) {
    effort = mapEffort(clientReasoningEffort)
  } else if (clientThinking?.type === 'enabled' && clientThinking.budget_tokens) {
    // budget_tokens 粗略映射到 effort level
    const b = clientThinking.budget_tokens
    if (b <= 4000) effort = 'low'
    else if (b <= 16000) effort = 'medium'
    else if (b <= 64000) effort = 'high'
    else effort = 'xhigh'
  } else {
    effort = thinkingConfig.defaultEffort || 'high'
  }

  // 确保 effort 在可用范围内，否则取最接近的
  if (!thinkingConfig.efforts.includes(effort)) {
    effort = thinkingConfig.efforts[thinkingConfig.efforts.length - 1] || 'high'
  }

  switch (thinkingConfig.schemaPath) {
    case 'output_config':
      return {
        thinking: { type: 'adaptive', display: 'summarized' },
        output_config: { effort }
      }
    case 'reasoning':
      return { reasoning: { effort } }
    default:
      return { thinking: { type: 'adaptive' } }
  }
}

function toKiroCachePoint(cacheControl?: { type: string }): KiroCachePoint | undefined {
  if (!cacheControl) return undefined
  if (cacheControl.type !== 'ephemeral') {
    throw new Error(`Unsupported cache_control type: ${cacheControl.type}`)
  }
  return KIRO_CACHE_POINT
}

function mergeCachePoint(
  first?: KiroCachePoint,
  second?: KiroCachePoint
): KiroCachePoint | undefined {
  return first || second
}

export function responsesToOpenAIChat(request: OpenAIResponsesRequest): OpenAIChatRequest {
  if (!request || typeof request !== 'object') {
    throw new Error('Responses request body must be an object')
  }
  if (!request.model) {
    throw new Error('Responses request requires model')
  }
  if (request.input === undefined) {
    throw new Error('Responses request requires input')
  }

  const messages: OpenAIMessage[] = []
  const responseTools: OpenAIResponseTool[] = [
    ...(Array.isArray(request.tools) ? request.tools : []),
    ...(Array.isArray(request.additional_tools) ? request.additional_tools : [])
  ]
  if (request.instructions) {
    messages.push({ role: 'system', content: request.instructions })
  }
  if (typeof request.input === 'string') {
    messages.push({ role: 'user', content: request.input })
  } else {
    if (!Array.isArray(request.input)) {
      throw new Error('Responses input must be a string or an array')
    }
    for (const item of request.input) {
      const itemType = item.type as string | undefined
      if (itemType === 'additional_tools') {
        collectAdditionalResponseTools(item, responseTools)
      } else if (itemType === 'function_call_output') {
        if (!item.call_id) {
          throw new Error('function_call_output requires call_id')
        }
        if (item.output === undefined) {
          throw new Error('function_call_output requires output')
        }
        messages.push({
          role: 'tool',
          content: stringifyResponseValue(item.output),
          tool_call_id: item.call_id
        })
      } else if (itemType === 'function_call') {
        if (!item.call_id) {
          throw new Error('function_call requires call_id')
        }
        if (!item.name) {
          throw new Error('function_call requires name')
        }
        if (item.arguments === undefined) {
          throw new Error('function_call requires arguments')
        }
        messages.push({
          role: 'assistant',
          content: '',
          tool_calls: [{
            id: item.call_id,
            type: 'function',
            function: {
              name: item.name,
              arguments: stringifyResponseArguments(item.arguments)
            }
          }]
        })
      } else if (isResponseToolCallType(itemType)) {
        messages.push(responseToolCallToMessage(item, itemType))
      } else if (isResponseToolOutputType(itemType)) {
        const callId = item.call_id || item.id
        if (!callId) {
          console.warn(`[Responses] Skipping ${itemType} without call_id/id`)
          continue
        }
        messages.push({
          role: 'tool',
          content: stringifyResponseValue(item.output ?? item.result ?? item.content),
          tool_call_id: callId
        })
      } else if (isStandaloneResponseContentType(itemType)) {
        messages.push({
          role: 'user',
          content: convertResponseInputContent([item as OpenAIResponseContentPart])
        })
      } else if (isResponsesStateOnlyItem(itemType)) {
        // reasoning / item_reference / compaction 等依赖 Responses 服务端状态，
        // Kiro 没有对等输入项。跳过比伪造成 user 文本更安全。
        continue
      } else {
        if (itemType !== undefined && itemType !== 'message') {
          console.warn(`[Responses] Skipping unsupported input item type: ${itemType}`)
          continue
        }
        if (item.content === undefined) {
          throw new Error('message input item requires content')
        }
        const role = normalizeResponseMessageRole(item.role)
        if (!role) {
          console.warn(`[Responses] Skipping message with unsupported role: ${item.role}`)
          continue
        }
        messages.push({
          role,
          content: convertResponseInputContent(item.content),
          ...(role === 'tool' && (item.call_id || item.id)
            ? { tool_call_id: item.call_id || item.id }
            : {})
        })
      }
    }
  }

  const chatRequest: OpenAIChatRequest = {
    model: request.model,
    messages
  }
  if (request.temperature !== undefined) chatRequest.temperature = request.temperature
  if (request.top_p !== undefined) chatRequest.top_p = request.top_p
  if (request.max_output_tokens !== undefined) chatRequest.max_tokens = request.max_output_tokens
  if (request.stream !== undefined) chatRequest.stream = request.stream
  const normalizedTools = normalizeOpenAITools(responseTools)
  if (normalizedTools.length > 0) chatRequest.tools = normalizedTools
  const toolChoice = convertResponseToolChoice(request.tool_choice)
  if (toolChoice !== undefined) chatRequest.tool_choice = toolChoice
  if (request.previous_response_id !== undefined) chatRequest.conversation_id = request.previous_response_id
  const reasoningEffort = extractResponsesReasoningEffort(request.reasoning)
  if (reasoningEffort !== undefined) chatRequest.reasoning_effort = reasoningEffort
  if (request.metadata !== undefined) chatRequest.metadata = request.metadata
  if (request.kiro_context !== undefined) chatRequest.kiro_context = request.kiro_context
  return chatRequest
}

function normalizeResponseMessageRole(role: string | undefined): OpenAIMessage['role'] | undefined {
  if (!role || role === 'user') return 'user'
  if (role === 'developer' || role === 'system') return 'system'
  if (role === 'assistant' || role === 'tool') return role
  return undefined
}

function stringifyResponseValue(value: unknown): string {
  if (typeof value === 'string') return value
  if (value === undefined || value === null) return ''
  try {
    return JSON.stringify(value)
  } catch {
    return String(value)
  }
}

function stringifyResponseArguments(value: unknown): string {
  if (typeof value === 'string') return value
  return stringifyResponseValue(value === undefined ? {} : value)
}

function isResponseToolCallType(type: string | undefined): type is string {
  return !!type && [
    'custom_tool_call', 'web_search_call', 'file_search_call',
    'code_interpreter_call', 'computer_call', 'local_shell_call',
    'image_generation_call', 'mcp_call', 'program'
  ].includes(type)
}

function isResponseToolOutputType(type: string | undefined): type is string {
  return !!type && [
    'custom_tool_call_output', 'computer_call_output', 'local_shell_call_output',
    'mcp_approval_response', 'program_output'
  ].includes(type)
}

function isStandaloneResponseContentType(type: string | undefined): boolean {
  return !!type && ['input_text', 'output_text', 'input_image', 'image_url', 'image', 'input_file'].includes(type)
}

function isResponsesStateOnlyItem(type: string | undefined): boolean {
  return !!type && [
    'reasoning', 'item_reference', 'compaction', 'mcp_list_tools',
    'mcp_approval_request', 'agent_message', 'multi_agent_call',
    'multi_agent_call_output'
  ].includes(type)
}

function responseToolCallToMessage(
  item: import('./types').OpenAIResponseInputItem,
  itemType: string
): OpenAIMessage {
  const callId = item.call_id || item.id || `call_${uuidv4()}`
  const isLocalShellCall = itemType === 'local_shell_call'
  const name = isLocalShellCall ? 'exec_command' : (item.name || itemType)
  const args = isLocalShellCall
    ? { cmd: localShellCommand(item.action ?? item.arguments ?? item.input) }
    : itemType === 'custom_tool_call'
    ? { input: stringifyResponseValue(item.input) }
    : item.arguments ?? item.action ?? item.input ?? {}
  return {
    role: 'assistant',
    content: '',
    tool_calls: [{
      id: callId,
      type: 'function',
      function: { name, arguments: stringifyResponseArguments(args) }
    }]
  }
}

function collectAdditionalResponseTools(
  item: import('./types').OpenAIResponseInputItem,
  target: OpenAIResponseTool[]
): void {
  for (const candidate of [item.tools, item.additional_tools]) {
    if (Array.isArray(candidate)) target.push(...candidate)
  }
}

export function normalizeOpenAITools(tools: OpenAIResponseTool[]): OpenAITool[] {
  const byName = new Map<string, OpenAITool>()
  for (const tool of tools) {
    if (!tool || typeof tool !== 'object') continue
    const nested = tool.function && typeof tool.function === 'object' ? tool.function : undefined
    const rawType = tool.type
    // Responses clients may advertise the built-in shell capability as
    // `local_shell`/`shell` without a function name. Kiro exposes tools as
    // function specifications, so provide a stable internal adapter instead
    // of silently dropping the workspace tool.
    const isShellTool = rawType === 'local_shell' || rawType === 'shell'
    const name = nested?.name || tool.name || (isShellTool ? 'exec_command' : undefined)
    const type = tool.type || (name ? 'function' : undefined)
    const responseToolType: OpenAITool['response_tool_type'] = tool.response_tool_type || (isShellTool
      ? 'local_shell'
      : type === 'custom' ? 'custom' : type === 'function' ? 'function' : undefined)
    if (!name || (!isShellTool && type !== 'function' && type !== 'custom')) {
      console.warn(`[OpenAI] Skipping unsupported tool type: ${type || 'unknown'}`)
      continue
    }
    const parameters = isShellTool
      ? {
          type: 'object',
          properties: {
            cmd: { type: 'string', description: 'Command to run in the current workspace.' }
          },
          required: ['cmd'],
          additionalProperties: false
        }
      : type === 'custom'
      ? {
          type: 'object',
          properties: {
            input: {
              type: 'string',
              description: 'Free-form input for this custom tool.'
            }
          },
          required: ['input'],
          additionalProperties: false
        }
      : nested?.parameters ?? tool.parameters ?? tool.inputSchema ?? {
          type: 'object',
          properties: {}
        }
    byName.set(name, {
      type: 'function',
      function: {
        name,
        description: nested?.description || tool.description || (isShellTool ? 'Run a command in the current workspace.' : `Tool: ${name}`),
        parameters
      },
      response_tool_type: responseToolType,
      ...(tool.cache_control ? { cache_control: tool.cache_control } : {})
    })
  }
  return [...byName.values()]
}

function extractResponsesReasoningEffort(reasoning: unknown): string | undefined {
  if (typeof reasoning === 'string') return reasoning
  if (!reasoning || typeof reasoning !== 'object') return undefined
  const effort = (reasoning as Record<string, unknown>).effort
  return typeof effort === 'string' ? effort : undefined
}

function convertResponseInputContent(content: unknown): OpenAIMessage['content'] {
  if (typeof content === 'string') return content
  if (content === undefined) return ''
  if (!Array.isArray(content)) {
    throw new Error('message content must be a string or an array')
  }
  return content.flatMap<OpenAIContentPart>(partValue => {
    if (!partValue || typeof partValue !== 'object') return []
    const part = partValue as OpenAIResponseContentPart
    const partType = part.type as string
    if (partType === 'input_image' || partType === 'image_url' || partType === 'image') {
      const imageUrl = typeof part.image_url === 'string'
        ? part.image_url
        : part.image_url?.url || part.url
      if (!imageUrl) {
        throw new Error('input_image requires image_url')
      }
      return [{ type: 'image_url' as const, image_url: { url: imageUrl } }]
    }
    if (partType === 'input_file') {
      if (!part.file_data) {
        console.warn('[Responses] Skipping input_file without inline file_data')
        return []
      }
      return [{
        type: 'file',
        file: {
          file_data: part.file_data,
          ...(part.filename !== undefined ? { filename: part.filename } : {})
        }
      }]
    }
    if (['input_text', 'output_text', 'text', 'refusal'].includes(partType)) {
      if (part.text === undefined) {
        throw new Error(`${partType} requires text`)
      }
      return [{ type: 'text' as const, text: part.text }]
    }
    // reasoning/summary/encrypted 等内容不应伪造成用户文本；未知 part 也宽容跳过。
    console.warn(`[Responses] Skipping unsupported content part type: ${partType || 'unknown'}`)
    return []
  })
}

function convertResponseToolChoice(toolChoice: OpenAIResponsesRequest['tool_choice']): OpenAIChatRequest['tool_choice'] {
  if (!toolChoice || typeof toolChoice === 'string') return toolChoice
  if (toolChoice.type === 'none' || toolChoice.type === 'auto') return toolChoice.type
  if ((toolChoice.type === 'function' || toolChoice.type === 'custom') && toolChoice.name) {
    return { type: 'function', function: { name: toolChoice.name } }
  }
  if (toolChoice.function?.name) return { type: 'function', function: { name: toolChoice.function.name } }
  throw new Error('Unsupported responses tool_choice')
}

export function openAIChatToResponsesResponse(
  response: OpenAIChatResponse,
  previousResponseId?: string,
  tools?: OpenAIResponseTool[]
): OpenAIResponsesResponse {
  const normalizedTools = normalizeOpenAITools(tools || [])
  const customToolNames = new Set(
    normalizedTools
      ?.filter(tool => tool.response_tool_type === 'custom')
      .map(tool => tool.function.name) || []
  )
  const output: OpenAIResponseOutputItem[] = response.choices.flatMap<OpenAIResponseOutputItem>(choice => {
    if (choice.message.tool_calls?.length) {
      return choice.message.tool_calls.map(toolCall => {
        const localShellTool = normalizedTools.some(tool =>
          tool.response_tool_type === 'local_shell' && tool.function.name === toolCall.function.name
        )
        if (localShellTool) {
          return {
            type: 'local_shell_call' as const,
            id: `shell_${uuidv4()}`,
            status: 'completed' as const,
            call_id: toolCall.id,
            action: localShellAction(toolCall.function.arguments)
          }
        }
        if (customToolNames.has(toolCall.function.name)) {
          return {
            type: 'custom_tool_call' as const,
            id: `ctc_${uuidv4()}`,
            status: 'completed' as const,
            call_id: toolCall.id,
            name: toolCall.function.name,
            input: extractCustomToolInput(toolCall.function.arguments)
          }
        }
        return {
          type: 'function_call' as const,
          id: `fc_${uuidv4()}`,
          status: 'completed' as const,
          call_id: toolCall.id,
          name: toolCall.function.name,
          arguments: toolCall.function.arguments
        }
      })
    }
    return [{
      type: 'message' as const,
      id: `msg_${uuidv4()}`,
      status: 'completed' as const,
      role: 'assistant' as const,
      content: [{
        type: 'output_text' as const,
        text: choice.message.content || '',
        annotations: []
      }]
    }]
  })

  const outputText = output
    .filter((item): item is Extract<OpenAIResponseOutputItem, { type: 'message' }> => item.type === 'message')
    .flatMap(item => item.content.map(part => part.text))
    .join('')

  const usage: OpenAIResponsesResponse['usage'] = {
    input_tokens: response.usage.prompt_tokens,
    output_tokens: response.usage.completion_tokens,
    total_tokens: response.usage.total_tokens
  }
  const cachedTokens = response.usage.prompt_tokens_details?.cached_tokens
  if (cachedTokens !== undefined) {
    usage.input_tokens_details = { cached_tokens: cachedTokens }
  }
  const reasoningTokens = response.usage.completion_tokens_details?.reasoning_tokens
  if (reasoningTokens !== undefined) {
    usage.output_tokens_details = { reasoning_tokens: reasoningTokens }
  }

  const responsesResponse: OpenAIResponsesResponse = {
    id: `resp_${uuidv4()}`,
    object: 'response',
    created_at: response.created,
    model: response.model,
    status: 'completed',
    output,
    output_text: outputText,
    usage
  }
  if (previousResponseId !== undefined) {
    responsesResponse.previous_response_id = previousResponseId
  }
  return responsesResponse
}

function localShellCommand(value: unknown): string {
  if (typeof value === 'string') return value
  if (Array.isArray(value)) return value.map(part => String(part)).join(' ')
  if (value && typeof value === 'object') {
    const obj = value as Record<string, unknown>
    if (typeof obj.cmd === 'string') return obj.cmd
    if (typeof obj.command === 'string') return obj.command
    if (Array.isArray(obj.command)) return obj.command.map(part => String(part)).join(' ')
  }
  return stringifyResponseValue(value ?? '')
}

function localShellAction(argumentsJson: string): Record<string, unknown> {
  let parsed: unknown
  try {
    parsed = JSON.parse(argumentsJson)
  } catch {
    parsed = argumentsJson
  }
  if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
    const obj = parsed as Record<string, unknown>
    if (obj.action && typeof obj.action === 'object' && !Array.isArray(obj.action)) return obj.action as Record<string, unknown>
    if (obj.type === 'exec' && (Array.isArray(obj.command) || typeof obj.command === 'string')) {
      return { ...obj, command: Array.isArray(obj.command) ? obj.command.map(part => String(part)) : [obj.command] }
    }
    if (typeof obj.cmd === 'string') return { type: 'exec', command: ['bash', '-lc', obj.cmd] }
    if (typeof obj.command === 'string') return { type: 'exec', command: ['bash', '-lc', obj.command] }
  }
  return { type: 'exec', command: ['bash', '-lc', String(parsed ?? '')] }
}

function extractCustomToolInput(argumentsJson: string): string {
  try {
    const parsed = JSON.parse(argumentsJson) as unknown
    if (parsed && typeof parsed === 'object' && typeof (parsed as { input?: unknown }).input === 'string') {
      return (parsed as { input: string }).input
    }
    if (typeof parsed === 'string') return parsed
  } catch {
    // Kiro may occasionally return the custom tool input as raw text.
  }
  return argumentsJson
}

// ============ OpenAI -> Kiro 转换 ============

export function openaiToKiro(
  request: OpenAIChatRequest,
  profileArn?: string,
  toolNameRegistry: ToolNameRegistry = new ToolNameRegistry(),
  thinkingConfig?: ThinkingConfig
): KiroPayload {
  const modelId = mapModelId(request.model)
  const origin = 'AI_EDITOR'

  // 提取系统提示
  let systemPrompt = ''
  let systemCachePoint: KiroCachePoint | undefined
  const nonSystemMessages: OpenAIMessage[] = []

  for (const msg of request.messages) {
    if (msg.role === 'system') {
      systemCachePoint = mergeCachePoint(systemCachePoint, toKiroCachePoint(msg.cache_control))
      if (typeof msg.content === 'string') {
        systemPrompt += (systemPrompt ? '\n' : '') + msg.content
      } else if (Array.isArray(msg.content)) {
        for (const part of msg.content) {
          systemCachePoint = mergeCachePoint(systemCachePoint, toKiroCachePoint(part.cache_control))
          if (part.type === 'text' && part.text) {
            systemPrompt += (systemPrompt ? '\n' : '') + part.text
          }
        }
      }
    } else {
      nonSystemMessages.push(msg)
    }
  }

  // 注入时间戳
  const timestamp = new Date().toISOString()
  systemPrompt = `[Context: Current time is ${timestamp}]\n\n${systemPrompt}`

  // 注入执行导向指令（防止 AI 在探索过程中丢失目标）
  const executionDirective = `
<execution_discipline>
当用户要求执行特定任务时，你必须遵循以下纪律：
1. **目标锁定**：在整个会话中始终牢记用户的原始目标，不要在代码探索过程中迷失方向
2. **行动优先**：优先执行任务而非仅分析或总结，除非用户明确只要求分析
3. **工具优先**：当请求提供了终端、文件、搜索或补丁工具时，涉及仓库/文件的事实必须先调用工具验证；禁止声称无法访问本地工作区，也禁止把应由工具执行的命令仅作为代码块交给用户运行
4. **工作区一致**：所有命令都在客户端当前工作区执行。Git 命令失败时先检查当前目录和 Git 安全目录配置，再采取修复措施，不要直接把命令转嫁给用户
5. **计划执行**：为任务创建明确的步骤计划，逐步执行并标记完成状态
6. **禁止确认性收尾**：在任务未完成前，禁止输出"需要我继续吗？"、"需要深入分析吗？"等确认性问题
7. **持续推进**：如果发现部分任务已完成，立即继续执行剩余未完成的任务
8. **完整交付**：直到所有任务步骤都执行完毕才算完成
</execution_discipline>
`
  systemPrompt = systemPrompt + '\n\n' + executionDirective

  // 构建历史消息（参考 Proxycast 实现）
  const history: KiroHistoryMessage[] = []
  const toolResults: KiroToolResult[] = []
  let currentContent = ''
  let currentCachePoint: KiroCachePoint | undefined
  const images: KiroImage[] = []
  const documents: KiroDocument[] = []
  for (let i = 0; i < nonSystemMessages.length; i++) {
    const msg = nonSystemMessages[i]
    const isLast = i === nonSystemMessages.length - 1

    if (msg.role === 'user') {
      const { content: userContent, images: userImages, documents: userDocuments, cachePoint } = extractOpenAIContent(msg)
      
      const mergedContent = userContent || 'Continue'
      const messageCachePoint = cachePoint
      
      if (isLast) {
        currentContent = mergedContent
        currentCachePoint = messageCachePoint
        images.push(...userImages)
        documents.push(...userDocuments)
      } else {
        history.push({
          userInputMessage: {
            content: mergedContent,
            modelId,
            origin,
            images: userImages.length > 0 ? userImages : undefined,
            documents: userDocuments.length > 0 ? userDocuments : undefined,
            ...(messageCachePoint ? { cachePoint: messageCachePoint } : {})
          }
        })
      }
    } else if (msg.role === 'assistant') {
      // Kiro API 要求 content 非空
      // 注意: 故意不读取 msg.reasoning_content (history 中不传给 Kiro)
      // Kiro 后端 schema 仅在响应输出中支持 assistantResponseMessage.reasoningContent，
      // 在请求 history 中传入此字段会触发 400 "Improperly formed request"
      let assistantContent = typeof msg.content === 'string' ? msg.content : ''
      if (!assistantContent.trim() && msg.tool_calls && msg.tool_calls.length > 0) {
        assistantContent = ' '
      } else if (!assistantContent.trim()) {
        assistantContent = 'I understand.'
      }
      const toolUses: KiroToolUse[] = []

      if (msg.tool_calls) {
        for (const tc of msg.tool_calls) {
          if (tc.type === 'function') {
            let input = {}
            try {
              input = JSON.parse(tc.function.arguments)
            } catch { /* ignore */ }
            toolUses.push({
              toolUseId: tc.id,
              name: toolNameRegistry.toKiroName(tc.function.name),
              input
            })
          }
        }
      }

      history.push({
        assistantResponseMessage: {
          content: assistantContent,
          toolUses: toolUses.length > 0 ? toolUses : undefined
        }
      })
    } else if (msg.role === 'tool') {
      // Tool result - 收集到待处理列表
      if (msg.tool_call_id) {
        let rawText = ''
        let extractedImageCount = 0
        // content 是数组时（部分客户端把图像/多模态结果挂在这里）：
        // 提取所有 text 块拼接为文本；image_url 块提取到外层 images，避免被 JSON.stringify 序列化丢失
        if (Array.isArray(msg.content)) {
          const textParts: string[] = []
          for (const part of msg.content) {
            if (part.type === 'text' && typeof part.text === 'string') {
              textParts.push(part.text)
            } else if (part.type === 'image_url' && part.image_url?.url) {
              const img = parseImageUrl(part.image_url.url)
              if (img) { images.push(img); extractedImageCount++ }
            }
          }
          rawText = textParts.join('')
          if (!rawText && extractedImageCount === 0) {
            // 退化：把不识别的结构 stringify 让模型至少看到原始结构
            rawText = JSON.stringify(msg.content)
          }
          if (extractedImageCount > 0) {
            rawText = (rawText ? rawText + '\n\n' : '') +
              `[Tool returned ${extractedImageCount} image${extractedImageCount > 1 ? 's' : ''}, attached to this message]`
          }
        } else {
          rawText = typeof msg.content === 'string' ? msg.content : JSON.stringify(msg.content)
        }
        toolResults.push({
          toolUseId: msg.tool_call_id,
          content: [{ text: rawText || '(no output)' }],
          status: 'success'
        })
      }
      
      // 检查下一条消息：如果不是 tool 消息或已到末尾，将收集的 toolResults 添加为 user 消息
      const nextMsg = nonSystemMessages[i + 1]
      const shouldFlush = !nextMsg || nextMsg.role !== 'tool'
      
      if (shouldFlush && toolResults.length > 0 && !isLast) {
        // 将 toolResults 作为 user 消息添加到 history
        history.push({
          userInputMessage: {
            content: 'Tool results provided.',
            modelId,
            origin,
            userInputMessageContext: {
              toolResults: [...toolResults]
            }
          }
        })
        // 清空已处理的 toolResults
        toolResults.length = 0
      }
    }
  }

  // 如果最后一条是 assistant 消息，自动发送 Continue（参考 Proxycast）
  if (history.length > 0 && history[history.length - 1].assistantResponseMessage && !currentContent) {
    currentContent = 'Continue.'
  }

  // 如果没有当前内容但有工具结果（最后一轮的），保留它们传给 currentMessage
  if (!currentContent && toolResults.length > 0) {
    currentContent = 'Tool results provided.'
  }

  // System prompt 以 Kiro 官方方式注入：作为 Human/AI pair 插入到 history 头部
  if (systemPrompt) {
    const systemMessages: KiroHistoryMessage[] = [
      {
        userInputMessage: {
          content: systemPrompt,
          userInputMessageContext: {},
          origin,
          ...(systemCachePoint ? { cachePoint: systemCachePoint } : {})
        }
      },
      {
        assistantResponseMessage: {
          content: 'I will follow these instructions.'
        }
      }
    ]
    history.unshift(...systemMessages)
  }
  const finalContent = currentContent || 'Continue.'

  // 转换工具定义
  const kiroTools = convertOpenAITools(
    normalizeOpenAITools(request.tools || []),
    toolNameRegistry
  )

  // OpenAI 兼容请求的 thinking/reasoning_effort 映射到 Kiro additionalModelRequestFields
  const additionalModelRequestFields = buildThinkingFields(
    thinkingConfig,
    request.thinking as { type: string; budget_tokens?: number },
    request.reasoning_effort
  )

  return buildKiroPayload(
    finalContent,
    modelId,
    origin,
    history,
    kiroTools,
    toolResults,
    images,
    profileArn,
    {
      maxTokens: request.max_tokens,
      temperature: request.temperature,
      topP: request.top_p
    },
    {
      cachePoint: currentCachePoint,
      documents,
      conversationId: request.conversation_id,
      context: request.kiro_context
    },
    additionalModelRequestFields
  )
}

function extractOpenAIContent(msg: OpenAIMessage): { content: string; images: KiroImage[]; documents: KiroDocument[]; cachePoint?: KiroCachePoint } {
  const images: KiroImage[] = []
  const documents: KiroDocument[] = []
  let content = ''
  let cachePoint = toKiroCachePoint(msg.cache_control)

  if (typeof msg.content === 'string') {
    content = msg.content
  } else if (Array.isArray(msg.content)) {
    for (const part of msg.content) {
      cachePoint = mergeCachePoint(cachePoint, toKiroCachePoint(part.cache_control))
      if (part.type === 'text' && part.text) {
        content += part.text
      } else if (part.type === 'image_url' && part.image_url?.url) {
        const image = parseImageUrl(part.image_url.url)
        if (image) {
          images.push(image)
        }
      } else if (part.type === 'file' || part.type === 'document') {
        if (part.file?.file_data) {
          const name = part.file.filename || part.name
          if (!name) {
            throw new Error(`${part.type} requires filename or name`)
          }
          documents.push(parseOpenAIFileData(part.file.file_data, name))
        } else if (part.source) {
          if (!part.name) {
            throw new Error(`${part.type} requires name`)
          }
          documents.push(parseClaudeDocumentSource(part.source, part.name))
        } else {
          throw new Error(`${part.type} requires file_data or source`)
        }
      }
    }
  }

  return { content, images, documents, cachePoint }
}

// 解析图像 URL（支持 data URL 和 HTTP URL）
function parseImageUrl(url: string): KiroImage | null {
  if (url.startsWith('data:')) {
    // 解析 data URL: data:image/png;base64,xxxxx
    const match = url.match(/^data:image\/(\w+);base64,(.+)$/)
    if (match) {
      return {
        format: normalizeImageFormat(match[1]),
        source: { bytes: match[2] }
      }
    }
  }
  return null
}

function parseOpenAIFileData(fileData: string, name: string): KiroDocument {
  const dataUrlMatch = fileData.match(/^data:([^;]+);base64,(.+)$/)
  if (dataUrlMatch) {
    return {
      format: normalizeDocumentFormat(dataUrlMatch[1], name),
      name,
      source: { bytes: dataUrlMatch[2] }
    }
  }

  return {
    format: normalizeDocumentFormat(undefined, name),
    name,
    source: { bytes: fileData }
  }
}

function parseClaudeDocumentSource(source: NonNullable<ClaudeContentBlock['source']>, name: string): KiroDocument {
  if (source.type === 'base64') {
    return {
      format: normalizeDocumentFormat(source.media_type, name),
      name,
      source: { bytes: source.data }
    }
  }
  if (source.type === 'text') {
    return {
      format: normalizeDocumentFormat(source.media_type, name),
      name,
      source: { bytes: Buffer.from(source.data, 'utf8').toString('base64') }
    }
  }
  throw new Error(`Unsupported document source type: ${source.type}`)
}

// 标准化图像格式
function normalizeImageFormat(format: string): string {
  const lower = format.toLowerCase()
  const formatMap: Record<string, string> = {
    'jpg': 'jpeg',
    'jpeg': 'jpeg',
    'png': 'png',
    'gif': 'gif',
    'webp': 'webp'
  }
  const normalized = formatMap[lower]
  if (!normalized) {
    throw new Error(`Unsupported image format: ${format}`)
  }
  return normalized
}

function normalizeDocumentFormat(mediaType: string | undefined, name: string): string {
  const lowerMediaType = mediaType?.toLowerCase()
  if (lowerMediaType === 'application/pdf') return 'pdf'
  if (lowerMediaType === 'text/markdown') return 'md'
  if (lowerMediaType === 'text/csv') return 'csv'
  if (lowerMediaType === 'text/html') return 'html'
  if (lowerMediaType?.startsWith('text/')) return 'txt'
  const extension = name.split('.').pop()?.toLowerCase()
  if (extension === 'pdf') return 'pdf'
  if (extension === 'md' || extension === 'markdown') return 'md'
  if (extension === 'csv') return 'csv'
  if (extension === 'html' || extension === 'htm') return 'html'
  return 'txt'
}


// Kiro API 工具描述最大长度
const KIRO_MAX_TOOL_DESC_LEN = 10237 // 留出 "..." 的空间

function convertOpenAITools(
  tools: OpenAITool[] | undefined,
  toolNameRegistry: ToolNameRegistry
): KiroToolWrapper[] {
  if (!tools) return []

  return tools.flatMap(tool => {
    let description = tool.function.description || `Tool: ${tool.function.name}`
    // 截断过长的描述
    if (description.length > KIRO_MAX_TOOL_DESC_LEN) {
      description = description.substring(0, KIRO_MAX_TOOL_DESC_LEN) + '...'
    }
    const kiroTool: KiroToolWrapper = {
      toolSpecification: {
        name: shortenToolName(tool.function.name, toolNameRegistry),
        description,
        inputSchema: { json: tool.function.parameters }
      }
    }
    const cachePoint = toKiroCachePoint(tool.cache_control)
    return cachePoint ? [kiroTool, { cachePoint }] : [kiroTool]
  })
}

function shortenToolName(name: string, toolNameRegistry: ToolNameRegistry): string {
  return toolNameRegistry.toKiroName(name)
}

// ============ Kiro -> OpenAI 转换 ============

export function kiroToOpenaiResponse(
  content: string,
  toolUses: KiroToolUse[],
  usage: KiroUsage,
  model: string,
  toolNameRegistry: ToolNameRegistry = new ToolNameRegistry(),
  reasoningContent?: { text?: string; signature?: string; redactedContent?: string }
): OpenAIChatResponse {
  const restoredToolUses = toolNameRegistry.restoreToolUses(toolUses)
  const openaiUsage: OpenAIChatResponse['usage'] = {
    prompt_tokens: usage.inputTokens,
    completion_tokens: usage.outputTokens,
    total_tokens: usage.inputTokens + usage.outputTokens
  }
  if (usage.cacheReadTokens) {
    openaiUsage.prompt_tokens_details = {
      cached_tokens: usage.cacheReadTokens
    }
  }
  if (usage.reasoningTokens) {
    openaiUsage.completion_tokens_details = {
      reasoning_tokens: usage.reasoningTokens
    }
  }
  const response: OpenAIChatResponse = {
    id: `chatcmpl-${uuidv4()}`,
    object: 'chat.completion',
    created: Math.floor(Date.now() / 1000),
    model,
    choices: [{
      index: 0,
      message: {
        role: 'assistant',
        content: (restoredToolUses.length > 0 || !content?.trim()) ? null : content,
        ...(reasoningContent?.text ? { reasoning_content: reasoningContent.text } : {}),
        tool_calls: restoredToolUses.length > 0 ? restoredToolUses.map(tu => ({
          id: tu.toolUseId,
          type: 'function' as const,
          function: {
            name: tu.name,
            arguments: JSON.stringify(tu.input)
          }
        })) : undefined
      },
      finish_reason: restoredToolUses.length > 0 ? 'tool_calls' : 'stop'
    }],
    usage: openaiUsage
  }

  return response
}

export interface OpenAIUsage {
  prompt_tokens: number
  completion_tokens: number
  total_tokens: number
  prompt_tokens_details?: {
    cached_tokens?: number
  }
  completion_tokens_details?: {
    reasoning_tokens?: number
  }
}

export function createOpenaiStreamChunk(
  id: string,
  model: string,
  delta: { role?: 'assistant'; content?: string; reasoning_content?: string; tool_calls?: { index: number; id?: string; type?: 'function'; function?: { name?: string; arguments?: string } }[] },
  finishReason: 'stop' | 'tool_calls' | null = null,
  usage?: OpenAIUsage
): OpenAIStreamChunk & { usage?: OpenAIUsage } {
  const chunk: OpenAIStreamChunk & { usage?: OpenAIUsage } = {
    id,
    object: 'chat.completion.chunk',
    created: Math.floor(Date.now() / 1000),
    model,
    choices: [{
      index: 0,
      delta: delta as OpenAIStreamChunk['choices'][0]['delta'],
      finish_reason: finishReason
    }]
  }
  if (usage) {
    chunk.usage = usage
  }
  return chunk
}

// ============ Claude -> Kiro 转换 ============

export function claudeToKiro(
  request: ClaudeRequest,
  profileArn?: string,
  toolNameRegistry: ToolNameRegistry = new ToolNameRegistry(),
  thinkingConfig?: ThinkingConfig
): KiroPayload {
  const modelId = mapModelId(request.model)
  const origin = 'AI_EDITOR'

  // 提取系统提示
  let systemPrompt = ''
  let systemCachePoint: KiroCachePoint | undefined
  if (typeof request.system === 'string') {
    systemPrompt = request.system
  } else if (Array.isArray(request.system)) {
    systemPrompt = request.system.map(b => {
      systemCachePoint = mergeCachePoint(systemCachePoint, toKiroCachePoint(b.cache_control))
      return b.text
    }).join('\n')
  }

  // 注入时间戳
  const timestamp = new Date().toISOString()
  systemPrompt = `[Context: Current time is ${timestamp}]\n\n${systemPrompt}`

  // 注入执行导向指令（防止 AI 在探索过程中丢失目标）
  const executionDirective = `
<execution_discipline>
当用户要求执行特定任务时，你必须遵循以下纪律：
1. **目标锁定**：在整个会话中始终牢记用户的原始目标，不要在代码探索过程中迷失方向
2. **行动优先**：优先执行任务而非仅分析或总结，除非用户明确只要求分析
3. **工具优先**：当请求提供了终端、文件、搜索或补丁工具时，涉及仓库/文件的事实必须先调用工具验证；禁止声称无法访问本地工作区，也禁止把应由工具执行的命令仅作为代码块交给用户运行
4. **工作区一致**：所有命令都在客户端当前工作区执行。Git 命令失败时先检查当前目录和 Git 安全目录配置，再采取修复措施，不要直接把命令转嫁给用户
5. **计划执行**：为任务创建明确的步骤计划，逐步执行并标记完成状态
6. **禁止确认性收尾**：在任务未完成前，禁止输出"需要我继续吗？"、"需要深入分析吗？"等确认性问题
7. **持续推进**：如果发现部分任务已完成，立即继续执行剩余未完成的任务
8. **完整交付**：直到所有任务步骤都执行完毕才算完成
</execution_discipline>
`
  systemPrompt = systemPrompt + '\n\n' + executionDirective

  // 构建历史消息 - Kiro API 要求严格的 user -> assistant 交替
  const history: KiroHistoryMessage[] = []
  let currentToolResults: KiroToolResult[] = []  // 只保存最后一条消息的 toolResults
  let currentContent = ''
  let currentCachePoint: KiroCachePoint | undefined
  const images: KiroImage[] = []
  const documents: KiroDocument[] = []

  // 临时存储，用于合并连续的同类型消息
  let pendingUserContent = ''
  let pendingUserImages: KiroImage[] = []
  let pendingUserDocuments: KiroDocument[] = []
  let pendingToolResults: KiroToolResult[] = []
  let pendingUserCachePoint: KiroCachePoint | undefined

  for (let i = 0; i < request.messages.length; i++) {
    const msg = request.messages[i]
    const isLast = i === request.messages.length - 1

    if (msg.role === 'user') {
      const { content: userContent, images: userImages, documents: userDocuments, toolResults: userToolResults, cachePoint: userCachePoint } = extractClaudeContent(msg)

      if (isLast) {
        // 最后一条消息：合并之前的 pending 内容，toolResults 放入 currentMessage
        currentContent = pendingUserContent ? pendingUserContent + '\n' + userContent : userContent
        images.push(...pendingUserImages, ...userImages)
        documents.push(...pendingUserDocuments, ...userDocuments)
        currentToolResults = [...pendingToolResults, ...userToolResults]
        currentCachePoint = mergeCachePoint(pendingUserCachePoint, userCachePoint)
        pendingUserContent = ''
        pendingUserImages = []
        pendingUserDocuments = []
        pendingToolResults = []
        pendingUserCachePoint = undefined
      } else {
        // 非最后一条：检查下一条是否是 assistant
        const nextMsg = request.messages[i + 1]
        if (nextMsg && nextMsg.role === 'assistant') {
          // 下一条是 assistant，可以安全添加到 history
          const finalUserContent = pendingUserContent ? pendingUserContent + '\n' + userContent : userContent
          const finalUserImages = [...pendingUserImages, ...userImages]
          const finalUserDocuments = [...pendingUserDocuments, ...userDocuments]
          const finalToolResults = [...pendingToolResults, ...userToolResults]
          const finalCachePoint = mergeCachePoint(pendingUserCachePoint, userCachePoint)
          
          if (finalUserContent.trim() || finalUserImages.length > 0 || finalUserDocuments.length > 0 || finalToolResults.length > 0) {
            const userInputMessage: KiroUserInputMessage = {
              content: finalUserContent || (finalToolResults.length > 0 ? 'Tool results provided.' : 'Continue'),
              modelId,
              origin,
              images: finalUserImages.length > 0 ? finalUserImages : undefined,
              documents: finalUserDocuments.length > 0 ? finalUserDocuments : undefined,
              ...(finalCachePoint ? { cachePoint: finalCachePoint } : {})
            }
            // 如果有 toolResults，放入 userInputMessageContext
            if (finalToolResults.length > 0) {
              userInputMessage.userInputMessageContext = {
                toolResults: finalToolResults
              }
            }
            history.push({ userInputMessage })
          }
          pendingUserContent = ''
          pendingUserImages = []
          pendingUserDocuments = []
          pendingToolResults = []
          pendingUserCachePoint = undefined
        } else {
          // 下一条不是 assistant（可能是连续 user 或结束），累积内容
          pendingUserContent = pendingUserContent ? pendingUserContent + '\n' + userContent : userContent
          pendingUserImages.push(...userImages)
          pendingUserDocuments.push(...userDocuments)
          pendingToolResults.push(...userToolResults)
          pendingUserCachePoint = mergeCachePoint(pendingUserCachePoint, userCachePoint)
        }
      }
    } else if (msg.role === 'assistant') {
      // 注意: 故意丢弃 reasoningContent (history 中不传给 Kiro)
      // Kiro 后端 schema 仅在响应输出中支持 assistantResponseMessage.reasoningContent，
      // 在请求 history 中传入此字段会触发 400 "Improperly formed request"
      // 当前消息的 thinking 开关由 additionalModelRequestFields.thinking 控制
      const { content: assistantContent, toolUses } = extractClaudeAssistantContent(msg, toolNameRegistry)

      // 如果有 pending 的 user 内容但还没添加到 history，先添加
      if (pendingUserContent.trim() || pendingUserImages.length > 0 || pendingUserDocuments.length > 0 || pendingToolResults.length > 0) {
        const userInputMessage: KiroUserInputMessage = {
          content: pendingUserContent || (pendingToolResults.length > 0 ? 'Tool results provided.' : 'Continue'),
          modelId,
          origin,
          images: pendingUserImages.length > 0 ? pendingUserImages : undefined,
          documents: pendingUserDocuments.length > 0 ? pendingUserDocuments : undefined,
          ...(pendingUserCachePoint ? { cachePoint: pendingUserCachePoint } : {})
        }
        if (pendingToolResults.length > 0) {
          userInputMessage.userInputMessageContext = {
            toolResults: pendingToolResults
          }
        }
        history.push({ userInputMessage })
        pendingUserContent = ''
        pendingUserImages = []
        pendingUserDocuments = []
        pendingToolResults = []
        pendingUserCachePoint = undefined
      }

      const assistantResponseMessage = {
        content: assistantContent,
        ...(toolUses.length > 0 ? { toolUses } : {})
      }
      history.push({ assistantResponseMessage })
    }
  }

  // 处理剩余的 pending 内容（如果最后几条都是 user 且不是 isLast）
  if (pendingUserContent.trim() || pendingUserImages.length > 0 || pendingUserDocuments.length > 0 || pendingToolResults.length > 0) {
    currentContent = pendingUserContent + (currentContent ? '\n' + currentContent : '')
    images.unshift(...pendingUserImages)
    documents.unshift(...pendingUserDocuments)
    currentToolResults = [...pendingToolResults, ...currentToolResults]
    currentCachePoint = mergeCachePoint(pendingUserCachePoint, currentCachePoint)
  }

  // 确保 history 以 user 开始（Kiro API 要求）
  // 如果 history 以 assistant 开始，在前面插入一个空的 user 消息
  if (history.length > 0 && history[0].assistantResponseMessage) {
    history.unshift({
      userInputMessage: {
        content: 'Begin conversation',
        modelId,
        origin
      }
    })
  }

  // 构建最终内容
  // System prompt 以 Kiro 官方方式注入：作为 Human/AI pair 插入到 history 头部
  // 官方 Kiro IDE: [Human(systemPrompt, forcedRole), AI("I will follow these instructions.", forcedRole)]
  if (systemPrompt) {
    const systemMessages: KiroHistoryMessage[] = [
      {
        userInputMessage: {
          content: systemPrompt,
          userInputMessageContext: {},
          origin,
          ...(systemCachePoint ? { cachePoint: systemCachePoint } : {})
        }
      },
      {
        assistantResponseMessage: {
          content: 'I will follow these instructions.'
        }
      }
    ]
    history.unshift(...systemMessages)
  }
  const finalContent = currentContent || (currentToolResults.length > 0 ? 'Tool results provided.' : 'Continue')

  // 转换工具定义
  const kiroTools = convertClaudeTools(request.tools, toolNameRegistry)

  // 将 Claude thinking 参数映射为 Kiro additionalModelRequestFields
  const additionalModelRequestFields = buildThinkingFields(
    thinkingConfig,
    request.thinking as { type: string; budget_tokens?: number; display?: string },
    undefined
  )

  return buildKiroPayload(
    finalContent,
    modelId,
    origin,
    history,
    kiroTools,
    currentToolResults,
    images,
    profileArn,
    {
      maxTokens: request.max_tokens,
      temperature: request.temperature,
      topP: request.top_p
    },
    {
      cachePoint: currentCachePoint,
      documents,
      conversationId: request.conversation_id,
      context: request.kiro_context
    },
    additionalModelRequestFields
  )
}

function extractClaudeContent(msg: ClaudeMessage): { content: string; images: KiroImage[]; documents: KiroDocument[]; toolResults: KiroToolResult[]; cachePoint?: KiroCachePoint } {
  const images: KiroImage[] = []
  const documents: KiroDocument[] = []
  const toolResults: KiroToolResult[] = []
  let content = ''
  let cachePoint = toKiroCachePoint(msg.cache_control)

  if (typeof msg.content === 'string') {
    content = msg.content
  } else if (Array.isArray(msg.content)) {
    for (const block of msg.content) {
      cachePoint = mergeCachePoint(cachePoint, toKiroCachePoint(block.cache_control))
      if (block.type === 'text' && block.text) {
        content += block.text
      } else if (block.type === 'image' && block.source?.type === 'base64') {
        const mediaTypeParts = block.source.media_type.split('/')
        const imageFormat = mediaTypeParts[1]
        if (mediaTypeParts[0] !== 'image' || !imageFormat) {
          throw new Error(`Unsupported image media_type: ${block.source.media_type}`)
        }
        images.push({
          format: normalizeImageFormat(imageFormat),
          source: { bytes: block.source.data }
        })
      } else if (block.type === 'document' && block.source) {
        if (!block.name) {
          throw new Error('document requires name')
        }
        documents.push(parseClaudeDocumentSource(block.source, block.name))
      } else if (block.type === 'tool_result' && block.tool_use_id) {
        let resultContent = ''
        // Kiro tool_result.content 只支持 text，但用户层 images 可以承载图片。
        // 把内嵌 image block 提取到外层 images，避免「读取本地图片」这类场景图像内容被静默丢弃。
        let extractedImageCount = 0
        if (typeof block.content === 'string') {
          resultContent = block.content || '(empty)'
        } else if (Array.isArray(block.content)) {
          const textParts: string[] = []
          for (const b of block.content) {
            if (b.type === 'text') {
              textParts.push(b.text || '')
            } else if (b.type === 'image' && b.source?.type === 'base64' && b.source.data) {
              const mediaTypeParts = (b.source.media_type || '').split('/')
              const imageFormat = mediaTypeParts[1]
              if (mediaTypeParts[0] === 'image' && imageFormat) {
                try {
                  images.push({
                    format: normalizeImageFormat(imageFormat),
                    source: { bytes: b.source.data }
                  })
                  extractedImageCount++
                } catch {
                  // 不支持的格式：跳过但不抛错（保留旧行为，避免整轮失败）
                }
              }
            }
          }
          resultContent = textParts.join('')
          if (!resultContent) {
            resultContent = extractedImageCount > 0
              ? `(tool returned ${extractedImageCount} image${extractedImageCount > 1 ? 's' : ''}, attached to this message)`
              : '(no text output)'
          } else if (extractedImageCount > 0) {
            // 既有文本又有图片：在文本末尾提示模型有附图
            resultContent += `\n\n[Tool also returned ${extractedImageCount} image${extractedImageCount > 1 ? 's' : ''}, attached to this message]`
          }
        } else if (block.content === undefined || block.content === null) {
          resultContent = '(no output)'
        } else {
          resultContent = String(block.content) || '(empty)'
        }
        toolResults.push({
          toolUseId: block.tool_use_id,
          content: [{ text: resultContent }],
          status: 'success'
        })
      }
    }
  }

  return { content, images, documents, toolResults, cachePoint }
}

function extractClaudeAssistantContent(
  msg: ClaudeMessage,
  toolNameRegistry: ToolNameRegistry
): { content: string; toolUses: KiroToolUse[]; reasoningContent?: KiroReasoningContent } {
  const toolUses: KiroToolUse[] = []
  let content = ''
  let thinking = ''
  let signature: string | undefined
  let redactedContent: string | undefined

  if (typeof msg.content === 'string') {
    content = msg.content
  } else if (Array.isArray(msg.content)) {
    for (const block of msg.content) {
      if (block.type === 'text' && block.text) {
        content += block.text
      } else if (block.type === 'thinking' && block.thinking) {
        thinking += block.thinking
        signature = block.signature || signature
      } else if (block.type === 'redacted_thinking' && block.data) {
        // redacted_thinking 是加密的思考内容，原样保留
        redactedContent = (redactedContent || '') + block.data
      } else if (block.type === 'tool_use' && block.id && block.name) {
        if (!block.input || typeof block.input !== 'object' || Array.isArray(block.input)) {
          throw new Error(`tool_use requires object input: ${block.name}`)
        }
        toolUses.push({
          toolUseId: block.id,
          name: toolNameRegistry.toKiroName(block.name),
          input: block.input as Record<string, unknown>
        })
      }
    }
  }

  // Kiro API 要求 content 非空
  if (!content.trim() && toolUses.length > 0) {
    content = ' '
  }

  if (thinking || redactedContent) {
    const reasoningContent: KiroReasoningContent = {}
    if (thinking) {
      reasoningContent.reasoningText = signature ? { text: thinking, signature } : { text: thinking }
    }
    if (redactedContent) {
      reasoningContent.redactedContent = redactedContent
    }
    return { content, toolUses, reasoningContent }
  }

  return { content, toolUses }
}

function convertClaudeTools(
  tools: { name: string; description: string; input_schema: unknown; cache_control?: { type: string } }[] | undefined,
  toolNameRegistry: ToolNameRegistry
): KiroToolWrapper[] {
  if (!tools) return []

  return tools.flatMap(tool => {
    let description = tool.description || `Tool: ${tool.name}`
    // 截断过长的描述
    if (description.length > KIRO_MAX_TOOL_DESC_LEN) {
      description = description.substring(0, KIRO_MAX_TOOL_DESC_LEN) + '...'
    }
    const kiroTool: KiroToolWrapper = {
      toolSpecification: {
        name: shortenToolName(tool.name, toolNameRegistry),
        description,
        inputSchema: { json: tool.input_schema }
      }
    }
    const cachePoint = toKiroCachePoint(tool.cache_control)
    return cachePoint ? [kiroTool, { cachePoint }] : [kiroTool]
  })
}

// ============ Kiro -> Claude 转换 ============

export function kiroToClaudeResponse(
  content: string,
  toolUses: KiroToolUse[],
  usage: KiroUsage,
  model: string,
  toolNameRegistry: ToolNameRegistry = new ToolNameRegistry(),
  reasoningContent?: { text?: string; signature?: string; redactedContent?: string }
): ClaudeResponse {
  const contentBlocks: ClaudeContentBlock[] = []
  const restoredToolUses = toolNameRegistry.restoreToolUses(toolUses)

  if (reasoningContent?.text) {
    contentBlocks.push(reasoningContent.signature ? {
      type: 'thinking',
      thinking: reasoningContent.text,
      signature: reasoningContent.signature
    } : {
      type: 'thinking',
      thinking: reasoningContent.text
    })
  }
  if (reasoningContent?.redactedContent) {
    contentBlocks.push({
      type: 'redacted_thinking',
      data: reasoningContent.redactedContent
    })
  }

  // 仅在有实际文本内容时添加 text block
  if (content && content.trim()) {
    contentBlocks.push({ type: 'text', text: content })
  }

  for (const tu of restoredToolUses) {
    contentBlocks.push({
      type: 'tool_use',
      id: tu.toolUseId,
      name: tu.name,
      input: tu.input
    })
  }

  const claudeUsage: ClaudeResponse['usage'] = {
    input_tokens: usage.inputTokens,
    output_tokens: usage.outputTokens
  }
  if (usage.cacheWriteTokens) {
    claudeUsage.cache_creation_input_tokens = usage.cacheWriteTokens
  }
  if (usage.cacheReadTokens) {
    claudeUsage.cache_read_input_tokens = usage.cacheReadTokens
  }

  const response: ClaudeResponse = {
    id: `msg_${uuidv4()}`,
    type: 'message',
    role: 'assistant',
    content: contentBlocks,
    model,
    stop_reason: restoredToolUses.length > 0 ? 'tool_use' : 'end_turn',
    stop_sequence: null,
    usage: claudeUsage
  }
  return response
}

export function createClaudeStreamEvent(
  type: ClaudeStreamEvent['type'],
  data?: Partial<ClaudeStreamEvent>
): ClaudeStreamEvent {
  return { type, ...data } as ClaudeStreamEvent
}
