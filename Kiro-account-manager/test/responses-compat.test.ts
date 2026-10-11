import { describe, expect, mock, test } from 'bun:test'
import type { OpenAIResponsesRequest } from '../src/main/proxy/types'

mock.module('electron', () => ({
  app: { getPath: () => process.cwd() }
}))

const {
  openaiToKiro,
  openAIChatToResponsesResponse,
  normalizeOpenAITools,
  responsesToOpenAIChat
} = await import('../src/main/proxy/translator')
const { mapModelId } = await import('../src/main/proxy/kiroApi')
const { getModelContextLength } = await import('../src/main/proxy/tokenCounter')

describe('Kiro-native GPT-5.6 model routing', () => {
  test('passes native variants through and keeps the base model as a Sol alias', () => {
    expect(mapModelId('gpt-5.6-sol')).toBe('gpt-5.6-sol')
    expect(mapModelId('gpt-5.6-terra')).toBe('gpt-5.6-terra')
    expect(mapModelId('gpt-5.6-luna')).toBe('gpt-5.6-luna')
    expect(mapModelId('gpt-5.6')).toBe('gpt-5.6-sol')
    expect(mapModelId('gpt-5-6-terra')).toBe('gpt-5.6-terra')
    expect(mapModelId('codex-auto-review')).toBe('gpt-5.6-sol')
    expect(getModelContextLength('gpt-5.6-sol')).toBe(272000)
  })
})

describe('OpenAI Chat Completions compatibility', () => {
  test('normalizes Cursor-style tools passed through Chat Completions', () => {
    const request = {
      model: 'gpt-5.6-sol',
      messages: [{ role: 'user' as const, content: 'Inspect the workspace.' }],
      tools: [
        {
          type: 'function',
          function: {
            name: 'nested',
            description: 'nested tool',
            parameters: { type: 'object', properties: {} }
          }
        },
        {
          type: 'function',
          name: 'flat',
          description: 'flat tool',
          parameters: { type: 'object', properties: { path: { type: 'string' } } }
        },
        {
          type: 'custom',
          name: 'apply_patch',
          description: 'Apply a patch to files in the workspace.'
        },
        { type: 'local_shell' },
        { type: 'web_search_preview' }
      ]
    }

    const converted = openaiToKiro(request)
    const tools = converted.conversationState.currentMessage.userInputMessage
      .userInputMessageContext?.tools || []
    const specifications = tools.flatMap(tool =>
      'toolSpecification' in tool ? [tool.toolSpecification] : []
    )

    expect(specifications.map(tool => tool.name)).toEqual([
      'nested',
      'flat',
      'apply_patch',
      'exec_command'
    ])
    expect(specifications[0]).toMatchObject({
      description: 'nested tool',
      inputSchema: { json: { type: 'object', properties: {} } }
    })
    expect(specifications[1]).toMatchObject({
      description: 'flat tool',
      inputSchema: {
        json: { type: 'object', properties: { path: { type: 'string' } } }
      }
    })
    expect(specifications[2].inputSchema).toMatchObject({
      json: {
        required: ['input'],
        properties: { input: { type: 'string' } }
      }
    })
    expect(specifications[3].inputSchema).toMatchObject({
      json: {
        required: ['cmd'],
        properties: { cmd: { type: 'string' } }
      }
    })

    const normalizedOnce = normalizeOpenAITools(request.tools)
    expect(normalizeOpenAITools(normalizedOnce)).toEqual(normalizedOnce)
  })
})

describe('OpenAI Responses compatibility', () => {
  test('injects a tool-first recovery discipline for workspace tasks', () => {
    const converted = openaiToKiro({
      model: 'gpt-5.6-sol',
      messages: [{ role: 'user', content: 'Summarize the latest commits.' }],
      tools: [{
        type: 'function',
        function: {
          name: 'exec_command',
          description: 'Run a command in the workspace.',
          parameters: { type: 'object', properties: { cmd: { type: 'string' } } }
        }
      }]
    })

    const systemMessage = converted.conversationState.history?.[0]?.userInputMessage?.content || ''
    expect(systemMessage).toContain('工具优先')
    expect(systemMessage).toContain('禁止声称无法访问本地工作区')
    expect(systemMessage).toContain('Git 安全目录配置')
  })

  test('accepts Codex additional_tools and extended replay items', () => {
    const request = {
      model: 'gpt-5.6-sol',
      reasoning: { effort: 'high' },
      tools: [{
        type: 'function',
        name: 'existing_tool',
        description: 'existing',
        parameters: { type: 'object', properties: {} }
      }],
      input: [
        {
          type: 'additional_tools',
          tools: [{
            type: 'function',
            name: 'workspace_search',
            description: 'Search the workspace',
            parameters: {
              type: 'object',
              properties: { query: { type: 'string' } },
              required: ['query']
            }
          }]
        },
        { type: 'message', role: 'developer', content: 'Follow repository rules.' },
        {
          type: 'function_call',
          call_id: 'call_1',
          name: 'workspace_search',
          arguments: { query: 'translator' }
        },
        {
          type: 'function_call_output',
          call_id: 'call_1',
          output: { matches: 2 }
        },
        { type: 'reasoning', encrypted_content: 'opaque' },
        { type: 'future_state_item', data: true },
        { type: 'message', role: 'user', content: [{ type: 'input_text', text: 'Continue' }] }
      ]
    } as OpenAIResponsesRequest

    const converted = responsesToOpenAIChat(request)

    expect(converted.reasoning_effort).toBe('high')
    expect(converted.tools?.map(tool => tool.function.name)).toEqual([
      'existing_tool',
      'workspace_search'
    ])
    expect(converted.messages.map(message => message.role)).toEqual([
      'system',
      'assistant',
      'tool',
      'user'
    ])
    expect(converted.messages[1].tool_calls?.[0].function.arguments).toBe(
      '{"query":"translator"}'
    )
    expect(converted.messages[2].content).toBe('{"matches":2}')
  })

  test('normalizes nested, flattened and MCP-style function tools', () => {
    const converted = responsesToOpenAIChat({
      model: 'gpt-5.6-terra',
      input: 'hello',
      tools: [
        {
          type: 'function',
          function: {
            name: 'nested',
            description: 'nested tool',
            parameters: { type: 'object' }
          }
        },
        {
          type: 'function',
          name: 'flat',
          parameters: { type: 'object' }
        },
        {
          name: 'mcp_style',
          inputSchema: { type: 'object', properties: { path: { type: 'string' } } }
        }
      ]
    })

    expect(converted.tools?.map(tool => tool.function.name)).toEqual([
      'nested',
      'flat',
      'mcp_style'
    ])
  })

  test('keeps built-in local shell tools instead of silently dropping them', () => {
    const converted = responsesToOpenAIChat({
      model: 'gpt-5.6-sol',
      input: 'Inspect the repository.',
      tools: [{ type: 'local_shell' }]
    })

    expect(converted.tools).toHaveLength(1)
    expect(converted.tools?.[0].response_tool_type).toBe('local_shell')
    expect(converted.tools?.[0]).toMatchObject({
      type: 'function',
      function: {
        name: 'exec_command',
        parameters: {
          required: ['cmd']
        }
      }
    })
  })

  test('returns Codex local shell calls in Responses-native shape', () => {
    const converted = responsesToOpenAIChat({
      model: 'gpt-5.6-sol', input: 'Inspect the repository.', tools: [{ type: 'local_shell' }]
    })
    const response = openAIChatToResponsesResponse({
      id: 'chatcmpl_shell', object: 'chat.completion', created: 123, model: 'gpt-5.6-sol',
      choices: [{ index: 0, message: { role: 'assistant', content: null, tool_calls: [{
        id: 'call_shell', type: 'function', function: { name: 'exec_command', arguments: '{"cmd":"git status --short"}' }
      }] }, finish_reason: 'tool_calls' }], usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 }
    }, undefined, converted.tools)
    expect(response.output[0]).toMatchObject({
      type: 'local_shell_call', call_id: 'call_shell',
      action: { type: 'exec', command: ['bash', '-lc', 'git status --short'] }
    })
  })

  test('preserves Codex custom tools and custom tool replay', () => {
    const converted = responsesToOpenAIChat({
      model: 'gpt-5.6-sol',
      input: [
        { type: 'message', role: 'user', content: 'Edit the file.' },
        {
          type: 'custom_tool_call',
          call_id: 'call_patch',
          name: 'apply_patch',
          input: '*** Begin Patch\n*** End Patch'
        },
        {
          type: 'custom_tool_call_output',
          call_id: 'call_patch',
          output: 'Done!'
        }
      ],
      tools: [{
        type: 'custom',
        name: 'apply_patch',
        description: 'Apply a patch to files in the workspace.',
        format: { type: 'grammar', syntax: 'lark', definition: 'start: /.+/' }
      }],
      additional_tools: [{
        type: 'function',
        name: 'exec_command',
        description: 'Run a terminal command.',
        parameters: {
          type: 'object',
          properties: { cmd: { type: 'string' } },
          required: ['cmd']
        }
      }]
    })

    expect(converted.tools?.map(tool => tool.function.name)).toEqual([
      'apply_patch',
      'exec_command'
    ])
    expect(converted.tools?.[0]).toMatchObject({
      response_tool_type: 'custom',
      function: {
        parameters: {
          properties: { input: { type: 'string' } },
          required: ['input']
        }
      }
    })
    expect(converted.messages[1].tool_calls?.[0].function.arguments).toBe(
      '{"input":"*** Begin Patch\\n*** End Patch"}'
    )
    expect(converted.messages[2]).toMatchObject({
      role: 'tool',
      tool_call_id: 'call_patch',
      content: 'Done!'
    })

    const kiroPayload = openaiToKiro(converted)
    const kiroTools = kiroPayload.conversationState.currentMessage.userInputMessage
      .userInputMessageContext?.tools || []
    expect(kiroTools.flatMap(tool => 'toolSpecification' in tool
      ? [tool.toolSpecification.name]
      : [])).toEqual(['apply_patch', 'exec_command'])

    const response = openAIChatToResponsesResponse({
      id: 'chatcmpl_custom',
      object: 'chat.completion',
      created: 123,
      model: 'gpt-5.6-sol',
      choices: [{
        index: 0,
        message: {
          role: 'assistant',
          content: null,
          tool_calls: [{
            id: 'call_patch_2',
            type: 'function',
            function: {
              name: 'apply_patch',
              arguments: '{"input":"*** Begin Patch\\n*** End Patch"}'
            }
          }]
        },
        finish_reason: 'tool_calls'
      }],
      usage: { prompt_tokens: 2, completion_tokens: 1, total_tokens: 3 }
    }, undefined, converted.tools)

    expect(response.output[0]).toMatchObject({
      type: 'custom_tool_call',
      call_id: 'call_patch_2',
      name: 'apply_patch',
      input: '*** Begin Patch\n*** End Patch'
    })
  })

  test('returns completed Responses objects with output_text', () => {
    const converted = openAIChatToResponsesResponse({
      id: 'chatcmpl_test',
      object: 'chat.completion',
      created: 123,
      model: 'gpt-5.6-sol',
      choices: [{
        index: 0,
        message: { role: 'assistant', content: 'done' },
        finish_reason: 'stop'
      }],
      usage: { prompt_tokens: 2, completion_tokens: 1, total_tokens: 3 }
    })

    expect(converted.status).toBe('completed')
    expect(converted.output_text).toBe('done')
    expect(converted.output[0]).toMatchObject({
      type: 'message',
      status: 'completed',
      content: [{ type: 'output_text', text: 'done', annotations: [] }]
    })
  })
})
