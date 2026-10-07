// 模型别名映射配置加载模块
//
// 背景：kiroApi.ts 中的 MODEL_ID_MAP 是硬编码的代码常量，每次新增 Kiro 模型
// 都需要改代码 + 重新编译。为了让用户在不改代码的情况下也能补充 / 覆盖模型别名，
// 此模块从 userData/model-aliases.json 读取用户自定义映射，与代码内置映射取合集。
//
// 合集规则（按用户要求）：
//   - 配置文件 key 优先（同 key 时覆盖代码内置值）
//   - 合集 = { ...内置 MODEL_ID_MAP, ...customAliases }
//   - 'default' 兜底值始终来自代码内置，不允许用户配置（避免改坏）
//
// 初始化：在 main 进程启动后调用 initModelAliases(app.getPath('userData'))
//   - 文件不存在：自动写入默认模板（含 _comment 字段说明用法）
//   - 文件存在：解析为 Record<string, string>，解析失败只 warn 不影响启动
//   - 解析成功后所有 mapModelId() 调用自动使用合集
//
// 注意：此模块必须在 app.ready 之后初始化，否则 app.getPath('userData') 不可用。
// 如果在初始化前调用 mapModelId()，会安全降级到仅使用代码内置映射。

import * as fs from 'fs/promises'
import * as fsSync from 'fs'
import * as path from 'path'

// 默认配置模板：首次启动时写入 userData，告知用户如何扩展映射
const DEFAULT_CONFIG_CONTENT = `{
  "_comment": "用户自定义模型别名映射。key 是客户端请求的模型名（小写匹配），value 是 Kiro 后端实际支持的 modelId。运行时与代码内置 MODEL_ID_MAP 取合集；同 key 时本文件覆盖代码内置值。请勿修改或删除此 _comment 字段。",
  "_examples": {
    "claude-sonnet-5": "claude-sonnet-4.5",
    "claude-opus-5": "claude-opus-4.5"
  }
}
`

const CONFIG_FILE_NAME = 'model-aliases.json'

// 内置映射的副本：保证 'default' 兜底值永远来自代码，不被配置文件污染
const BUILTIN_DEFAULT = 'claude-sonnet-4.5'

// 内部状态
let customAliases: Record<string, string> = {}
let builtinAliases: Record<string, string> = {} // 在 initModelAliases 中由 kiroApi 注入
let initialized = false
let initializedUserDataDir: string | undefined

/**
 * 由 kiroApi 模块在加载时调用，把内置映射注入到此模块。
 * 这样 getEffectiveModelMap() 才能返回真正的合集。
 */
export function registerBuiltinAliases(builtin: Record<string, string>): void {
  builtinAliases = { ...builtin }
}

/**
 * 初始化：从 userData/model-aliases.json 加载用户自定义映射。
 * - 文件不存在 → 自动写入默认模板（含 _comment 说明）
 * - 文件存在但解析失败 → 仅 warn，不影响内置映射
 * - 重复调用 → 仅在 userDataDir 变化时重新加载（热重载用）
 */
export async function initModelAliases(userDataDir: string): Promise<void> {
  if (!userDataDir) {
    console.warn('[ModelAliases] userDataDir is empty, skip initialization')
    return
  }

  // 同目录重复初始化 → 跳过；不同目录 → 重新加载
  if (initialized && initializedUserDataDir === userDataDir) return
  initializedUserDataDir = userDataDir

  const configPath = path.join(userDataDir, CONFIG_FILE_NAME)

  try {
    if (!fsSync.existsSync(configPath)) {
      // 文件不存在 → 创建 userData 目录并写入默认模板
      await fs.mkdir(userDataDir, { recursive: true })
      await fs.writeFile(configPath, DEFAULT_CONFIG_CONTENT, 'utf-8')
      console.log(`[ModelAliases] Created default config at ${configPath}`)
      customAliases = {}
      initialized = true
      return
    }

    const raw = await fs.readFile(configPath, 'utf-8')
    const parsed = JSON.parse(raw) as Record<string, unknown>

    // 过滤掉 _comment / _examples 这类元数据字段，只保留 string → string 映射
    const filtered: Record<string, string> = {}
    let skipped = 0
    for (const [key, value] of Object.entries(parsed)) {
      if (key.startsWith('_')) continue // 元数据字段
      if (typeof value !== 'string' || !value.trim()) {
        skipped++
        continue
      }
      filtered[key.toLowerCase()] = value
    }

    customAliases = filtered
    initialized = true
    const customCount = Object.keys(filtered).length
    const builtinCount = Object.keys(builtinAliases).length
    console.log(
      `[ModelAliases] Loaded ${customCount} custom aliases from ${configPath} ` +
        `(builtin: ${builtinCount}, total effective: ${builtinCount + customCount}${skipped > 0 ? `, skipped ${skipped} invalid entries` : ''})`
    )
  } catch (error) {
    // 解析失败 / IO 错误 → 仅 warn，不影响启动
    console.warn(`[ModelAliases] Failed to load ${configPath}:`, error)
    console.warn('[ModelAliases] Falling back to built-in MODEL_ID_MAP only')
    customAliases = {}
    initialized = true
  }
}

/**
 * 获取有效的模型映射（合集）。
 * - 配置文件 key 优先级高于内置（实现"覆盖"语义）
 * - 即使未初始化也安全返回：内置映射始终可用
 */
export function getEffectiveModelMap(): Record<string, string> {
  // 顺序很重要：内置在前，自定义在后，自定义会覆盖内置的同名 key
  // default 兜底值始终用内置的，不允许配置文件改坏
  const merged: Record<string, string> = { ...builtinAliases, ...customAliases }
  if (!merged.default) merged.default = BUILTIN_DEFAULT
  return merged
}

/**
 * 检查是否已初始化（用于诊断日志）
 */
export function isModelAliasesInitialized(): boolean {
  return initialized
}

/**
 * 获取配置文件路径（用于在 UI 中显示给用户）
 */
export function getModelAliasesConfigPath(userDataDir: string): string {
  return path.join(userDataDir, CONFIG_FILE_NAME)
}
