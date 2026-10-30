// 统一敏感信息脱敏工具
//
// 用途：在写日志 / 推送日志到 UI 前，把代理账密、token、password、apiKey 等敏感字段脱敏，
// 避免凭据明文落到磁盘日志或在界面/控制台暴露。
//
// 设计：
//   - redactString：处理裸字符串里的代理 URL 账密、Bearer token、长 JWT 串
//   - redactValue：递归处理对象/数组，对"敏感键名"整体打码，并对字符串值套用 redactString
//   - 防御递归深度与循环引用，避免日志路径上出问题

/** 命中即整体打码的敏感键名（小写匹配） */
const SENSITIVE_KEYS = [
  'password', 'passwd', 'pwd',
  'accesstoken', 'access_token',
  'refreshtoken', 'refresh_token',
  'idtoken', 'id_token',
  'bearertoken', 'bearer',
  'authorization', 'auth',
  'apikey', 'api_key', 'x-api-key',
  'clientsecret', 'client_secret',
  'secret', 'epin', 'cookie', 'set-cookie',
  'proxyauthorization', 'proxy-authorization'
]

/** 白名单：这些键虽然包含敏感子串但本身是安全的计量/统计字段 */
const SAFE_KEYS = new Set([
  'inputtokens', 'outputtokens', 'cachetokens',
  'cachereadtokens', 'cachewritetokens', 'reasoningtokens',
  'totaltokens', 'maxtokens', 'tokensused', 'tokencount'
])

/** 仅保留头尾少量字符，中间打码；过短直接全打码 */
function maskMiddle(value: string, head = 3, tail = 2): string {
  if (!value) return value
  if (value.length <= head + tail + 2) return '***'
  return `${value.slice(0, head)}***${value.slice(-tail)}`
}

/** 脱敏裸字符串中的常见敏感片段 */
export function redactString(input: string): string {
  if (!input) return input
  let out = input

  // 1) 代理 / 任意 URL 里的 user:pass@host → user:***@host
  out = out.replace(/(\b[a-z][a-z0-9+.-]*:\/\/)([^/\s:@]+):([^/\s@]+)@/gi, (_m, scheme, user) => {
    return `${scheme}${user}:***@`
  })

  // 2) Authorization: Bearer xxx / Basic xxx
  out = out.replace(/(authorization\s*[:=]\s*)(bearer|basic)\s+([A-Za-z0-9._\-+/=]+)/gi, (_m, p, scheme) => {
    return `${p}${scheme} ***`
  })

  // 3) JWT（三段 base64url）整体打码
  out = out.replace(/\beyJ[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{5,}\b/g, (m) => maskMiddle(m, 6, 4))

  // 4) 形如 accessToken=xxx / "refreshToken":"xxx" 的内联键值（兜底，针对已被 JSON.stringify 的串）
  out = out.replace(/("?(?:access_?token|refresh_?token|id_?token|password|api_?key|client_?secret|secret|epin)"?\s*[:=]\s*"?)([^",}\s]+)("?)/gi,
    (_m, prefix, val, suffix) => `${prefix}${maskMiddle(String(val))}${suffix}`)

  return out
}

function isSensitiveKey(key: string): boolean {
  const k = key.toLowerCase().replace(/[_-]/g, '')
  if (SAFE_KEYS.has(k)) return false
  return SENSITIVE_KEYS.some((s) => k === s.replace(/[_-]/g, '') || k.includes(s.replace(/[_-]/g, '')))
}

/** 单个数组/对象最多保留的元素数，防止超大结构拖慢脱敏与日志写入 */
const MAX_COLLECTION_ITEMS = 200

/**
 * 递归脱敏任意值（对象/数组/字符串），返回与输入无引用关系的副本。
 * - maxDepth 防御过深结构与循环引用；超大数组/对象只保留前 MAX_COLLECTION_ITEMS 项
 * - 二进制数据只保留类型与字节数，Date 转 ISO 字符串，Error 显式保留 name/message/stack（它们不可枚举）
 * @param value 任意待脱敏的值
 * @param maxDepth 剩余可展开深度
 * @param seen 已访问对象集合，用于识别循环引用
 * @returns 脱敏后的值
 */
export function redactValue(value: unknown, maxDepth = 6, seen = new WeakSet<object>()): unknown {
  if (value == null) return value
  if (typeof value === 'string') return redactString(value)
  if (typeof value === 'number' || typeof value === 'boolean') return value
  if (typeof value === 'bigint') return value.toString()
  if (typeof value !== 'object') return String(value)
  // 二进制数据逐字节展开成对象代价极高，只记录类型与长度
  if (ArrayBuffer.isView(value)) return `[${value.constructor?.name ?? 'Binary'} ${value.byteLength} bytes]`
  if (value instanceof ArrayBuffer) return `[ArrayBuffer ${value.byteLength} bytes]`
  if (value instanceof Date) return isNaN(value.getTime()) ? 'Invalid Date' : value.toISOString()
  if (maxDepth <= 0) return '[depth-limit]'
  if (seen.has(value)) return '[circular]'
  seen.add(value)

  if (value instanceof Map) return redactValue(Object.fromEntries(value), maxDepth - 1, seen)
  if (value instanceof Set) return redactValue(Array.from(value), maxDepth - 1, seen)

  if (Array.isArray(value)) {
    const out = value.slice(0, MAX_COLLECTION_ITEMS).map((v) => redactValue(v, maxDepth - 1, seen))
    if (value.length > MAX_COLLECTION_ITEMS) out.push(`[+${value.length - MAX_COLLECTION_ITEMS} more items]`)
    return out
  }

  const out: Record<string, unknown> = {}
  if (value instanceof Error) {
    out.name = value.name
    out.message = redactString(value.message)
    if (value.stack) out.stack = redactString(value.stack)
  }
  const keys = Object.keys(value)
  for (const k of keys.length > MAX_COLLECTION_ITEMS ? keys.slice(0, MAX_COLLECTION_ITEMS) : keys) {
    const v = (value as Record<string, unknown>)[k]
    if (isSensitiveKey(k)) {
      out[k] = typeof v === 'string' ? maskMiddle(v) : '***'
    } else {
      out[k] = redactValue(v, maxDepth - 1, seen)
    }
  }
  if (keys.length > MAX_COLLECTION_ITEMS) out['…'] = `[+${keys.length - MAX_COLLECTION_ITEMS} more keys]`
  return out
}
