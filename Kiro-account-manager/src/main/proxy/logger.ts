// 代理服务器日志模块
import * as fs from 'fs'
import * as path from 'path'
import { app } from 'electron'
import { redactString, redactValue } from '../utils/redact'

// 模块加载时（早于 interceptConsole）保存原始 console.error：日志存储自身出错时用它输出，
// 避免错误信息再次进入日志存储，形成"写盘失败 → 记日志 → 再写盘"的循环
const rawConsoleError = console.error.bind(console)

export interface LogEntry {
  timestamp: string
  level: 'DEBUG' | 'INFO' | 'WARN' | 'ERROR'
  category: string
  message: string
  data?: unknown
}

/** 增量拉取日志（proxy-get-logs-since）的返回结构 */
export interface LogsSinceResult {
  /** 本次返回的日志，按时间正序 */
  entries: LogEntry[]
  /** 当前最新一条日志的序号，客户端下次拉取时作为 afterSeq 传回 */
  lastSeq: number
  /** 日志存储代际号，每次清空日志后递增 */
  epoch: number
  /** 内存中保留的日志总条数 */
  total: number
  /** true：客户端应丢弃本地列表、用 entries 整体替换；false：把 entries 追加到本地列表末尾 */
  reset: boolean
}

export interface LoggerConfig {
  enabled: boolean
  logDir?: string
  maxFileSize?: number // 最大文件大小 (bytes)
  maxFiles?: number // 最大文件数量
  logToConsole?: boolean
}

const DEFAULT_CONFIG: LoggerConfig = {
  enabled: false,
  maxFileSize: 10 * 1024 * 1024, // 10MB
  maxFiles: 5,
  logToConsole: true
}

/** 单条日志消息、字符串数据的最大保留长度 */
const MAX_LOG_TEXT_LENGTH = 4000
/** 单条日志附带数据序列化后的最大长度，超出则改存截断后的 JSON 文本 */
const MAX_LOG_DATA_JSON_LENGTH = 16 * 1024

/** 超长文本截断，并注明被截掉的字符数 */
function truncateText(text: string, max: number): string {
  return text.length > max ? `${text.slice(0, max)}…[truncated ${text.length - max} chars]` : text
}

/**
 * 脱敏并截断日志文本。
 * 先脱敏再截断，避免截断点把 token 切成半截导致脱敏规则失配；超长文本只取前 2 倍上限参与脱敏，控制正则开销。
 * @param text 原始文本
 * @returns 脱敏且长度不超过上限（另加截断说明）的文本
 */
export function sanitizeLogText(text: string): string {
  if (text.length <= MAX_LOG_TEXT_LENGTH) return redactString(text)
  const head = redactString(text.slice(0, MAX_LOG_TEXT_LENGTH * 2)).slice(0, MAX_LOG_TEXT_LENGTH)
  return `${head}…[truncated ${text.length - MAX_LOG_TEXT_LENGTH} chars]`
}

/**
 * 规整日志附带数据：脱敏、限制体积，并生成与调用方对象无关的副本，避免日志长期持有调用方的大对象引用。
 * @param data 调用方传入的任意数据
 * @returns 可安全序列化、体积受限的数据（基础类型、脱敏后的普通对象，或截断后的 JSON 文本）
 */
export function sanitizeLogData(data: unknown): unknown {
  if (data === undefined || data === null) return data
  if (typeof data === 'string') return sanitizeLogText(data)
  const redacted = redactValue(data)
  if (redacted === null || typeof redacted !== 'object') return redacted
  try {
    const json = JSON.stringify(redacted)
    return json.length > MAX_LOG_DATA_JSON_LENGTH ? truncateText(json, MAX_LOG_DATA_JSON_LENGTH) : redacted
  } catch {
    return '[unserializable]'
  }
}

class ProxyLogger {
  private config: LoggerConfig
  private logStream: fs.WriteStream | null = null
  private currentLogFile: string = ''
  private currentFileSize: number = 0

  constructor() {
    this.config = { ...DEFAULT_CONFIG }
  }

  configure(config: Partial<LoggerConfig>): void {
    this.config = { ...this.config, ...config }

    if (this.config.enabled && !this.config.logDir) {
      // 默认日志目录
      this.config.logDir = path.join(app.getPath('userData'), 'logs', 'proxy')
    }

    if (this.config.enabled) {
      this.initLogFile()
    } else {
      this.close()
    }
  }

  private initLogFile(): void {
    if (!this.config.logDir) return

    try {
      // 确保目录存在
      fs.mkdirSync(this.config.logDir, { recursive: true })

      // 创建新的日志文件
      const timestamp = new Date().toISOString().replace(/[:.]/g, '-')
      this.currentLogFile = path.join(this.config.logDir, `proxy-${timestamp}.log`)
      this.logStream = fs.createWriteStream(this.currentLogFile, { flags: 'a' })
      this.currentFileSize = 0

      this.info('Logger', 'Log file initialized', { file: this.currentLogFile })
    } catch (error) {
      console.error('[ProxyLogger] Failed to init log file:', error)
    }
  }

  private rotateIfNeeded(): void {
    if (!this.config.maxFileSize || this.currentFileSize < this.config.maxFileSize) {
      return
    }

    this.close()
    this.cleanOldLogs()
    this.initLogFile()
  }

  private cleanOldLogs(): void {
    if (!this.config.logDir || !this.config.maxFiles) return

    try {
      const files = fs.readdirSync(this.config.logDir)
        .filter(f => f.startsWith('proxy-') && f.endsWith('.log'))
        .map(f => ({
          name: f,
          path: path.join(this.config.logDir!, f),
          time: fs.statSync(path.join(this.config.logDir!, f)).mtime.getTime()
        }))
        .sort((a, b) => b.time - a.time)

      // 删除超出数量限制的旧文件
      while (files.length >= this.config.maxFiles) {
        const oldest = files.pop()
        if (oldest) {
          fs.unlinkSync(oldest.path)
        }
      }
    } catch (error) {
      console.error('[ProxyLogger] Failed to clean old logs:', error)
    }
  }

  private isWriting = false
  private write(rawEntry: LogEntry): void {
    // 统一脱敏并限制体积：message + data 里的代理账密 / token / password 等，避免明文落盘或上屏
    const entry: LogEntry = {
      ...rawEntry,
      message: sanitizeLogText(rawEntry.message),
      data: rawEntry.data === undefined ? undefined : sanitizeLogData(rawEntry.data)
    }
    const line = JSON.stringify(entry) + '\n'

    if (this.config.logToConsole) {
      const prefix = `[${entry.level}][${entry.category}]`
      // 设置 flag 防止 console 拦截器重复写入 proxyLogStore
      this.isWriting = true
      if (entry.level === 'ERROR') {
        console.error(prefix, entry.message, entry.data || '')
      } else if (entry.level === 'WARN') {
        console.warn(prefix, entry.message, entry.data || '')
      } else {
        console.log(prefix, entry.message, entry.data || '')
      }
      this.isWriting = false
    }

    if (this.config.enabled && this.logStream) {
      this.logStream.write(line)
      this.currentFileSize += Buffer.byteLength(line)
      this.rotateIfNeeded()
    }

    // 同时添加到内存存储（用于 UI 显示）
    proxyLogStore.add(entry)
  }

  get _isWriting(): boolean { return this.isWriting }

  debug(category: string, message: string, data?: unknown): void {
    this.write({
      timestamp: new Date().toISOString(),
      level: 'DEBUG',
      category,
      message,
      data
    })
  }

  info(category: string, message: string, data?: unknown): void {
    this.write({
      timestamp: new Date().toISOString(),
      level: 'INFO',
      category,
      message,
      data
    })
  }

  warn(category: string, message: string, data?: unknown): void {
    this.write({
      timestamp: new Date().toISOString(),
      level: 'WARN',
      category,
      message,
      data
    })
  }

  error(category: string, message: string, data?: unknown): void {
    this.write({
      timestamp: new Date().toISOString(),
      level: 'ERROR',
      category,
      message,
      data
    })
  }

  // 记录请求
  request(info: {
    path: string
    method: string
    model?: string
    accountId?: string
  }): void {
    this.info('Request', `${info.method} ${info.path}`, info)
  }

  // 记录响应
  response(info: {
    path: string
    status: number
    tokens?: number
    responseTime?: number
    error?: string
  }): void {
    if (info.error) {
      this.error('Response', `${info.path} -> ${info.status}`, info)
    } else {
      this.info('Response', `${info.path} -> ${info.status}`, info)
    }
  }

  // 记录 Token 刷新
  tokenRefresh(accountId: string, success: boolean, error?: string): void {
    if (success) {
      this.info('TokenRefresh', `Account ${accountId} refreshed successfully`)
    } else {
      this.error('TokenRefresh', `Account ${accountId} refresh failed`, { error })
    }
  }

  close(): void {
    if (this.logStream) {
      this.logStream.end()
      this.logStream = null
    }
  }

  getLogDir(): string | undefined {
    return this.config.logDir
  }
}

/** 内存中最多保留的日志条数（5 万条 × 平均约 200 字节 ≈ 10 MB） */
const MAX_LOGS = 50000
/** 追加写盘间隔：期间产生的日志合并成一次追加写 */
const FLUSH_INTERVAL_MS = 5000
/** 单个日志文件的大小上限，超过后轮转为 .1 文件，磁盘上最多保留两份 */
const MAX_LOG_FILE_BYTES = 5 * 1024 * 1024

/** 校验从磁盘读回的日志条目结构，过滤损坏或旧版本遗留的无效数据 */
function isValidLogEntry(value: unknown): value is LogEntry {
  if (!value || typeof value !== 'object') return false
  const log = value as Partial<LogEntry>
  return typeof log.timestamp === 'string' && !isNaN(new Date(log.timestamp).getTime()) &&
    typeof log.level === 'string' && !!log.level &&
    typeof log.category === 'string' && !!log.category &&
    typeof log.message === 'string'
}

/** 把一批日志序列化为 NDJSON 文本；个别条目无法序列化时丢弃其 data 字段 */
function serializeBatch(batch: LogEntry[]): string {
  let out = ''
  for (const entry of batch) {
    let line: string
    try {
      line = JSON.stringify(entry)
    } catch {
      line = JSON.stringify({ ...entry, data: '[unserializable]' })
    }
    out += line + '\n'
  }
  return out
}

/** 固定容量的环形缓冲区：写入 O(1)，写满后覆盖最旧的条目 */
class LogRingBuffer {
  private items: Array<LogEntry | undefined>
  /** 最旧条目所在下标 */
  private head = 0
  private length = 0

  constructor(private readonly capacity: number) {
    this.items = new Array(capacity)
  }

  /** 当前保存的条目数 */
  get size(): number {
    return this.length
  }

  /** 追加一条日志，已满时覆盖最旧的一条 */
  push(entry: LogEntry): void {
    if (this.length < this.capacity) {
      this.items[(this.head + this.length) % this.capacity] = entry
      this.length++
    } else {
      this.items[this.head] = entry
      this.head = (this.head + 1) % this.capacity
    }
  }

  /** 按时间正序返回最新的 count 条 */
  last(count: number): LogEntry[] {
    const n = Math.max(0, Math.min(count, this.length))
    const out: LogEntry[] = new Array(n)
    const start = this.head + this.length - n
    for (let i = 0; i < n; i++) {
      out[i] = this.items[(start + i) % this.capacity] as LogEntry
    }
    return out
  }

  /** 清空全部条目 */
  clear(): void {
    this.items = new Array(this.capacity)
    this.head = 0
    this.length = 0
  }
}

// 内存日志存储（用于 UI 显示）
//
// 性能设计：
// 1. 内存用固定容量环形缓冲区，写满后覆盖最旧条目，单条写入 O(1)
//    （旧实现到上限后每条日志都要复制整个 5 万元素数组）
// 2. 落盘改为追加写 NDJSON：每 FLUSH_INTERVAL_MS 把新增日志合并成一次 appendFile，
//    单文件超过 MAX_LOG_FILE_BYTES 轮转为 .1，磁盘上最多两份
//    （旧实现每 30 秒把整个数组同步 JSON.stringify 后整文件重写，满载约 10 MB，会卡住主进程）
// 3. 每条日志带单调递增序号，UI 通过 getSince 只拉取新增部分，清空日志时 epoch 递增通知 UI 整体重载
// 4. 所有文件操作串行执行，避免追加、轮转、清空互相穿插；应用退出时通过 flushSaveNow() 强制落盘
class ProxyLogStore {
  private ring = new LogRingBuffer(MAX_LOGS)
  /** 下一条日志的序号（从 1 开始单调递增，清空日志也不重置） */
  private nextSeq = 1
  /** 存储代际号，每次清空日志后递增 */
  private epoch = 0
  private listeners: ((entry: LogEntry) => void)[] = []

  private initialized = false
  private filePath = ''
  private rotatedFilePath = ''
  /** 旧版整文件 JSON 日志路径（迁移后删除） */
  private legacyFilePath = ''
  private legacyFilePending = false
  private currentFileBytes = 0
  /** 等待追加写盘的日志 */
  private pending: LogEntry[] = []
  private flushTimer: NodeJS.Timeout | null = null
  /** 串行化所有文件操作 */
  private ioQueue: Promise<void> = Promise.resolve()

  /**
   * 初始化存储位置并载入历史日志（同步执行，只在启动时调用一次）。
   * @param userDataPath 应用 userData 目录
   */
  initialize(userDataPath: string): void {
    if (this.initialized) return
    this.initialized = true
    this.filePath = path.join(userDataPath, 'proxy-logs.ndjson')
    this.rotatedFilePath = path.join(userDataPath, 'proxy-logs.1.ndjson')
    this.legacyFilePath = path.join(userDataPath, 'proxy-logs.json')

    const { entries: history, migrated } = this.loadHistory()
    // 初始化前已产生的日志（如有）排在磁盘历史之后
    const early = this.ring.last(this.ring.size)
    this.ring.clear()
    for (const entry of history) this.ring.push(entry)
    for (const entry of early) this.ring.push(entry)
    this.nextSeq = 1 + history.length + early.length
    // 从旧格式迁移时，把历史日志一并写入新文件
    if (migrated) this.pending = history.concat(this.pending)
    console.log(`[ProxyLogStore] Loaded ${history.length} logs${migrated ? ' (migrated from proxy-logs.json)' : ''}`)
    if (this.pending.length > 0) this.scheduleFlush()
  }

  /** 读取磁盘上的历史日志：优先新格式 NDJSON，其次迁移旧版 proxy-logs.json */
  private loadHistory(): { entries: LogEntry[]; migrated: boolean } {
    if (fs.existsSync(this.filePath) || fs.existsSync(this.rotatedFilePath)) {
      const entries = this.readNdjson(this.rotatedFilePath).concat(this.readNdjson(this.filePath))
      try {
        this.currentFileBytes = fs.statSync(this.filePath).size
      } catch {
        this.currentFileBytes = 0
      }
      // 新格式已存在时旧文件只是残留，下次写盘后顺带删除
      this.legacyFilePending = fs.existsSync(this.legacyFilePath)
      return { entries: entries.length > MAX_LOGS ? entries.slice(-MAX_LOGS) : entries, migrated: false }
    }
    if (fs.existsSync(this.legacyFilePath)) {
      try {
        const parsed = JSON.parse(fs.readFileSync(this.legacyFilePath, 'utf-8'))
        const entries = Array.isArray(parsed) ? parsed.filter(isValidLogEntry) : []
        this.legacyFilePending = true
        return { entries: entries.length > MAX_LOGS ? entries.slice(-MAX_LOGS) : entries, migrated: true }
      } catch (error) {
        // 旧文件损坏时保留原文件不动，从空日志开始
        rawConsoleError('[ProxyLogStore] Failed to load legacy logs:', error)
      }
    }
    return { entries: [], migrated: false }
  }

  /** 逐行解析 NDJSON 日志文件，跳过崩溃时可能残留的半行或损坏行 */
  private readNdjson(file: string): LogEntry[] {
    let text: string
    try {
      text = fs.readFileSync(file, 'utf-8')
    } catch {
      return []
    }
    const entries: LogEntry[] = []
    for (const line of text.split('\n')) {
      if (!line) continue
      try {
        const entry = JSON.parse(line)
        if (isValidLogEntry(entry)) entries.push(entry)
      } catch {
        // 跳过损坏行
      }
    }
    return entries
  }

  /** 在文件操作队列末尾追加任务，保证追加、轮转、清空按调用顺序串行执行 */
  private enqueueIo(task: () => Promise<void>): Promise<void> {
    const run = this.ioQueue.then(task).catch((error) => {
      rawConsoleError('[ProxyLogStore] Log file operation failed:', error)
    })
    this.ioQueue = run
    return run
  }

  private scheduleFlush(): void {
    if (!this.initialized || this.flushTimer) return
    this.flushTimer = setTimeout(() => {
      this.flushTimer = null
      void this.flush()
    }, FLUSH_INTERVAL_MS)
  }

  /** 把待写日志以 NDJSON 追加到当前文件末尾（异步、串行） */
  private flush(): Promise<void> {
    if (!this.initialized) return Promise.resolve()
    return this.enqueueIo(async () => {
      if (this.pending.length === 0) return
      const batch = this.pending
      this.pending = []
      const chunk = serializeBatch(batch)
      const bytes = Buffer.byteLength(chunk)
      try {
        if (this.currentFileBytes > 0 && this.currentFileBytes + bytes > MAX_LOG_FILE_BYTES) {
          await this.rotate()
        }
        await fs.promises.appendFile(this.filePath, chunk, 'utf-8')
        this.currentFileBytes += bytes
      } catch (error) {
        // 写盘失败：放回队首等待下次重试，最多保留 MAX_LOGS 条，避免磁盘持续异常时无限堆积
        const merged = batch.concat(this.pending)
        this.pending = merged.length > MAX_LOGS ? merged.slice(-MAX_LOGS) : merged
        rawConsoleError('[ProxyLogStore] Failed to append logs:', error)
        this.scheduleFlush()
        return
      }
      if (this.legacyFilePending) {
        this.legacyFilePending = false
        await fs.promises.rm(this.legacyFilePath, { force: true })
      }
    })
  }

  /** 当前文件写满后轮转：current → .1（覆盖旧的 .1），之后从空文件继续追加 */
  private async rotate(): Promise<void> {
    try {
      await fs.promises.rename(this.filePath, this.rotatedFilePath)
    } catch {
      // 重命名失败（如 .1 被其他程序占用）时直接截断当前文件，保证磁盘占用有上限
      await fs.promises.writeFile(this.filePath, '', 'utf-8').catch(() => undefined)
    }
    this.currentFileBytes = 0
  }

  /**
   * 强制立即把待写日志落盘（应用退出时调用）。
   * @returns 写盘完成（或失败已记录）后 resolve
   */
  async flushSaveNow(): Promise<void> {
    if (this.flushTimer) {
      clearTimeout(this.flushTimer)
      this.flushTimer = null
    }
    await this.flush()
  }

  /**
   * 追加一条日志：写入内存、通知监听器，并安排一次批量追加写盘。
   * @param entry 已脱敏的日志条目
   */
  add(entry: LogEntry): void {
    this.ring.push(entry)
    this.nextSeq++
    this.pending.push(entry)
    // 仅在写盘持续失败时才会触发，按 1.2 倍阈值批量裁剪，避免每条日志都复制数组
    if (this.pending.length > MAX_LOGS * 1.2) {
      this.pending = this.pending.slice(-MAX_LOGS)
    }

    // 通知监听器（异常隔离）
    for (const listener of this.listeners) {
      try { listener(entry) } catch { /* ignore */ }
    }

    this.scheduleFlush()
  }

  /**
   * 获取内存中保留的全部日志。
   * @returns 按时间正序排列的日志副本
   */
  getAll(): LogEntry[] {
    return this.ring.last(this.ring.size)
  }

  /**
   * 获取最新的若干条日志。
   * @param count 需要的条数
   * @returns 按时间正序排列的日志
   */
  getLast(count: number): LogEntry[] {
    return this.ring.last(count)
  }

  /**
   * 增量获取日志：只返回序号大于 afterSeq 的部分。
   * 客户端首次拉取、epoch 不一致、落后超过内存保留范围或新增条数超过 limit 时，
   * 返回最近 limit 条并置 reset=true，由客户端整体替换。
   * @param afterSeq 客户端已持有的最新序号，首次拉取传 0
   * @param epoch 客户端持有的代际号，首次拉取传 -1
   * @param limit 客户端最多保留的条数，0 表示不限制
   * @returns 增量结果
   */
  getSince(afterSeq: number, epoch: number, limit = 0): LogsSinceResult {
    const total = this.ring.size
    const lastSeq = this.nextSeq - 1
    const oldestSeq = lastSeq - total + 1
    const max = limit > 0 ? Math.min(limit, total) : total
    let reset = epoch !== this.epoch || afterSeq <= 0 || afterSeq > lastSeq || afterSeq < oldestSeq - 1
    let count = reset ? max : lastSeq - afterSeq
    if (!reset && count > max) {
      reset = true
      count = max
    }
    return { entries: this.ring.last(count), lastSeq, epoch: this.epoch, total, reset }
  }

  /** 清空内存与磁盘上的日志，并递增 epoch 通知增量拉取的客户端整体重载 */
  clear(): void {
    this.ring.clear()
    this.pending = []
    this.epoch++
    if (this.flushTimer) {
      clearTimeout(this.flushTimer)
      this.flushTimer = null
    }
    if (!this.initialized) return
    void this.enqueueIo(async () => {
      await Promise.all(
        [this.filePath, this.rotatedFilePath, this.legacyFilePath].map((file) => fs.promises.rm(file, { force: true }))
      )
      this.legacyFilePending = false
      this.currentFileBytes = 0
    })
  }

  /**
   * 当前内存中保留的日志条数。
   * @returns 条数
   */
  count(): number {
    return this.ring.size
  }

  /**
   * 订阅新增日志。
   * @param listener 每条新日志写入后调用
   * @returns 取消订阅函数
   */
  onLog(listener: (entry: LogEntry) => void): () => void {
    this.listeners.push(listener)
    return () => {
      const index = this.listeners.indexOf(listener)
      if (index >= 0) {
        this.listeners.splice(index, 1)
      }
    }
  }
}

export const proxyLogStore = new ProxyLogStore()

// 单例导出
export const proxyLogger = new ProxyLogger()

// 拦截主进程 console 输出，自动转发到 proxyLogStore
// 这样所有 console.log/warn/error 都能在日志页面显示
let consoleIntercepted = false
export function interceptConsole(): void {
  if (consoleIntercepted) return
  consoleIntercepted = true

  const originalLog = console.log
  const originalWarn = console.warn
  const originalError = console.error

  const parseConsoleCategory = (args: unknown[]): { category: string; message: string } => {
    const first = String(args[0] || '')
    // 匹配 [Category] 或 [INFO][Category] 格式
    const bracketMatch = first.match(/^\[(?:DEBUG|INFO|WARN|ERROR)\]?\[?([^\]]*)\]?\s*(.*)/)
    if (bracketMatch) {
      return { category: bracketMatch[1] || 'App', message: bracketMatch[2] || '' }
    }
    const simpleMatch = first.match(/^\[([^\]]+)\]\s*(.*)/)
    if (simpleMatch) {
      return { category: simpleMatch[1], message: simpleMatch[2] || '' }
    }
    return { category: 'App', message: first }
  }

  const buildEntry = (args: unknown[], level: 'INFO' | 'WARN' | 'ERROR'): LogEntry => {
    const { category, message } = parseConsoleCategory(args)
    const rest = args.slice(1)
    // data: 后续参数（对象/数组保留结构，字符串拼接）
    let data: unknown = undefined
    if (rest.length === 1) {
      data = rest[0]
    } else if (rest.length > 1) {
      // 如果全是字符串，拼成一个；否则保持数组
      const allStrings = rest.every(r => typeof r === 'string')
      data = allStrings ? (rest as string[]).join(' ') : rest
    }
    // 与 ProxyLogger 一致：脱敏 + 限制体积，且不再直接引用调用方传入的对象
    return {
      timestamp: new Date().toISOString(),
      level,
      category,
      message: sanitizeLogText(message),
      data: sanitizeLogData(data)
    }
  }

  console.log = (...args: unknown[]) => {
    originalLog.apply(console, args)
    if (proxyLogger._isWriting) return
    proxyLogStore.add(buildEntry(args, 'INFO'))
  }

  console.warn = (...args: unknown[]) => {
    originalWarn.apply(console, args)
    if (proxyLogger._isWriting) return
    proxyLogStore.add(buildEntry(args, 'WARN'))
  }

  console.error = (...args: unknown[]) => {
    originalError.apply(console, args)
    if (proxyLogger._isWriting) return
    proxyLogStore.add(buildEntry(args, 'ERROR'))
  }
}
