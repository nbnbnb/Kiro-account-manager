import { useCallback, useEffect, useRef, useState } from 'react'

/** 主进程日志条目 */
export interface ProxyLogEntry {
  timestamp: string
  level: 'DEBUG' | 'INFO' | 'WARN' | 'ERROR'
  category: string
  message: string
  data?: unknown
}

/** useProxyLogs 的参数 */
export interface UseProxyLogsOptions {
  /** 是否启用轮询（如对话框关闭时传 false） */
  enabled: boolean
  /** 本地最多保留的条数，0 表示不限制（以主进程内存中保留的条数为上限） */
  limit: number
  /** 轮询间隔（毫秒），默认 1500 */
  intervalMs?: number
  /** 追加了新日志时回调，参数为本次新增条数；整体重载时不触发 */
  onAppend?: (count: number) => void
}

/** useProxyLogs 的返回值 */
export interface UseProxyLogsResult {
  /** 按时间正序排列的日志 */
  logs: ProxyLogEntry[]
  /** 主进程内存中保留的日志总条数 */
  total: number
  /** 是否正在进行首次（或重载）拉取 */
  loading: boolean
  /** 丢弃本地列表，重新拉取最近 limit 条 */
  reload: () => Promise<void>
}

/**
 * 增量拉取主进程日志。
 * 首次拉取最近 limit 条，之后每次轮询只取新增部分；主进程清空日志或本地落后过多时自动整体替换。
 * 没有新日志时不更新状态，不会触发重渲染。所有请求串行执行，避免并发请求重复追加。
 * @param options 拉取参数
 * @returns 日志列表与控制方法
 */
export function useProxyLogs({
  enabled,
  limit,
  intervalMs = 1500,
  onAppend
}: UseProxyLogsOptions): UseProxyLogsResult {
  const [logs, setLogs] = useState<ProxyLogEntry[]>([])
  const [total, setTotal] = useState(0)
  const [loading, setLoading] = useState(false)
  // 本地游标：已持有的最新序号与日志代际号
  const cursorRef = useRef({ seq: 0, epoch: -1 })
  // 请求串行队列，以及"已有一次轮询在排队"标记（轮询最多排一个，重载总是排队）
  const queueRef = useRef<Promise<void>>(Promise.resolve())
  const pollQueuedRef = useRef(false)
  const onAppendRef = useRef(onAppend)

  useEffect(() => {
    onAppendRef.current = onAppend
  }, [onAppend])

  const fetchOnce = useCallback(
    async (forceReset: boolean): Promise<void> => {
      const cursor = forceReset ? { seq: 0, epoch: -1 } : cursorRef.current
      try {
        const result = await window.api.proxyGetLogsSince(cursor.seq, cursor.epoch, limit)
        cursorRef.current = { seq: result.lastSeq, epoch: result.epoch }
        const entries = result.entries as ProxyLogEntry[]
        if (result.reset) {
          setLogs(entries)
        } else if (entries.length > 0) {
          setLogs((prev) => {
            const merged = prev.concat(entries)
            return limit > 0 && merged.length > limit ? merged.slice(merged.length - limit) : merged
          })
          onAppendRef.current?.(entries.length)
        }
        setTotal(result.total)
      } catch {
        // 拉取失败时保留现有列表，下次轮询重试
      }
    },
    [limit]
  )

  const enqueue = useCallback(
    (forceReset: boolean): Promise<void> => {
      if (!forceReset) {
        if (pollQueuedRef.current) return queueRef.current
        pollQueuedRef.current = true
      }
      const run = queueRef.current.then(() => {
        if (!forceReset) pollQueuedRef.current = false
        return fetchOnce(forceReset)
      })
      queueRef.current = run
      return run
    },
    [fetchOnce]
  )

  useEffect(() => {
    if (!enabled) return
    let active = true
    setLoading(true)
    void enqueue(true).finally(() => {
      if (active) setLoading(false)
    })
    const timer = setInterval(() => {
      void enqueue(false)
    }, intervalMs)
    return () => {
      active = false
      clearInterval(timer)
    }
  }, [enabled, enqueue, intervalMs])

  const reload = useCallback(() => enqueue(true), [enqueue])

  return { logs, total, loading, reload }
}
