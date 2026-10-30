import './styles/globals.css'

import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { MotionConfig } from 'framer-motion'
import App from './App'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    {/* 跟随系统"减少动态效果"设置：开启时 framer-motion 跳过位移/布局动画，只保留透明度等过渡 */}
    <MotionConfig reducedMotion="user">
      <App />
    </MotionConfig>
  </StrictMode>
)
