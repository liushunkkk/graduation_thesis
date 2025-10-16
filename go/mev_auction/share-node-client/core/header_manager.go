package core

import (
	"share-node-client/client"
	"sync/atomic"
	"time"
)

// BlockNumberManager 负责维护全局的 block number
type BlockNumberManager struct {
	current uint64
}

// GlobalBlockManager 全局实例
var GlobalBlockManager = &BlockNumberManager{current: 100}

// StartAutoIncrement 每隔3秒自动自增1
func (m *BlockNumberManager) StartAutoIncrement() {
	go func() {
		ticker := time.NewTicker(3 * time.Second)
		defer ticker.Stop()
		for range ticker.C {
			atomic.AddUint64(&m.current, 1)
			go client.GlobalRpcClient.ResetHeader(atomic.LoadUint64(&m.current))
		}
	}()
}

// GetCurrentBlock 返回当前区块号
func (m *BlockNumberManager) GetCurrentBlock() uint64 {
	return atomic.LoadUint64(&m.current)
}

// SetCurrentBlock 手动设置（例如初始化时）
func (m *BlockNumberManager) SetCurrentBlock(n uint64) {
	atomic.StoreUint64(&m.current, n)
}
