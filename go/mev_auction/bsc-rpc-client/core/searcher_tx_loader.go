package core

import (
	"bsc-rpc-client/zap_logger"
	"fmt"
	"sync/atomic"
)

// EthereumTransaction 只保留必要字段
type EthereumTransaction struct {
	ID               uint   `gorm:"primaryKey;autoIncrement"`
	TxHash           string `gorm:"column:tx_hash"`
	OriginJsonString string `gorm:"column:origin_json_string"`
}

// STxLoader 负责加载并管理交易数据
type STxLoader struct {
	txs    []EthereumTransaction // 缓存的交易数据
	index  uint64                // 当前读取位置
	loaded bool                  // 是否已加载
}

// SearcherTxLoader 全局实例
var SearcherTxLoader = &STxLoader{}

// LoadFromDB 从数据库中加载所有交易（只取必要字段）
func (l *STxLoader) LoadFromDB(limit int) error {
	if l.loaded {
		return fmt.Errorf("交易数据已加载，禁止重复加载")
	}

	var txs []EthereumTransaction
	query := DB.Table("ethereum_transactions").Select("id", "tx_hash", "origin_json_string")
	if limit > 0 {
		query = query.Limit(limit)
	}

	if err := query.Find(&txs).Error; err != nil {
		return fmt.Errorf("加载交易失败: %w", err)
	}

	if len(txs) == 0 {
		return fmt.Errorf("数据库中没有交易数据")
	}

	l.txs = txs
	l.index = 0
	l.loaded = true
	zap_logger.Zap.Info(fmt.Sprintf("已加载 %d 条交易到内存", len(l.txs)))
	return nil
}

// Next 返回下一条交易（线程安全）
func (l *STxLoader) Next() *EthereumTransaction {
	if !l.loaded {
		zap_logger.Zap.Info(fmt.Sprintf("交易数据未加载，请先调用 LoadFromDB()"))
		return nil
	}

	i := atomic.AddUint64(&l.index, 1) - 1
	if int(i) >= len(l.txs) {
		l.ResetIndex()
		return l.Next() // 读完
	}
	return &l.txs[i]
}

// ResetIndex 重置指针
func (l *STxLoader) ResetIndex() {
	atomic.StoreUint64(&l.index, 0)
}
