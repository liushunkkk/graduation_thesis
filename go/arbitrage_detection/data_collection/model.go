package data_collection

import (
	"encoding/json"
	"time"
)

// ArbitraryTransaction 套利数据信息
type ArbitraryTransaction struct {
	ID         int       `gorm:"primaryKey;autoIncrement"`
	Searcher   string    `gorm:"type:char(42);default:null"`
	Builder    string    `gorm:"type:varchar(255);default:null"`
	From       string    `gorm:"column:from;type:char(42);default:null"` // `from` 是 SQL 保留字
	To         string    `gorm:"type:char(42);default:null"`
	BlockNum   int64     `gorm:"type:bigint;default:null"`
	TxHash     string    `gorm:"type:char(66);uniqueIndex;default:null"`
	TimeStamp  time.Time `gorm:"type:datetime;default:null"`
	MevType    string    `gorm:"type:varchar(255);default:null"`
	Position   int64     `gorm:"type:bigint;default:null"`
	BribeValue string    `gorm:"type:text"`
	Bribee     string    `gorm:"type:varchar(255);default:null"`
	BribeType  string    `gorm:"type:varchar(255);default:null"`
	ArbProfit  float64   `gorm:"type:double;default:null"`
}

func (ArbitraryTransaction) TableName() string {
	return Table_ArbitraryTransaction
}

// EthereumTransaction 交易详细信息
type EthereumTransaction struct {
	ID          uint   `gorm:"primaryKey;autoIncrement"`
	TxType      string `gorm:"type:char(20);not null"` // LegacyTx, AccessListTx, DynamicFeeTx
	Nonce       uint64 `gorm:"not null"`
	TxHash      string `gorm:"type:char(128);not null"`        // common.Hash.Hex()
	BlockNumber string `gorm:"type:varchar(128);default:null"` // big.Int.String()

	// Fee fields
	GasPrice  string `gorm:"type:varchar(128);default:null"` // LegacyTx 和 AccessListTx
	GasTipCap string `gorm:"type:varchar(128);default:null"` // DynamicFeeTx (maxPriorityFeePerGas)
	GasFeeCap string `gorm:"type:varchar(128);default:null"` // DynamicFeeTx (maxFeePerGas)
	Gas       uint64 `gorm:"not null"`

	// Transaction details
	To         string `gorm:"type:char(128);default:null"` // nil 表示创建合约
	Value      string `gorm:"type:varchar(128);default:null"`
	Data       string `gorm:"type:longtext"` // 十六进制字符串表示 input data
	AccessList string `gorm:"type:longtext"` // JSON 字符串形式

	// 签名字段
	V string `gorm:"type:varchar(128)"`
	R string `gorm:"type:varchar(128)"`
	S string `gorm:"type:varchar(128)"`

	OriginJsonString string `gorm:"type:longtext"` // string(tx.MarshalJSON())
}

func (t *EthereumTransaction) TableName() string {
	return Table_EthereumTransactions
}

func (t *EthereumTransaction) String() string {
	marshal, err := json.Marshal(t)
	if err != nil {
		return ""
	}
	return string(marshal)
}

// EthereumReceipt 交易收据信息
type EthereumReceipt struct {
	ID                uint   `gorm:"primaryKey;autoIncrement"`
	TxType            string `gorm:"type:char(20);not null"` // LegacyTx, AccessListTx, DynamicFeeTx
	PostState         string `gorm:"type:longtext"`          // hex.EncodeToString([]byte)
	Status            string `gorm:"type:varchar(20);default:null"`
	CumulativeGasUsed uint64
	Bloom             string `gorm:"type:longtext"`               // hex.EncodeToString([256]byte)
	Logs              string `gorm:"type:longtext"`               // JSON 字符串：[]*Log
	TxHash            string `gorm:"type:char(128);not null"`     // common.Hash.Hex()
	ContractAddress   string `gorm:"type:char(128);default:null"` // common.Address.Hex()
	GasUsed           uint64
	EffectiveGasPrice string `gorm:"type:varchar(128);default:null"` // big.Int.String()
	BlobGasUsed       uint64
	BlobGasPrice      string `gorm:"type:varchar(128);default:null"` // big.Int.String()
	BlockHash         string `gorm:"type:char(128);default:null"`    // common.Hash.Hex()
	BlockNumber       string `gorm:"type:varchar(128);default:null"` // big.Int.String()
	TransactionIndex  uint
	OriginJsonString  string `gorm:"type:longtext"` // string(receipt.MarshalJSON())
}

func (r *EthereumReceipt) TableName() string {
	return Table_EthereumReceipts
}

func (r *EthereumReceipt) String() string {
	marshal, err := json.Marshal(r)
	if err != nil {
		return ""
	}
	return string(marshal)
}

// ComparisonTransaction 交易详细信息
type ComparisonTransaction struct {
	ID          uint   `gorm:"primaryKey;autoIncrement"`
	TxType      string `gorm:"type:char(20);not null"` // LegacyTx, AccessListTx, DynamicFeeTx
	Nonce       uint64 `gorm:"not null"`
	TxHash      string `gorm:"type:char(128);not null"`        // common.Hash.Hex()
	BlockNumber string `gorm:"type:varchar(128);default:null"` // big.Int.String()

	// Fee fields
	GasPrice  string `gorm:"type:varchar(128);default:null"` // LegacyTx 和 AccessListTx
	GasTipCap string `gorm:"type:varchar(128);default:null"` // DynamicFeeTx (maxPriorityFeePerGas)
	GasFeeCap string `gorm:"type:varchar(128);default:null"` // DynamicFeeTx (maxFeePerGas)
	Gas       uint64 `gorm:"not null"`

	// Transaction details
	To         string `gorm:"type:char(128);default:null"` // nil 表示创建合约
	Value      string `gorm:"type:varchar(128);default:null"`
	Data       string `gorm:"type:longtext"` // 十六进制字符串表示 input data
	AccessList string `gorm:"type:longtext"` // JSON 字符串形式

	// 签名字段
	V string `gorm:"type:varchar(128)"`
	R string `gorm:"type:varchar(128)"`
	S string `gorm:"type:varchar(128)"`

	OriginJsonString string `gorm:"type:longtext"` // string(tx.MarshalJSON())
}

func (t *ComparisonTransaction) TableName() string {
	return Table_ComparisonTransactions
}

func (t *ComparisonTransaction) String() string {
	marshal, err := json.Marshal(t)
	if err != nil {
		return ""
	}
	return string(marshal)
}

// ComparisonReceipt 交易收据信息
type ComparisonReceipt struct {
	ID                uint   `gorm:"primaryKey;autoIncrement"`
	TxType            string `gorm:"type:char(20);not null"` // LegacyTx, AccessListTx, DynamicFeeTx
	PostState         string `gorm:"type:longtext"`          // hex.EncodeToString([]byte)
	Status            string `gorm:"type:varchar(20);default:null"`
	CumulativeGasUsed uint64
	Bloom             string `gorm:"type:longtext"`               // hex.EncodeToString([256]byte)
	Logs              string `gorm:"type:longtext"`               // JSON 字符串：[]*Log
	TxHash            string `gorm:"type:char(128);not null"`     // common.Hash.Hex()
	ContractAddress   string `gorm:"type:char(128);default:null"` // common.Address.Hex()
	GasUsed           uint64
	EffectiveGasPrice string `gorm:"type:varchar(128);default:null"` // big.Int.String()
	BlobGasUsed       uint64
	BlobGasPrice      string `gorm:"type:varchar(128);default:null"` // big.Int.String()
	BlockHash         string `gorm:"type:char(128);default:null"`    // common.Hash.Hex()
	BlockNumber       string `gorm:"type:varchar(128);default:null"` // big.Int.String()
	TransactionIndex  uint
	OriginJsonString  string `gorm:"type:longtext"` // string(receipt.MarshalJSON())
}

func (r *ComparisonReceipt) TableName() string {
	return Table_ComparisonReceipts
}

func (r *ComparisonReceipt) String() string {
	marshal, err := json.Marshal(r)
	if err != nil {
		return ""
	}
	return string(marshal)
}
