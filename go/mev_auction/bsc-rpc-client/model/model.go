package model

import (
	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/common/hexutil"
)

type SendMevBundleArgs struct {
	Hash              common.Hash
	Txs               []hexutil.Bytes
	RevertingTxHashes []common.Hash
	MaxBlockNumber    uint64
	Hint              map[string]bool
	RefundAddress     common.Address
	RefundPercent     int
}

type SendMevBundleResponse struct {
	BundleHash common.Hash `json:"bundleHash"`
}

type SendRawTransactionArgs struct {
	Input          hexutil.Bytes
	MaxBlockNumber uint64
}

type SendRawTransactionResponse struct {
	TxHash common.Hash `json:"txHash"`
}

type ResetHeaderArgs struct {
	HeaderNumber uint64
}

type ResetHeaderResponse struct {
	HeaderNumber uint64 `json:"headerNumber"`
}
