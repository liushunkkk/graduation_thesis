package model

import (
	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/common/hexutil"
)

const (
	HintHash             = "hash"
	HintFrom             = "from"
	HintTo               = "to"
	HintValue            = "value"
	HintNonce            = "nonce"
	HintCallData         = "calldata"
	HintFunctionSelector = "functionSelector"
	HintGasLimit         = "gasLimit"
	HintGasPrice         = "gasPrice"
	HintLogs             = "logs"
)

func GetAllHints() map[string]bool {
	return map[string]bool{
		HintHash:             true,
		HintFrom:             true,
		HintTo:               true,
		HintValue:            true,
		HintNonce:            true,
		HintCallData:         true,
		HintFunctionSelector: true,
		HintGasLimit:         true,
		HintGasPrice:         true,
		HintLogs:             true,
	}
}

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
	UserId         int
	MaxBlockNumber uint64
}

type SendRawTransactionResponse struct {
	TxHash common.Hash `json:"txHash"`
}

type ResetHeaderArgs struct {
	HeaderNumber uint64
	Time         uint64
}

type ResetHeaderResponse struct {
	HeaderNumber uint64 `json:"headerNumber"`
}
