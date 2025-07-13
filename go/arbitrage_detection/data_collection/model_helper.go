package data_collection

import (
	"encoding/hex"
	"encoding/json"
	"errors"
	"github.com/ethereum/go-ethereum/common/hexutil"
	"github.com/ethereum/go-ethereum/core/types"
)

// ConvertToEthereumTransaction 转换交易函数，将链上数据类型转换为数据库类型
func ConvertToEthereumTransaction(tx *types.Transaction) (*EthereumTransaction, error) {
	ethTx := &EthereumTransaction{}

	ethTx.TxHash = tx.Hash().Hex()
	ethTx.TxType = uintToStringType(tx.Type())

	if ethTx.TxType == OtherTx {
		return nil, errors.New("unknown transaction type")
	}

	jsonBytes, err := tx.MarshalJSON()
	if err != nil {
		return nil, err
	}
	ethTx.OriginJsonString = string(jsonBytes)

	// 根据交易类型来转换
	switch tx.Type() {
	case types.LegacyTxType:
		ethTx.Nonce = tx.Nonce()
		ethTx.GasPrice = tx.GasPrice().String()
		ethTx.Gas = tx.Gas()
		if tx.To() != nil {
			ethTx.To = tx.To().Hex()
		} else {
			return nil, errors.New("unknown transaction to")
		}
		ethTx.Value = tx.Value().String()
		ethTx.Data = hexutil.Encode(tx.Data())
		v, r, s := tx.RawSignatureValues()
		ethTx.V = v.String()
		ethTx.R = r.String()
		ethTx.S = s.String()
	case types.AccessListTxType:
		ethTx.Nonce = tx.Nonce()
		ethTx.GasPrice = tx.GasPrice().String()
		ethTx.Gas = tx.Gas()
		if tx.To() != nil {
			ethTx.To = tx.To().Hex()
		} else {
			return nil, errors.New("unknown transaction to")
		}
		ethTx.Value = tx.Value().String()
		ethTx.Data = hexutil.Encode(tx.Data())
		jsonString, err := json.Marshal(tx.AccessList())
		if err == nil {
			ethTx.AccessList = string(jsonString)
		}
		v, r, s := tx.RawSignatureValues()
		ethTx.V = v.String()
		ethTx.R = r.String()
		ethTx.S = s.String()
	case types.DynamicFeeTxType:
		ethTx.Nonce = tx.Nonce()
		ethTx.GasTipCap = tx.GasTipCap().String()
		ethTx.GasFeeCap = tx.GasFeeCap().String()
		ethTx.Gas = tx.Gas()
		if tx.To() != nil {
			ethTx.To = tx.To().Hex()
		} else {
			return nil, errors.New("unknown transaction to")
		}
		ethTx.Value = tx.Value().String()
		ethTx.Data = hexutil.Encode(tx.Data())
		jsonString, err := json.Marshal(tx.AccessList())
		if err == nil {
			ethTx.AccessList = string(jsonString)
		}
		v, r, s := tx.RawSignatureValues()
		ethTx.V = v.String()
		ethTx.R = r.String()
		ethTx.S = s.String()
	}
	return ethTx, nil
}

// ConvertToComparisonTransaction 转换对比交易函数，将链上数据类型转换为数据库类型
func ConvertToComparisonTransaction(tx *types.Transaction) (*ComparisonTransaction, error) {
	ethTx := &ComparisonTransaction{}

	ethTx.TxHash = tx.Hash().Hex()
	ethTx.TxType = uintToStringType(tx.Type())

	if ethTx.TxType == OtherTx {
		return nil, errors.New("unknown transaction type")
	}

	jsonBytes, err := tx.MarshalJSON()
	if err != nil {
		return nil, err
	}
	ethTx.OriginJsonString = string(jsonBytes)

	// 根据交易类型来转换
	switch tx.Type() {
	case types.LegacyTxType:
		ethTx.Nonce = tx.Nonce()
		ethTx.GasPrice = tx.GasPrice().String()
		ethTx.Gas = tx.Gas()
		if tx.To() != nil {
			ethTx.To = tx.To().Hex()
		} else {
			return nil, errors.New("unknown transaction to")
		}
		ethTx.Value = tx.Value().String()
		ethTx.Data = hexutil.Encode(tx.Data())
		v, r, s := tx.RawSignatureValues()
		ethTx.V = v.String()
		ethTx.R = r.String()
		ethTx.S = s.String()
	case types.AccessListTxType:
		ethTx.Nonce = tx.Nonce()
		ethTx.GasPrice = tx.GasPrice().String()
		ethTx.Gas = tx.Gas()
		if tx.To() != nil {
			ethTx.To = tx.To().Hex()
		} else {
			return nil, errors.New("unknown transaction to")
		}
		ethTx.Value = tx.Value().String()
		ethTx.Data = hexutil.Encode(tx.Data())
		jsonString, err := json.Marshal(tx.AccessList())
		if err == nil {
			ethTx.AccessList = string(jsonString)
		}
		v, r, s := tx.RawSignatureValues()
		ethTx.V = v.String()
		ethTx.R = r.String()
		ethTx.S = s.String()
	case types.DynamicFeeTxType:
		ethTx.Nonce = tx.Nonce()
		ethTx.GasTipCap = tx.GasTipCap().String()
		ethTx.GasFeeCap = tx.GasFeeCap().String()
		ethTx.Gas = tx.Gas()
		if tx.To() != nil {
			ethTx.To = tx.To().Hex()
		} else {
			return nil, errors.New("unknown transaction to")
		}
		ethTx.Value = tx.Value().String()
		ethTx.Data = hexutil.Encode(tx.Data())
		jsonString, err := json.Marshal(tx.AccessList())
		if err == nil {
			ethTx.AccessList = string(jsonString)
		}
		v, r, s := tx.RawSignatureValues()
		ethTx.V = v.String()
		ethTx.R = r.String()
		ethTx.S = s.String()
	}
	return ethTx, nil
}

// ConvertToEthereumReceipt 转换收据函数，将链上数据类型转换为数据库类型
func ConvertToEthereumReceipt(receipt *types.Receipt) (*EthereumReceipt, error) {
	ethReceipt := &EthereumReceipt{}

	ethReceipt.TxHash = receipt.TxHash.Hex()
	jsonBytes, err := receipt.MarshalJSON()
	if err != nil {
		return nil, err
	}
	ethReceipt.OriginJsonString = string(jsonBytes)

	ethReceipt.TxType = uintToStringType(receipt.Type)
	if ethReceipt.TxType == OtherTx {
		return nil, errors.New("unknown transaction type")
	}

	ethReceipt.PostState = hex.EncodeToString(receipt.PostState)
	ethReceipt.Status = uintToStringStatus(receipt.Status)
	ethReceipt.CumulativeGasUsed = receipt.CumulativeGasUsed
	bloom, err := receipt.Bloom.MarshalText()
	if err == nil {
		ethReceipt.Bloom = string(bloom)
	}
	ethReceipt.GasUsed = receipt.GasUsed
	ethReceipt.EffectiveGasPrice = receipt.EffectiveGasPrice.String()
	ethReceipt.BlobGasUsed = receipt.BlobGasUsed
	if receipt.BlobGasPrice != nil {
		ethReceipt.BlobGasPrice = receipt.BlobGasPrice.String()
	}
	ethReceipt.BlockHash = receipt.BlockHash.Hex()

	if receipt.BlockNumber != nil {
		ethReceipt.BlockNumber = receipt.BlockNumber.String()
	}
	ethReceipt.TransactionIndex = receipt.TransactionIndex

	logsJson, err := json.Marshal(receipt.Logs)
	if err == nil {
		ethReceipt.Logs = string(logsJson)
	}

	return ethReceipt, nil
}

// ConvertToComparisonReceipt 转换对比收据函数，将链上数据类型转换为数据库类型
func ConvertToComparisonReceipt(receipt *types.Receipt) (*ComparisonReceipt, error) {
	ethReceipt := &ComparisonReceipt{}

	ethReceipt.TxHash = receipt.TxHash.Hex()
	jsonBytes, err := receipt.MarshalJSON()
	if err != nil {
		return nil, err
	}
	ethReceipt.OriginJsonString = string(jsonBytes)

	ethReceipt.TxType = uintToStringType(receipt.Type)
	if ethReceipt.TxType == OtherTx {
		return nil, errors.New("unknown transaction type")
	}

	ethReceipt.PostState = hex.EncodeToString(receipt.PostState)
	ethReceipt.Status = uintToStringStatus(receipt.Status)
	ethReceipt.CumulativeGasUsed = receipt.CumulativeGasUsed
	bloom, err := receipt.Bloom.MarshalText()
	if err == nil {
		ethReceipt.Bloom = string(bloom)
	}
	ethReceipt.GasUsed = receipt.GasUsed
	ethReceipt.EffectiveGasPrice = receipt.EffectiveGasPrice.String()
	ethReceipt.BlobGasUsed = receipt.BlobGasUsed
	if receipt.BlobGasPrice != nil {
		ethReceipt.BlobGasPrice = receipt.BlobGasPrice.String()
	}
	ethReceipt.BlockHash = receipt.BlockHash.Hex()

	if receipt.BlockNumber != nil {
		ethReceipt.BlockNumber = receipt.BlockNumber.String()
	}
	ethReceipt.TransactionIndex = receipt.TransactionIndex

	logsJson, err := json.Marshal(receipt.Logs)
	if err == nil {
		ethReceipt.Logs = string(logsJson)
	}

	return ethReceipt, nil
}

func uintToStringType(t uint8) string {
	switch t {
	case types.LegacyTxType:
		return LegacyTx
	case types.AccessListTxType:
		return AccessListTx
	case types.DynamicFeeTxType:
		return DynamicFeeTx
	default:
		return OtherTx
	}
}

func uintToStringStatus(t uint64) string {
	switch t {
	case types.ReceiptStatusFailed:
		return FailedStatus
	case types.ReceiptStatusSuccessful:
		return SuccessfulStatus
	default:
		return OtherStatus
	}
}
