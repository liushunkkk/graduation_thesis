package base

import (
	"github.com/ethereum/go-ethereum-test/push/define"
	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/common/hexutil"
	"github.com/ethereum/go-ethereum/core/types"
	"github.com/ethereum/go-ethereum/log"
	"github.com/ethereum/go-ethereum/params"
	"github.com/spf13/cast"
	"math/big"
	"math/rand"
	"time"
)

var (
	RpcIdList []string // 需要观察的rpcId列表

	BidContractAddress      = common.Address{} // rpc system address to receive refund
	ProxyContractAddress    = common.Address{}
	RpcBribePercent         = 10  // bribe percentage the rpc system can obtain
	RpcBuilderProfitPercent = 100 //profit percentage builder can obtain after bundle accepted

	// direct transfer transaction setting
	RpcGasPrice = new(big.Int).Mul(big.NewInt(1), big.NewInt(params.GWei))
	RpcGasLimit = big.NewInt(21_000)
)

type BundleSimulator interface {
	ExecuteBundle(parent *types.Header, bundle *Bundle, rpcBribeAddress common.Address) (*big.Int, *define.SseBundleData, error)
}

type BundleSimulatorImpl struct {
	RpcSimulator *RpcSimulator
}

func NewBundleSimulator() *BundleSimulatorImpl {
	return &BundleSimulatorImpl{
		RpcSimulator: NewRpcSimulator(),
	}
}

func (s *BundleSimulatorImpl) ExecuteBundle(parent *types.Header, bundle *Bundle, rpcBribeAddress common.Address) (*big.Int, *define.SseBundleData, error) {
	price := 0
	if bundle.Parent != nil {
		time.Sleep(30 * time.Millisecond)
		if bundle.Parent.Parent != nil {
			price = 200 + rand.Intn(200)
		} else {
			price = 200 + rand.Intn(100)
		}
	} else {
		time.Sleep(20 * time.Millisecond)
		price = 100 + rand.Intn(100)
	}
	var sseTxs []define.SseTxData
	if bundle.Parent != nil {
		if bundle.Parent.Parent != nil {
			d, _ := s.RpcSimulator.BuildTxData(bundle, bundle.Parent.Parent.Txs[0], &types.Receipt{})
			sseTxs = append(sseTxs, d)
		}
		d, _ := s.RpcSimulator.BuildTxData(bundle, bundle.Txs[0], &types.Receipt{})
		sseTxs = append(sseTxs, d)
	}
	d, _ := s.RpcSimulator.BuildTxData(bundle, bundle.Txs[0], &types.Receipt{})
	sseTxs = append(sseTxs, d)
	return big.NewInt(int64(price)), &define.SseBundleData{
		ChainID:          "56",
		Hash:             bundle.hash.Load().(common.Hash).Hex(),
		SseTxs:           sseTxs,
		NextBlockNumber:  bundle.MaxBlockNumber,
		MaxBlockNumber:   bundle.MaxBlockNumber,
		ProxyBidContract: "0x74Ce839c6aDff544139f27C1257D34944B794605",
		RefundAddress:    bundle.RefundAddress.Hex(),
		RefundCfg:        1009050,
	}, nil
}

// RpcSimulator simulator for rpc service
type RpcSimulator struct{}

// NewRpcSimulator returns a default RpcSimulator
func NewRpcSimulator() *RpcSimulator {
	return &RpcSimulator{}
}

// GetSingleTxFee return SingleTxFee, SingleTxFee = RpcGasPrice * RpcGasLimit
// if RpcGasPrice < baseFee, return baseFee * RpcGasLimit
func (r *RpcSimulator) GetSingleTxFee(baseFee *big.Int) *big.Int {
	if RpcGasPrice.Cmp(baseFee) >= 0 {
		return new(big.Int).Mul(RpcGasPrice, RpcGasLimit)
	} else {
		return new(big.Int).Mul(baseFee, RpcGasLimit)
	}
}

// GetSingleTxTip get tip fee for single tx with baseFee
// SingleTxTip = (RpcGasPrice - baseFee) * RpcGasLimit
func (r *RpcSimulator) GetSingleTxTip(baseFee *big.Int) *big.Int {
	if RpcGasPrice.Cmp(baseFee) >= 0 {
		return new(big.Int).Mul(new(big.Int).Sub(RpcGasPrice, baseFee), RpcGasLimit)
	} else {
		return new(big.Int)
	}
}

// GetRemainBribe returns the remaining bribe after deducting commission for RPC service.
func (r *RpcSimulator) GetRemainBribe(totalBribe *big.Int) *big.Int {
	return common.PercentOf(totalBribe, 100-RpcBribePercent)
}

func (r *RpcSimulator) GetBribeToBuilderAndSender(totalBribe *big.Int, refundPercent int) (*big.Int, *big.Int) {
	remainBribe := r.GetRemainBribe(totalBribe)
	rpcBribeToBuilder := common.PercentOf(remainBribe, 100-refundPercent)
	rpcBribeToSender := new(big.Int).Sub(remainBribe, rpcBribeToBuilder)
	return rpcBribeToBuilder, rpcBribeToSender
}

func (r *RpcSimulator) BuildTxData(bundle *Bundle, tx *types.Transaction, receipt *types.Receipt) (define.SseTxData, error) {
	var logs []define.SseLog
	receiptJson, _ := receipt.MarshalJSON()
	txJson, _ := tx.MarshalJSON()
	for _, receiptLog := range receipt.Logs {
		var topicStrings []string
		for _, b := range receiptLog.Topics {
			topicStrings = append(topicStrings, b.Hex())
		}
		dataLog := define.SseLog{
			Address: receiptLog.Address.Hex(),
			Topics:  topicStrings,
			Data:    hexutil.Encode(receiptLog.Data),
		}
		logs = append(logs, dataLog)
	}
	// transfer wei to 0x 16
	v := "0x" + tx.Value().Text(16)

	// get from address of tx
	var from common.Address
	from, err := types.Sender(types.LatestSignerForChainID(tx.ChainId()), tx)
	if err != nil {
		return define.SseTxData{}, err
	}
	var callData, functionSelector string
	if len(tx.Data()) > 0 {
		callData = hexutil.Encode(tx.Data())
		// it must have a function selector, so length of callData Hex string must be bigger than 10
		if len(callData) < 10 {
			log.Warn("transaction's input data is invalid", "hash", tx.Hash().String())
			return define.SseTxData{}, err
		}
		functionSelector = callData[0:10]
	}
	// set txData according to hint
	currTxData := define.SseTxData{
		Hash:             bundle.ValueBaseHint(HintHash, tx.Hash().Hex(), "").(string),
		From:             bundle.ValueBaseHint(HintFrom, from.Hex(), "").(string),
		Value:            bundle.ValueBaseHint(HintValue, v, "").(string),
		Nonce:            cast.ToUint64(bundle.ValueBaseHint(HintNonce, tx.Nonce(), 0)),
		CallData:         bundle.ValueBaseHint(HintCallData, callData, "").(string),
		FunctionSelector: bundle.ValueBaseHint(HintFunctionSelector, functionSelector, "").(string),
		GasLimit:         cast.ToUint64(bundle.ValueBaseHint(HintGasLimit, tx.Gas(), 0)),
		GasPrice:         cast.ToUint64(bundle.ValueBaseHint(HintGasPrice, tx.GasPrice().Uint64(), 0)),
		Logs:             bundle.ValueBaseHint(HintLogs, logs, []define.SseLog{}).([]define.SseLog),
		Selector:         functionSelector,
		ReceiptJson:      string(receiptJson),
		Tx:               string(txJson),
	}
	if tx.To() != nil {
		currTxData.To = bundle.ValueBaseHint(HintTo, tx.To().Hex(), "").(string)
	}
	return currTxData, nil
}
