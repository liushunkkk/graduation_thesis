package types

import (
	"github.com/ethereum/go-ethereum/portal/zrpc_client/typed/rpc_portal/txv2"
	"github.com/ethereum/go-ethereum/push/define"
	"math/big"
	"sync/atomic"
	"time"

	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/common/hexutil"
	"github.com/ethereum/go-ethereum/rlp"
)

const (
	// MaxTxPerBundle is the max transactions in each bundle
	MaxTxPerBundle = 50
	// MaxBundleCounter is the max depth for bundle
	MaxBundleCounter = 2
	// MaxBundleAliveBlock is the max alive block for bundle
	MaxBundleAliveBlock = 100
	// MaxBundleAliveTime is the max alive time for bundle
	MaxBundleAliveTime = 5 * 60 // second
)

// SendBundleArgs represents the arguments for a call.
type SendBundleArgs struct {
	Txs               []hexutil.Bytes `json:"txs"`
	MaxBlockNumber    uint64          `json:"maxBlockNumber"`
	MinTimestamp      *uint64         `json:"minTimestamp"`
	MaxTimestamp      *uint64         `json:"maxTimestamp"`
	RevertingTxHashes []common.Hash   `json:"revertingTxHashes"`
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

type SearcherNeed struct {
	Hash  common.Hash     // parent bundle hash
	Txs   []hexutil.Bytes // txs
	Eager bool            // 非常渴望，留块4个
}

// constant for hints
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

const (
	BundleOK = iota
	BundleNonceTooHigh
	BundleFeeCapTooLow
	BundleErr
)

type Bundle struct {
	Txs               Transactions
	MaxBlockNumber    uint64
	MinTimestamp      uint64
	MaxTimestamp      uint64
	RevertingTxHashes []common.Hash

	Price *big.Int // for bundle compare and prune

	// caches
	hash atomic.Value
	size atomic.Value

	ParentHash    common.Hash // Record the original hash of the bundle
	Parent        *Bundle
	Counter       int // Record the number of times this bundle is used，maximum is 2
	Hint          map[string]bool
	RefundAddress common.Address
	RefundPercent int
	From          common.Address
	RPCID         string
	State         int

	PrivacyPeriod    uint32
	PrivacyBuilder   []string
	BroadcastBuilder []string

	ArrivalTime time.Time
	Erc20Tx     bool // can send tx to public node
}

func (bundle *Bundle) IsProtected() bool {
	return len(bundle.RevertingTxHashes) == 0
}

func (bundle *Bundle) IsPrivate() bool {
	for _, v := range bundle.Hint {
		if v {
			return false
		}
	}
	return true
}

func (bundle *Bundle) ValueBaseHint(hint string, trueValue any, falseValue any) any {
	if bundle.Hint[hint] {
		return trueValue
	} else {
		return falseValue
	}
}

func (bundle *Bundle) Size() uint64 {
	if size := bundle.size.Load(); size != nil {
		return size.(uint64)
	}
	c := writeCounter(0)
	rlp.Encode(&c, bundle)

	size := uint64(c)
	bundle.size.Store(size)
	return size
}

// Hash returns the bundle hash.
func (bundle *Bundle) Hash() common.Hash {
	if hash := bundle.hash.Load(); hash != nil {
		return hash.(common.Hash)
	}

	var hashStr []common.Hash
	if bundle.Parent != nil {
		hashStr = append(hashStr, bundle.Parent.Hash())
	}
	for _, tx := range bundle.Txs {
		hashStr = append(hashStr, tx.Hash())
	}
	h := rlpHash(hashStr)
	bundle.hash.Store(h)
	return h
}

func (bundle *Bundle) GetRevertingTxHashes() []string {
	var hashes []string

	if bundle.Parent != nil {
		hashes = append(hashes, bundle.Parent.GetRevertingTxHashes()...)
	}

	for _, hash := range bundle.RevertingTxHashes {
		hashes = append(hashes, hash.String())
	}
	return hashes
}

func (bundle *Bundle) GetTxs() []string {
	var txs []string
	if bundle.Parent != nil {
		txs = append(txs, bundle.Parent.GetTxs()...)
	}
	for _, tx := range bundle.Txs {
		binary, _ := tx.MarshalBinary()
		txs = append(txs, hexutil.Encode(binary))
	}
	return txs
}

func (bundle *Bundle) GetTxHashes() []string {
	var txHashes []string
	for _, tx := range bundle.Txs {
		txHashes = append(txHashes, tx.Hash().Hex())
	}
	return txHashes
}

func (bundle *Bundle) GenBuilderReq(header *Header) (*define.Param, *txv2.BundleSaveRequest) {
	p := &define.Param{
		Txs:               bundle.GetTxs(),
		MaxBlockNumber:    bundle.MaxBlockNumber,
		BlockNumber:       hexutil.EncodeBig(big.NewInt(0).Add(header.Number, big.NewInt(1))),
		MinTimestamp:      0,
		MaxTimestamp:      0,
		RevertingTxHashes: bundle.GetRevertingTxHashes(),
	}
	if p.MaxBlockNumber > header.Number.Uint64()+100 {
		p.MaxBlockNumber = header.Number.Uint64() + 100
	}
	var bsr = &txv2.BundleSaveRequest{
		ChainId:    "56",
		RpcId:      bundle.RPCID,
		BundleHash: bundle.Hash().Hex(),
		ParentHash: bundle.ParentHash.Hex(),
	}
	if bundle.Counter == 0 {
		t := &txv2.TxsData{
			TxHash:         bundle.Txs[0].Hash().Hex(),
			Type:           0,
			ArrivalTime:    bundle.ArrivalTime.Format(time.RFC3339Nano),
			BundleSendTime: time.Now().Format(time.RFC3339Nano),
		}
		bsr.TxsData = append(bsr.TxsData, t)
	} else if bundle.Counter == 1 && bundle.Parent != nil {
		t := &txv2.TxsData{
			TxHash:         bundle.Parent.Txs[0].Hash().Hex(),
			Type:           0,
			ArrivalTime:    bundle.Parent.ArrivalTime.Format(time.RFC3339Nano),
			BundleSendTime: time.Now().Format(time.RFC3339Nano),
		}
		bsr.TxsData = append(bsr.TxsData, t)

		builderPercent := 0
		if percent, ok := BuilderPercentMap[bundle.Parent.RPCID]; ok {
			builderPercent = percent
		} else {
			builderPercent = RpcBuilderProfitPercent
		}
		for _, bgTx := range bundle.Txs {
			t := &txv2.TxsData{
				TxHash:         bgTx.Hash().Hex(),
				Type:           1,
				ArrivalTime:    bundle.ArrivalTime.Format(time.RFC3339Nano),
				BundleSendTime: time.Now().Format(time.RFC3339Nano),
				RefundData: &txv2.RefundData{
					RefundAddress: bundle.Parent.RefundAddress.Hex(),
					RefundPercent: uint32(bundle.Parent.RefundPercent),
					BribePercent:  uint32(builderPercent),
					ScutumPercent: uint32(RpcBribePercent),
				},
			}
			bsr.TxsData = append(bsr.TxsData, t)
		}
	} else if bundle.Counter == 2 && bundle.Parent != nil && bundle.Parent.Parent != nil {
		t := &txv2.TxsData{
			TxHash:         bundle.Parent.Parent.Txs[0].Hash().Hex(),
			Type:           0,
			ArrivalTime:    bundle.Parent.Parent.ArrivalTime.Format(time.RFC3339Nano),
			BundleSendTime: time.Now().Format(time.RFC3339Nano),
		}
		bsr.TxsData = append(bsr.TxsData, t)

		builderPercent := 0
		if percent, ok := BuilderPercentMap[bundle.Parent.Parent.RPCID]; ok {
			builderPercent = percent
		} else {
			builderPercent = RpcBuilderProfitPercent
		}

		for _, bgTx := range bundle.Parent.Txs {
			t := &txv2.TxsData{
				TxHash:         bgTx.Hash().Hex(),
				Type:           1,
				ArrivalTime:    bundle.Parent.ArrivalTime.Format(time.RFC3339Nano),
				BundleSendTime: time.Now().Format(time.RFC3339Nano),
				RefundData: &txv2.RefundData{
					RefundAddress: bundle.Parent.Parent.RefundAddress.Hex(),
					RefundPercent: uint32(bundle.Parent.Parent.RefundPercent),
					BribePercent:  uint32(builderPercent),
					ScutumPercent: uint32(RpcBribePercent),
				},
			}
			bsr.TxsData = append(bsr.TxsData, t)
		}

		builderPercent = 0
		if percent, ok := BuilderPercentMap[bundle.Parent.RPCID]; ok {
			builderPercent = percent
		} else {
			builderPercent = RpcBuilderProfitPercent
		}

		for _, bgTx := range bundle.Txs {
			t := &txv2.TxsData{
				TxHash:         bgTx.Hash().Hex(),
				Type:           2,
				ArrivalTime:    bundle.ArrivalTime.Format(time.RFC3339Nano),
				BundleSendTime: time.Now().Format(time.RFC3339Nano),
				RefundData: &txv2.RefundData{
					RefundAddress: bundle.Parent.RefundAddress.Hex(),
					RefundPercent: uint32(bundle.Parent.RefundPercent),
					BribePercent:  uint32(builderPercent),
					ScutumPercent: uint32(RpcBribePercent),
				},
			}
			bsr.TxsData = append(bsr.TxsData, t)
		}
	} else if bundle.Counter == 1 && bundle.Parent == nil {
		for _, rawTx := range bundle.Txs {
			t := &txv2.TxsData{
				TxHash:         rawTx.Hash().Hex(),
				Type:           0,
				ArrivalTime:    bundle.ArrivalTime.Format(time.RFC3339Nano),
				BundleSendTime: time.Now().Format(time.RFC3339Nano),
			}
			bsr.TxsData = append(bsr.TxsData, t)
		}
	} else if bundle.Counter == 2 && bundle.Parent != nil && bundle.Parent.Parent == nil {
		for _, rawTx := range bundle.Parent.Txs {
			t := &txv2.TxsData{
				TxHash:         rawTx.Hash().Hex(),
				Type:           0,
				ArrivalTime:    bundle.Parent.ArrivalTime.Format(time.RFC3339Nano),
				BundleSendTime: time.Now().Format(time.RFC3339Nano),
			}
			bsr.TxsData = append(bsr.TxsData, t)
		}

		builderPercent := 0
		if percent, ok := BuilderPercentMap[bundle.Parent.RPCID]; ok {
			builderPercent = percent
		} else {
			builderPercent = RpcBuilderProfitPercent
		}

		for _, bgTx := range bundle.Txs {
			t := &txv2.TxsData{
				TxHash:         bgTx.Hash().Hex(),
				Type:           1,
				ArrivalTime:    bundle.ArrivalTime.Format(time.RFC3339Nano),
				BundleSendTime: time.Now().Format(time.RFC3339Nano),
				RefundData: &txv2.RefundData{
					RefundAddress: bundle.Parent.RefundAddress.Hex(),
					RefundPercent: uint32(bundle.Parent.RefundPercent),
					BribePercent:  uint32(builderPercent),
					ScutumPercent: uint32(RpcBribePercent),
				},
			}
			bsr.TxsData = append(bsr.TxsData, t)
		}
	} else {
		bsr = nil
	}
	return p, bsr
}

type SimulatedBundle struct {
	OriginalBundle *Bundle

	BundleGasFees   *big.Int
	BundleGasPrice  *big.Int
	BundleGasUsed   uint64
	EthSentToSystem *big.Int

	RpcBundlePrice *big.Int // bundler price for rpc service
}

type RpcTransactionResult struct {
	BlockHash   string `json:"blockHash"`
	BlockNumber string `json:"blockNumber"`
	GasPrice    string `json:"gasPrice"`
	Time        string `json:"time"`
	From        string `json:"from"`
	Status      string `json:"status"`
	GasUsed     string `json:"gasUsed"`
	Data        []byte `json:"data"`
	Logs        []*Log `json:"logs"`
}
