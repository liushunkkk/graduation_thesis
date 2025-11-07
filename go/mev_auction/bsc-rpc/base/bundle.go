package base

import (
	"github.com/ethereum/go-ethereum-test/push/define"
	"github.com/ethereum/go-ethereum/core/types"
	"github.com/ethereum/go-ethereum/crypto"
	"github.com/ethereum/go-ethereum/portal/zrpc_client/typed/rpc_portal/txv2"
	"golang.org/x/crypto/sha3"
	"math/big"
	"sync/atomic"
	"time"

	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/common/hexutil"
	"github.com/ethereum/go-ethereum/rlp"
)

var CurrentHeader *types.Header

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

type Bundle struct {
	UserId            int
	Txs               types.Transactions
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
	//c := types.writeCounter(0)
	//rlp.Encode(&c, bundle)

	size := uint64(0)
	bundle.size.Store(size)
	return size
}

// Hash returns the bundle hash.
func (bundle *Bundle) Hash() (h common.Hash) {
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
	sha := sha3.NewLegacyKeccak256().(crypto.KeccakState)
	sha.Reset()
	rlp.Encode(sha, hashStr)
	sha.Read(h[:])
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

func (bundle *Bundle) GenBuilderReq(header *types.Header) (*define.Param, *txv2.BundleSaveRequest) {
	p := &define.Param{
		UserId:            bundle.UserId,
		Txs:               bundle.GetTxs(),
		MaxBlockNumber:    bundle.MaxBlockNumber,
		BlockNumber:       hexutil.EncodeBig(big.NewInt(0).Add(header.Number, big.NewInt(1))),
		MinTimestamp:      0,
		MaxTimestamp:      0,
		RevertingTxHashes: bundle.GetRevertingTxHashes(),
		ArrivalTime:       bundle.ArrivalTime,
	}
	if p.MaxBlockNumber > header.Number.Uint64()+100 {
		p.MaxBlockNumber = header.Number.Uint64() + 100
	}
	return p, nil
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
	BlockHash   string       `json:"blockHash"`
	BlockNumber string       `json:"blockNumber"`
	GasPrice    string       `json:"gasPrice"`
	Time        string       `json:"time"`
	From        string       `json:"from"`
	Status      string       `json:"status"`
	GasUsed     string       `json:"gasUsed"`
	Data        []byte       `json:"data"`
	Logs        []*types.Log `json:"logs"`
}
