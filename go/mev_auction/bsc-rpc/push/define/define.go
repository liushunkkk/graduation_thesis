package define

import (
	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/common/hexutil"
	"time"
)

type BuilderParam struct {
	Param      *Param `json:"param"`
	BundleHash common.Hash
	Counter    int
}

type Param struct {
	UserId             int
	Txs                []string      `json:"txs"`
	MaxBlockNumber     uint64        `json:"maxBlockNumber,omitempty"`
	BlockNumber        string        `json:"blockNumber,omitempty"`
	MinTimestamp       uint64        `json:"minTimestamp"`
	MaxTimestamp       uint64        `json:"maxTimestamp"`
	RevertingTxHashes  []string      `json:"revertingTxHashes"`
	SoulPointSignature hexutil.Bytes `json:"48spSign,omitempty"` // <<<<<<------ This is the 48SP signature
	ArrivalTime        time.Time     `json:"arrivalTime,omitempty"`
}

// SseBundleData SSE
type SseBundleData struct {
	ChainID          string                       `json:"chainID"`
	Hash             string                       `json:"hash"`
	SseTxs           []SseTxData                  `json:"txs"`
	NextBlockNumber  uint64                       `json:"nextBlockNumber"`
	MaxBlockNumber   uint64                       `json:"maxBlockNumber"`
	ProxyBidContract string                       `json:"proxyBidContract"`
	RefundAddress    string                       `json:"refundAddress"`
	RefundCfg        int                          `json:"refundCfg"`
	State            map[string]map[string]string `json:"state,omitempty"`
	CreateTime       *time.Time                   `json:"createTime,omitempty"`
	TimeOut          int                          `json:"timeOut,omitempty"`
	RpcID            string                       `json:"rpc_id,omitempty"`
}

type SseTxData struct {
	Tx               string   `json:"-"` // 不参与序列话
	Hash             string   `json:"hash,omitempty"`
	From             string   `json:"from,omitempty"`
	To               string   `json:"to,omitempty"`
	Value            string   `json:"value,omitempty"`
	Nonce            uint64   `json:"nonce,omitempty"`
	CallData         string   `json:"calldata,omitempty"`
	FunctionSelector string   `json:"functionSelector,omitempty"`
	GasLimit         uint64   `json:"gasLimit,omitempty"`
	GasPrice         uint64   `json:"gasPrice,omitempty"`
	Logs             []SseLog `json:"logs,omitempty"`
	ReceiptJson      string   `json:"-"` // 不参与序列话
	Selector         string   `json:"-"` // 不参与序列话
}

type SseLog struct {
	Address string   `json:"address,omitempty"`
	Topics  []string `json:"topics,omitempty"`
	Data    string   `json:"data,omitempty"`
}
