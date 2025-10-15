package core

import (
	"bsc-rpc-client/client"
	"bsc-rpc-client/model"
	"bufio"
	"encoding/json"
	"fmt"
	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/common/hexutil"
	"io"
	"math/rand"
	"net/http"
	"time"
)

// --- 数据结构，与服务端一致 ---

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
	Tx               string   `json:"-"` // 不参与序列化
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
	ReceiptJson      string   `json:"-"` // 不参与序列化
	Selector         string   `json:"-"` // 不参与序列化
}

type SseLog struct {
	Address string   `json:"address,omitempty"`
	Topics  []string `json:"topics,omitempty"`
	Data    string   `json:"data,omitempty"`
}

// --- 搜索者定义 ---

type Searcher struct {
	id        int
	group     string
	prob      float64
	stream    string
	rpc       string
	client    *http.Client
	rpcClient *client.JSONRPCClient
}

func NewSearcher(id int, group string, prob float64, streamURL, rpcURL string) *Searcher {
	return &Searcher{
		id:        id,
		group:     group,
		prob:      prob,
		stream:    streamURL,
		rpc:       rpcURL,
		rpcClient: client.NewJSONRPCClient(rpcURL),
		client:    &http.Client{Timeout: 0}, // SSE连接不设置超时
	}
}

func (s *Searcher) Start() {
	req, _ := http.NewRequest("GET", s.stream, nil)
	resp, err := s.client.Do(req)
	if err != nil {
		fmt.Printf("[Searcher %02d][%s] SSE连接失败: %v\n", s.id, s.group, err)
		return
	}
	defer resp.Body.Close()

	reader := bufio.NewReader(resp.Body)
	fmt.Printf("[Searcher %02d][%s] 已连接SSE\n", s.id, s.group)

	for {
		line, err := reader.ReadString('\n')
		if err != nil {
			if err == io.EOF {
				fmt.Printf("[Searcher %02d] SSE连接关闭\n", s.id)
				return
			}
			fmt.Printf("[Searcher %02d] 读取错误: %v\n", s.id, err)
			return
		}

		if len(line) < 6 || line[:5] != "data:" {
			continue
		}

		payload := line[5:]
		var msg SseBundleData
		if err := json.Unmarshal([]byte(payload), &msg); err != nil {
			continue
		}

		s.handleMessage(&msg)
	}
}

func (s *Searcher) handleMessage(msg *SseBundleData) {
	r := rand.Float64()
	if r < s.prob {
		fmt.Printf("[Searcher %02d][%s] 收到 bundle (%s)，发送响应 (%.0f%%)\n",
			s.id, s.group, msg.Hash, s.prob*100)
		s.sendBundle(msg)
	} else {
		fmt.Printf("[Searcher %02d][%s] 收到 bundle (%s)，跳过发送 (%.0f%%)\n",
			s.id, s.group, msg.Hash, s.prob*100)
	}
}

func (s *Searcher) sendBundle(msg *SseBundleData) {
	args := &model.SendMevBundleArgs{
		Hash:           randomHash(),
		Txs:            []hexutil.Bytes{randomTx()},
		MaxBlockNumber: 12345678,
		Hint:           map[string]bool{"arbitrage": true},
		RefundAddress:  common.HexToAddress("0x0000000000000000000000000000000000000000"),
		RefundPercent:  80,
	}
	resp, err := s.rpcClient.SendMevBundle(args)
	if err != nil {
		fmt.Printf("[Searcher %02d] 发送bundle失败: %v\n", s.id, err)
		return
	}
	fmt.Printf("[Searcher %02d] 发送bundle成功，BundleHash: %s\n", s.id, resp.BundleHash.Hex())
}

func randomHash() common.Hash {
	var b [32]byte
	rand.Read(b[:])
	return common.BytesToHash(b[:])
}

func randomTx() hexutil.Bytes {
	raw := make([]byte, 128)
	rand.Read(raw)
	return hexutil.Bytes(raw)
}

func InitSearcher() {
	rand.Seed(time.Now().UnixNano())

	streamURL := "http://localhost:8080/stream"
	rpcURL := "http://localhost:8080"

	var searchers []*Searcher
	for i := 0; i < 5; i++ {
		searchers = append(searchers, NewSearcher(i+1, "A", 0.9, streamURL, rpcURL))
	}
	for i := 0; i < 5; i++ {
		searchers = append(searchers, NewSearcher(6+i, "B", 0.7, streamURL, rpcURL))
	}
	for i := 0; i < 5; i++ {
		searchers = append(searchers, NewSearcher(11+i, "C", 0.5, streamURL, rpcURL))
	}

	for _, s := range searchers {
		go s.Start()
	}
}
