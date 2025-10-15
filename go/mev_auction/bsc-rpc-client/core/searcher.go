package core

import (
	"bsc-rpc-client/client"
	"bsc-rpc-client/model"
	"bsc-rpc-client/zap_logger"
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/common/hexutil"
	"github.com/ethereum/go-ethereum/core/types"
	"go.uber.org/zap"
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

func (s *Searcher) Start(ctx context.Context, block bool) {
	req, _ := http.NewRequest("GET", s.stream, nil)
	resp, err := s.client.Do(req)
	if err != nil {
		zap_logger.Zap.Info(fmt.Sprintf("[Searcher %02d][%s] SSE连接失败: %v", s.id, s.group, err))
		return
	}
	defer resp.Body.Close()

	reader := bufio.NewReader(resp.Body)
	zap_logger.Zap.Info(fmt.Sprintf("[Searcher %02d][%s] 已连接SSE", s.id, s.group))

	for {
		select {
		case <-ctx.Done():
			return
		default:
			line, err := reader.ReadString('\n')
			if err != nil {
				if err == io.EOF {
					zap_logger.Zap.Info(fmt.Sprintf("[Searcher %02d] SSE连接关闭", s.id))
					return
				}
				zap_logger.Zap.Info(fmt.Sprintf("[Searcher %02d] 读取错误: %v", s.id, err))
				return
			}

			if block && s.id == 6 {
				time.Sleep(1 * time.Minute) // 模拟某个搜索者阻塞的情况，阻塞一分钟
			}

			if len(line) < 6 || line[:5] != "data:" {
				continue
			}

			payload := line[5:]
			var msg SseBundleData
			if err := json.Unmarshal([]byte(payload), &msg); err != nil {
				continue
			}

			go s.handleMessage(&msg)
		}
	}
}

func getAllTxHash(msg *SseBundleData) []string {
	var res []string
	for _, tx := range msg.SseTxs {
		res = append(res, tx.Hash)
	}
	return res
}

func (s *Searcher) handleMessage(msg *SseBundleData) {
	r := rand.Float64()
	if len(msg.SseTxs) == 1 && r < s.prob {
		zap_logger.Zap.Info(fmt.Sprintf("[Searcher][%02d] receive one level bundle，准备发送，概率值： (%.2f)/[%s]",
			s.id, r, s.group), zap.Any("receiveTime", time.Now().UnixMicro()), zap.Any("parentHash", msg.Hash), zap.Any("txHashes", getAllTxHash(msg)))
		// 随机休眠一段时间，模拟在计算
		time.Sleep(time.Millisecond * time.Duration(RandomAround300()))
		s.sendBundle(msg)
	} else if len(msg.SseTxs) == 2 && r < s.prob/3.0 {
		zap_logger.Zap.Info(fmt.Sprintf("[Searcher][%02d] receive two level bundle，准备发送，概率值： (%.2f)/[%s]",
			s.id, r, s.group), zap.Any("receiveTime", time.Now().UnixMicro()), zap.Any("parentHash", msg.Hash), zap.Any("txHashes", getAllTxHash(msg)))
		// 随机休眠一段时间，模拟在计算
		time.Sleep(time.Millisecond * time.Duration(RandomAround300()))
		s.sendBundle(msg)
	}
}

func (s *Searcher) sendBundle(msg *SseBundleData) {
	next := SearcherTxLoader.Next()
	if next == nil {
		return
	}
	tx := new(types.Transaction)
	_ = tx.UnmarshalJSON(hexutil.Bytes(next.OriginJsonString))

	input, _ := tx.MarshalBinary()

	args := &model.SendMevBundleArgs{
		Hash:           common.HexToHash(msg.Hash),
		Txs:            []hexutil.Bytes{input},
		MaxBlockNumber: msg.MaxBlockNumber,
		Hint:           model.GetAllHints(),
		RefundAddress:  common.HexToAddress(msg.RefundAddress),
		RefundPercent:  80,
	}
	resp, err := s.rpcClient.SendMevBundle(args)
	if err != nil {
		zap_logger.Zap.Info(fmt.Sprintf("[Searcher][%02d] 发送bundle失败: %v", s.id, err))
		return
	}
	zap_logger.Zap.Info(fmt.Sprintf("[Searcher][%02d] 发送bundle成功，BundleHash: %s", s.id, resp.BundleHash.Hex()))
}

func InitSearcher(ctx context.Context, num int, block bool) {
	rand.Seed(time.Now().UnixNano())

	streamURL := "http://localhost:8080/stream"
	rpcURL := "http://localhost:8080"
	var searchers []*Searcher
	for i := 0; i < num; i++ {
		searchers = append(searchers, NewSearcher(i+1, "A", 0.9, streamURL, rpcURL))
	}
	for i := 0; i < num; i++ {
		searchers = append(searchers, NewSearcher(6+i, "B", 0.7, streamURL, rpcURL))
	}
	for i := 0; i < num; i++ {
		searchers = append(searchers, NewSearcher(11+i, "C", 0.5, streamURL, rpcURL))
	}

	for _, s := range searchers {
		go s.Start(ctx, block)
	}
}

func RandomAround300() int64 {
	rand.Seed(time.Now().UnixNano())        // 记得只在程序启动时调用一次
	return int64(300 + rand.Intn(101) - 50) // 300 ±50 => [250,350]
}
