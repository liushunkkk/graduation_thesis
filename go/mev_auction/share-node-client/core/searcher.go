package core

import (
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
	"share-node-client/client"
	"share-node-client/model"
	"share-node-client/zap_logger"
	"time"
)

// --- 数据结构，与服务端一致 ---

type Hint struct {
	Hash        common.Hash     `json:"hash"`
	Logs        []CleanLog      `json:"logs"`
	Txs         []TxHint        `json:"txs"`
	MevGasPrice *hexutil.Big    `json:"mevGasPrice,omitempty"`
	GasUsed     *hexutil.Uint64 `json:"gasUsed,omitempty"`
}

type TxHint struct {
	Hash             *common.Hash    `json:"hash,omitempty"`
	To               *common.Address `json:"to,omitempty"`
	FunctionSelector *hexutil.Bytes  `json:"functionSelector,omitempty"`
	CallData         *hexutil.Bytes  `json:"callData,omitempty"`
}

type CleanLog struct {
	// address of the contract that generated the event
	Address common.Address `json:"address"`
	// list of topics provided by the contract.
	Topics []common.Hash `json:"topics"`
	// supplied by the contract, usually ABI-encoded
	Data hexutil.Bytes `json:"data"`
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

func (s *Searcher) Start(ctx context.Context) {
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

			if len(line) < 6 || line[:5] != "data:" {
				continue
			}

			payload := line[5:]
			var msg Hint
			if err := json.Unmarshal([]byte(payload), &msg); err != nil {
				continue
			}

			go s.handleMessage(&msg)
		}
	}
}

func getAllTxHash(msg *Hint) []string {
	var res []string
	for _, tx := range msg.Txs {
		res = append(res, tx.Hash.Hex())
	}
	return res
}

func (s *Searcher) handleMessage(msg *Hint) {
	r := rand.Float64()
	if len(msg.Txs) == 1 && r < s.prob {
		zap_logger.Zap.Info(fmt.Sprintf("[Searcher][%02d] receive one level bundle，准备发送，概率值： (%.2f)/[%s]",
			s.id, r, s.group), zap.Any("receiveTime", time.Now().UnixMicro()), zap.Any("parentHash", msg.Hash), zap.Any("txHashes", getAllTxHash(msg)))
		// 随机休眠一段时间，模拟在计算
		time.Sleep(time.Millisecond * time.Duration(RandomAround300()))
		s.sendBundle(msg)
	}
	// mev share node只允许searcher套利一次
	//else if len(msg.Txs) == 2 && r < s.prob/3.0 {
	//	zap_logger.Zap.Info(fmt.Sprintf("[Searcher][%02d] receive two level bundle，准备发送，概率值： (%.2f)/[%s]",
	//		s.id, r, s.group), zap.Any("receiveTime", time.Now().UnixMicro()), zap.Any("parentHash", msg.Hash), zap.Any("txHashes", getAllTxHash(msg)))
	//	// 随机休眠一段时间，模拟在计算
	//	time.Sleep(time.Millisecond * time.Duration(RandomAround300()))
	//	s.sendBundle(msg)
	//}
}

func (s *Searcher) sendBundle(msg *Hint) {
	next := SearcherTxLoader.Next()
	if next == nil {
		return
	}
	tx := new(types.Transaction)
	_ = tx.UnmarshalJSON(hexutil.Bytes(next.OriginJsonString))

	input, _ := tx.MarshalBinary()

	args := &model.SendMevBundleArgs{
		UserId:  0,
		Version: "v0.1",
		Inclusion: model.MevBundleInclusion{
			BlockNumber: hexutil.Uint64(GlobalBlockManager.GetCurrentBlock() + 1),
			MaxBlock:    hexutil.Uint64(GlobalBlockManager.GetCurrentBlock() + 2),
		},
		Body: []model.MevBundleBody{
			{
				Hash: &msg.Hash,
			},
			{
				Tx: (*hexutil.Bytes)(&input),
			},
		},
		Metadata: &model.MevBundleMetadata{Signer: common.HexToAddress("0x6927b1FF9E8ef81F58f457d87d775b1f44d72027")},
	}
	resp, err := s.rpcClient.SendMevBundle(args)
	if err != nil {
		zap_logger.Zap.Info(fmt.Sprintf("[Searcher][%02d] 发送bundle失败: %v", s.id, err))
		return
	}
	zap_logger.Zap.Info(fmt.Sprintf("[Searcher][%02d] 发送bundle成功，BundleHash: %s", s.id, resp.BundleHash.Hex()))
}

func InitSearcher(ctx context.Context, num int) {
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
		go s.Start(ctx)
	}
}

func RandomAround300() int64 {
	rand.Seed(time.Now().UnixNano())        // 记得只在程序启动时调用一次
	return int64(300 + rand.Intn(101) - 50) // 300 ±50 => [250,350]
}
