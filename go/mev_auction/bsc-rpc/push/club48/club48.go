package club48

import (
	"bytes"
	"encoding/json"
	"fmt"
	"github.com/ethereum/go-ethereum-test/push/define"
	"github.com/ethereum/go-ethereum-test/zap_logger"
	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/common/lru"
	"github.com/ethereum/go-ethereum/log"
	"github.com/spf13/cast"
	"go.uber.org/zap"
	"io"
	"net/http"
	"time"
)

var cache *lru.Cache[string, struct{}]

func init() {
	cache = lru.NewCache[string, struct{}](10000)
}

type Club48 struct {
	HttpAddress   string
	PublicAddress string
}

type BuilderData struct {
	JsonRPC string         `json:"jsonrpc"`
	Method  string         `json:"method"`
	Params  []define.Param `json:"params"`
	ID      string         `json:"id"`
}

func NewClub48() *Club48 {
	return &Club48{HttpAddress: "https://puissant-builder.48.club/", PublicAddress: "0x4848489f0b2BEdd788c696e2D79b6b69D7484848"}
}

func (club48 *Club48) GetPublicAddress() common.Address {
	return common.HexToAddress(club48.PublicAddress)
}

func (club48 *Club48) SendBundle(param define.Param, hash common.Hash) {
	param.MaxBlockNumber = cast.ToUint64(param.BlockNumber)
	param.BlockNumber = "" // For compatibility with smith

	bd := &BuilderData{}
	bd.ID = "1"
	bd.JsonRPC = "2.0"
	bd.Method = "eth_sendBundle"
	bd.Params = []define.Param{param}

	data, _ := json.Marshal(bd)

	cost := time.Now().Sub(param.ArrivalTime).Microseconds()
	zap_logger.Zap.Info("[club48-send]", zap.Any("hash", hash), zap.Any("cost", cost), zap.Any("txs", len(param.Txs)), zap.Any("userId", param.UserId))
	time.Sleep(20 * time.Millisecond)
	return

	client := &http.Client{
		Timeout: 3 * time.Second,
	}
	httpReq, err := http.NewRequest("POST", club48.HttpAddress, bytes.NewBuffer(data))
	if err != nil {
		log.Error("Error creating club48 request:", err)
		return
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := client.Do(httpReq)
	if err != nil {
		log.Error("send to club48 builder error:", err)
		return
	}
	defer resp.Body.Close()

	// 读取响应体
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		log.Error("receive club48 builder resp body error:", err)
		return
	}
	zap_logger.Zap.Info(fmt.Sprintf("club48 builder resp[%v]:%s", hash, string(body)))
}

func (club48 *Club48) SendRawPrivateTransaction(tx string, bundleHash common.Hash) {
	if _, ok := cache.Get(bundleHash.String()); ok {
		return
	}
	zap_logger.Zap.Info(" club48 send private tx:", zap.Any("bundleHash", bundleHash), zap.Any("tx", tx))

	bd := &map[string]any{
		"jsonrpc": "2.0",
		"id":      "1",
		"method":  "eth_sendPrivateTransactionWith48SP",
		"params":  []any{tx},
	}

	data, _ := json.Marshal(bd)

	client := &http.Client{
		Timeout: 3 * time.Second,
	}
	httpReq, err := http.NewRequest("POST", club48.HttpAddress, bytes.NewBuffer(data))
	if err != nil {
		log.Error("Error creating club48 request private tx:", err)
		return
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := client.Do(httpReq)
	if err != nil {
		log.Error("send to club48 builder private tx error:", err)
		return
	}
	defer resp.Body.Close()

	// 读取响应体
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		log.Error("receive builder club48 private tx resp body error:", err)
		return
	}
	zap_logger.Zap.Info(fmt.Sprintf("club48 builder private tx resp[%v]:%s", bundleHash, string(body)))

	cache.Add(bundleHash.String(), struct{}{})
}
