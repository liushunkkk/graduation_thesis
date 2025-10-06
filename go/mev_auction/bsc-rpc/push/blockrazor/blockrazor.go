package blockrazor

import (
	"bytes"
	"encoding/json"
	"fmt"
	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/log"
	. "github.com/ethereum/go-ethereum/log/zap"
	"github.com/ethereum/go-ethereum/metrics"
	"github.com/ethereum/go-ethereum/push/define"
	"github.com/spf13/cast"
	"go.uber.org/zap"
	"io"
	"net/http"
	"strings"
	"time"
)

var (
	BlockRazorFailureGauge    = metrics.NewRegisteredGauge("builder/blockrazor/failure/total", nil)
	BlockRazorLowFailureGauge = metrics.NewRegisteredGauge("builder/blockrazor/failure/gaslow", nil)
)

type BlockRazor struct {
	HttpAddress   string
	PublicAddress string
}

type BuilderData struct {
	JsonRPC string         `json:"jsonrpc"`
	Method  string         `json:"method"`
	Params  []define.Param `json:"params"`
	ID      string         `json:"id"`
}

func NewBlockRazor() *BlockRazor {
	return &BlockRazor{HttpAddress: "https://virginia.builder.blockrazor.io", PublicAddress: "0x1266C6bE60392A8Ff346E8d5ECCd3E69dD9c5F20"}
}

func (blockRazor *BlockRazor) GetPublicAddress() common.Address {
	return common.HexToAddress(blockRazor.PublicAddress)
}

func (blockRazor *BlockRazor) SendBundle(param define.Param, hash common.Hash) {
	param.MaxBlockNumber = cast.ToUint64(param.BlockNumber)
	param.BlockNumber = "" // For compatibility with smith

	bd := &BuilderData{}
	bd.ID = "1"
	bd.JsonRPC = "2.0"
	bd.Method = "eth_sendBundle"
	bd.Params = []define.Param{param}

	data, _ := json.Marshal(bd)

	Zap.Info("blockRazor send", zap.Any("hash", hash), zap.Any("req", bd))

	client := &http.Client{
		Timeout: 3 * time.Second,
	}
	httpReq, err := http.NewRequest("POST", blockRazor.HttpAddress, bytes.NewBuffer(data))
	if err != nil {
		log.Error("Error creating blockrazor request:", err)
		return
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "YWCVWDpexIVHRIDshlrH8Ohfv8gVCOEmHBbvH38DEkD3TIWWVGEiIUBYJUTnuqYb5ECf1NwssJoIB6UG4jnPmJCFMTFJC4G8")

	resp, err := client.Do(httpReq)
	if err != nil {
		log.Error("send to blockrazor builder error:", err)
		return
	}
	defer resp.Body.Close()

	// 读取响应体
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		log.Error("receive blockrazor builder resp body error:", err)
		return
	}
	Zap.Info(fmt.Sprintf("blockrazor builder resp[%v]:%s", hash, string(body)))
	str := string(body)
	if strings.Contains(str, "error") {
		BlockRazorFailureGauge.Inc(1)
		if strings.Contains(str, "effective bundle gas price too low") {
			BlockRazorLowFailureGauge.Inc(1)
		}
	}
}

func (blockRazor *BlockRazor) SendRawPrivateTransaction(txHex string, bundleHash common.Hash) {
	bd := &map[string]any{
		"jsonrpc": "2.0",
		"id":      "1",
		"method":  "eth_sendPrivateTransaction",
		"params":  []string{txHex},
	}

	data, _ := json.Marshal(bd)

	Zap.Info(" blockrazor send private tx:", zap.Any("bundleHash", bundleHash), zap.Any("tx", txHex))

	client := &http.Client{
		Timeout: 3 * time.Second,
	}
	httpReq, err := http.NewRequest("POST", blockRazor.HttpAddress, bytes.NewBuffer(data))
	if err != nil {
		log.Error("Error creating blockrazor request private tx:", err)
		return
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := client.Do(httpReq)
	if err != nil {
		log.Error("send to blockrazor builder private tx error:", err)
		return
	}
	defer resp.Body.Close()

	// 读取响应体
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		log.Error("receive blockrazor builder private tx resp body error:", err)
		return
	}
	Zap.Info(fmt.Sprintf("blockrazor builder private tx resp[%v]:%s", bundleHash, string(body)))
}
