package smith

import (
	"bytes"
	"encoding/json"
	"fmt"
	"github.com/ethereum/go-ethereum-test/push/define"
	"github.com/ethereum/go-ethereum-test/zap_logger"
	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/log"
	"go.uber.org/zap"
	"io"
	"net/http"
	"sync/atomic"
	"time"
)

type Smith struct {
	HttpAddress   []string
	Number        atomic.Int64
	PublicAddress string
}

type BuilderData struct {
	JsonRPC string         `json:"jsonrpc"`
	Method  string         `json:"method"`
	Params  []define.Param `json:"params"`
	ID      string         `json:"id"`
}

func NewSmith() *Smith {
	return &Smith{HttpAddress: []string{"https://fastbundle-us.blocksmith.org/", "https://fastbundle-eu.blocksmith.org/", "https://fastbundle-ap.blocksmith.org/"}, PublicAddress: "0x0000000000007592b04bB3BB8985402cC37Ca224"}
}

func (smith *Smith) GetPublicAddress() common.Address {
	return common.HexToAddress(smith.PublicAddress)
}

func (smith *Smith) SendBundle(param define.Param, hash common.Hash) {
	param.MaxBlockNumber = 0 // // For compatibility with blockrazor

	bd := &BuilderData{}
	bd.ID = "1"
	bd.JsonRPC = "2.0"
	bd.Method = "eth_sendBundle"
	bd.Params = []define.Param{param}

	data, _ := json.Marshal(bd)
	cost := time.Now().Sub(param.ArrivalTime).Microseconds()

	zap_logger.Zap.Info("[smith-send]", zap.Any("hash", hash), zap.Any("cost", cost), zap.Any("txs", len(param.Txs)), zap.Any("userId", param.UserId))
	time.Sleep(20 * time.Millisecond)
	return

	client := &http.Client{
		Timeout: 10 * time.Second,
	}
	addr := smith.HttpAddress[smith.Number.Add(1)%int64(len(smith.HttpAddress))]
	httpReq, err := http.NewRequest("POST", addr, bytes.NewBuffer(data))
	if err != nil {
		log.Error("Error creating club48 request:", err)
		return
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Basic ZjMyM2EzMWQtMmMwNy0zMGNhLTg0ZjEtN2I3MGI1OWRkY2VkOjNkYzBmYWE4NDJmOWE0ZmM5MzViYTRhZjU4Y2U3ODcz")

	resp, err := client.Do(httpReq)
	if err != nil {
		log.Error("send to smith builder error:", err)
		return
	}
	defer resp.Body.Close()

	// 读取响应体
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		log.Error("receive smith builder resp body error:", err)
		return
	}
	zap_logger.Zap.Info(fmt.Sprintf("smith builder resp[%v]:%s", hash, string(body)))
}

func (smith *Smith) SendRawPrivateTransaction(txHex string, bundleHash common.Hash) {
	fmt.Println(time.Now(), " smith send private tx:", txHex)

	bd := &map[string]any{
		"jsonrpc": "2.0",
		"id":      "1",
		"method":  "eth_sendPrivateRawTransaction",
		"params":  []string{txHex},
	}

	data, _ := json.Marshal(bd)

	client := &http.Client{
		Timeout: 3 * time.Second,
	}
	addr := smith.HttpAddress[smith.Number.Add(1)%int64(len(smith.HttpAddress))]
	httpReq, err := http.NewRequest("POST", addr, bytes.NewBuffer(data))
	if err != nil {
		log.Error("Error creating club48 request private tx:", err)
		return
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := client.Do(httpReq)
	if err != nil {
		log.Error("send to smith builder private tx error:", err)
		return
	}
	defer resp.Body.Close()

	// 读取响应体
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		log.Error("receive smith builder private tx resp body error:", err)
		return
	}
	fmt.Println(time.Now(), " smith builder private tx resp:", string(body))
}
