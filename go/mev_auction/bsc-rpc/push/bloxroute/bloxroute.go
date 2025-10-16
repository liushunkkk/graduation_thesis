package bloxroute

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
	"time"
)

type Bloxroute struct {
	HttpAddress   string
	PublicAddress string
}

type BuilderData struct {
	JsonRPC string `json:"jsonrpc"`
	Method  string `json:"method"`
	Params  Req    `json:"params"`
	ID      string `json:"id"`
}

func NewBloxRoute() *Bloxroute {
	return &Bloxroute{HttpAddress: "https://mev.api.blxrbdn.com", PublicAddress: "0x74c5F8C6ffe41AD4789602BDB9a48E6Cad623520"}
}

func (bloxroute *Bloxroute) GetPublicAddress() common.Address {
	return common.HexToAddress(bloxroute.PublicAddress)
}

type Req struct {
	Transaction       []string `json:"transaction"`
	BlockNumber       string   `json:"block_number,omitempty"`
	RevertingTxHashes []string `json:"reverting_hashes"`
	BlockchainNetwork string   `json:"blockchain_network"`
}

func (bloxroute *Bloxroute) SendBundle(param define.Param, hash common.Hash) {
	req := Req{
		Transaction:       param.Txs,
		BlockNumber:       param.BlockNumber,
		RevertingTxHashes: param.RevertingTxHashes,
		BlockchainNetwork: "BSC-Mainnet",
	}

	bd := &BuilderData{}
	bd.ID = "1"
	bd.JsonRPC = "2.0"
	bd.Method = "blxr_submit_bundle"
	bd.Params = req

	data, _ := json.Marshal(bd)
	cost := time.Now().Sub(param.ArrivalTime).Microseconds()

	zap_logger.Zap.Info("[bloxroute-send]", zap.Any("hash", hash), zap.Any("cost", cost), zap.Any("txs", len(param.Txs)), zap.Any("userId", param.UserId))
	time.Sleep(20 * time.Millisecond)
	return

	client := &http.Client{
		Timeout: 3 * time.Second,
	}
	httpReq, err := http.NewRequest("POST", bloxroute.HttpAddress, bytes.NewBuffer(data))
	if err != nil {
		log.Error("Error creating blockrazor request:", err)
		return
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "NTkwY2RjZGItNDdmZS00NzQ0LWI3MjItNDMyZjI0YmQ3ZDkzOmIwZWI2ZjMyYjI2ZWVkYTdhMmQ3OTQ5YjBlNTMyODIw")

	resp, err := client.Do(httpReq)
	if err != nil {
		log.Error("send to blockrazor builder error:", err)
		return
	}
	defer resp.Body.Close()

	// 读取响应体
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		log.Error("receive bloxroute builder resp body error:", err)
		return
	}
	zap_logger.Zap.Info(fmt.Sprintf("bloxroute builder resp[%v]:%s", hash, string(body)))
}

func (bloxroute *Bloxroute) SendRawPrivateTransaction(txHex string, bundleHash common.Hash) {
	fmt.Println(time.Now(), " bloxroute send private tx:", txHex)

	bd := &map[string]any{
		"jsonrpc": "2.0",
		"id":      "1",
		"method":  "blxr_tx",
		"params": map[string]string{
			"transaction": txHex[2:],
		},
	}

	data, _ := json.Marshal(bd)

	client := &http.Client{
		Timeout: 3 * time.Second,
	}
	httpReq, err := http.NewRequest("POST", bloxroute.HttpAddress, bytes.NewBuffer(data))
	if err != nil {
		log.Error("Error creating bloxroute request private tx:", err)
		return
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := client.Do(httpReq)
	if err != nil {
		log.Error("send to bloxroute builder private tx error:", err)
		return
	}
	defer resp.Body.Close()

	// 读取响应体
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		log.Error("receive bloxroute builder private tx resp body error:", err)
		return
	}
	fmt.Println(time.Now(), " bloxroute builder private tx resp:", string(body))
}
