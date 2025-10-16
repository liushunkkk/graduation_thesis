package nodereal

import (
	"bytes"
	"encoding/json"
	"fmt"
	"github.com/ethereum/go-ethereum-test/push/define"
	"github.com/ethereum/go-ethereum-test/zap_logger"
	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/log"
	"github.com/spf13/cast"
	"go.uber.org/zap"
	"io"
	"net/http"
	"time"
)

type NodeReal struct {
	HttpAddress   string
	PublicAddress string
}

type BuilderData struct {
	JsonRPC string         `json:"jsonrpc"`
	Method  string         `json:"method"`
	Params  []define.Param `json:"params"`
	ID      string         `json:"id"`
}

func NewNodeReal() *NodeReal {
	return &NodeReal{HttpAddress: "https://bsc-mainnet.nodereal.io/v1/fe2c9a1d2bfe434e9549b37495548264", PublicAddress: ""}
}

func (nodereal *NodeReal) GetPublicAddress() common.Address {
	return common.HexToAddress(nodereal.PublicAddress)
}

func (nodereal *NodeReal) SendBundle(param define.Param, hash common.Hash) {
	param.MaxBlockNumber = cast.ToUint64(param.BlockNumber)
	param.BlockNumber = "" // For compatibility with smith

	bd := &BuilderData{}
	bd.ID = "1"
	bd.JsonRPC = "2.0"
	bd.Method = "eth_sendBundle"
	bd.Params = []define.Param{param}

	data, _ := json.Marshal(bd)
	cost := time.Now().Sub(param.ArrivalTime).Microseconds()

	zap_logger.Zap.Info("[nodereal-send]", zap.Any("hash", hash), zap.Any("cost", cost), zap.Any("txs", len(param.Txs)), zap.Any("userId", param.UserId))
	time.Sleep(20 * time.Millisecond)
	return

	client := &http.Client{
		Timeout: 3 * time.Second,
	}
	httpReq, err := http.NewRequest("POST", nodereal.HttpAddress, bytes.NewBuffer(data))
	if err != nil {
		log.Error("Error creating nodereal request:", err)
		return
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := client.Do(httpReq)
	if err != nil {
		log.Error("send to nodereal builder error:", err)
		return
	}
	defer resp.Body.Close()

	// 读取响应体
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		log.Error("receive nodereal builder resp body error:", err)
		return
	}
	zap_logger.Zap.Info(fmt.Sprintf("nodereal builder resp[%v]:%s", hash, string(body)))
}

func (nodereal *NodeReal) SendRawPrivateTransaction(txHash string, bundleHash common.Hash) {
}
