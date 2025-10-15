package client

import (
	"bsc-rpc-client/model"
	"bsc-rpc-client/zap_logger"
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"time"
)

const (
	SendMevBundleEndpointName      = "eth_sendMevBundle"
	SendRawTransactionEndpointName = "eth_sendRawTransaction"
	ResetHeaderEndpointName        = "eth_resetHeader"
)

// 与服务端完全一致的结构
type JSONRPCRequest struct {
	JSONRPC string            `json:"jsonrpc"`
	ID      any               `json:"id"`
	Method  string            `json:"method"`
	Params  []json.RawMessage `json:"params"`
}

type JSONRPCResponse struct {
	JSONRPC string           `json:"jsonrpc"`
	ID      any              `json:"id"`
	Result  *json.RawMessage `json:"result,omitempty"`
	Error   *JSONRPCError    `json:"error,omitempty"`
}

type JSONRPCError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
	Data    *any   `json:"data,omitempty"`
}

// JSONRPCClient 封装结构体
type JSONRPCClient struct {
	url    string
	client *http.Client
}

var GlobalRpcClient = NewJSONRPCClient("http://127.0.0.1:8080")

// NewJSONRPCClient 创建客户端实例
func NewJSONRPCClient(url string) *JSONRPCClient {
	return &JSONRPCClient{
		url: url,
		client: &http.Client{
			Timeout: 10 * time.Second,
		},
	}
}

// Call 发起 JSON-RPC 调用
func (c *JSONRPCClient) Call(method string, params interface{}) (*JSONRPCResponse, error) {
	rawParams, err := encodeParams(params)
	if err != nil {
		return nil, fmt.Errorf("encode params error: %w", err)
	}

	req := JSONRPCRequest{
		JSONRPC: "2.0",
		ID:      1,
		Method:  method,
		Params:  rawParams,
	}

	data, _ := json.Marshal(req)
	resp, err := c.client.Post(c.url, "application/json", bytes.NewReader(data))
	if err != nil {
		return nil, fmt.Errorf("post error: %w", err)
	}
	defer resp.Body.Close()

	var rpcResp JSONRPCResponse
	if err := json.NewDecoder(resp.Body).Decode(&rpcResp); err != nil {
		return nil, fmt.Errorf("decode response error: %w", err)
	}

	if rpcResp.Error != nil {
		return &rpcResp, fmt.Errorf("RPC error: %d - %s", rpcResp.Error.Code, rpcResp.Error.Message)
	}
	return &rpcResp, nil
}

// encodeParams 将任意参数类型转为 []json.RawMessage
func encodeParams(params interface{}) ([]json.RawMessage, error) {
	if params == nil {
		return []json.RawMessage{}, nil
	}

	b, err := json.Marshal(params)
	if err != nil {
		return nil, err
	}
	return []json.RawMessage{b}, nil
}

// SendMevBundle 发送一组 MEV bundle
func (c *JSONRPCClient) SendMevBundle(args *model.SendMevBundleArgs) (*model.SendMevBundleResponse, error) {
	resp, err := c.Call(SendMevBundleEndpointName, args)
	if err != nil {
		return nil, err
	}
	var result model.SendMevBundleResponse
	json.Unmarshal(*resp.Result, &result)
	return &result, nil
}

// SendRawTransaction 发送原始交易
func (c *JSONRPCClient) SendRawTransaction(args *model.SendRawTransactionArgs) (*model.SendRawTransactionResponse, error) {
	resp, err := c.Call(SendRawTransactionEndpointName, args)
	if err != nil {
		return nil, err
	}
	var result model.SendRawTransactionResponse
	json.Unmarshal(*resp.Result, &result)
	return &result, nil
}

// ResetHeader 重置区块头
func (c *JSONRPCClient) ResetHeader(blockNumber uint64) error {
	resp, err := c.Call(ResetHeaderEndpointName, blockNumber)
	if err != nil {
		return fmt.Errorf("ResetHeader 调用失败: %w", err)
	}
	var result model.ResetHeaderResponse
	json.Unmarshal(*resp.Result, &result)
	zap_logger.Zap.Info(fmt.Sprintf("ResetHeader result header number: %d", result.HeaderNumber))
	return nil
}
