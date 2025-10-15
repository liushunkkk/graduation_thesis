package client

import (
	"bsc-rpc-client/model"
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
// params 可以是 map / slice / nil，会被自动编码为 []json.RawMessage
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

	switch v := params.(type) {
	case []interface{}:
		arr := make([]json.RawMessage, 0, len(v))
		for _, p := range v {
			b, err := json.Marshal(p)
			if err != nil {
				return nil, err
			}
			arr = append(arr, b)
		}
		return arr, nil
	case map[string]interface{}, string, int, float64, bool:
		b, err := json.Marshal(v)
		if err != nil {
			return nil, err
		}
		return []json.RawMessage{b}, nil
	default:
		b, err := json.Marshal(v)
		if err != nil {
			return nil, err
		}
		return []json.RawMessage{b}, nil
	}
}

// SendMevBundle 发送一组 MEV bundle
func (c *JSONRPCClient) SendMevBundle(args *model.SendMevBundleArgs) (*model.SendMevBundleResponse, error) {
	resp, err := c.Call(SendMevBundleEndpointName, []interface{}{args})
	if err != nil {
		return nil, err
	}
	var result model.SendMevBundleResponse
	json.Unmarshal(*resp.Result, &result)
	return &result, nil
}

// SendRawTransaction 发送原始交易
func (c *JSONRPCClient) SendRawTransaction(args *model.SendRawTransactionArgs) (*model.SendRawTransactionResponse, error) {
	resp, err := c.Call(SendRawTransactionEndpointName, []interface{}{args})
	if err != nil {
		return nil, err
	}
	var result model.SendRawTransactionResponse
	json.Unmarshal(*resp.Result, &result)
	return &result, nil
}

// ResetHeader 重置区块头
func (c *JSONRPCClient) ResetHeader(blockNumber uint64) error {
	resp, err := c.Call(ResetHeaderEndpointName, []interface{}{blockNumber})
	if err != nil {
		return fmt.Errorf("ResetHeader 调用失败: %w", err)
	}
	var result model.ResetHeaderResponse
	json.Unmarshal(*resp.Result, &result)
	fmt.Println("ResetHeader result header number:", result.HeaderNumber)
	return nil
}
