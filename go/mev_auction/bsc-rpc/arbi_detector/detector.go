package arbi_detector

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"time"
)

// --------------------------
// 定义与 Flask 接口对应的结构体
// --------------------------

// ArbiRequest 对应 Flask 的 ArbiRequest（单个请求参数）
type ArbiRequest struct {
	TxHash      string `json:"txHash"`      // 交易Hash（与Python结构体字段名一致，注意JSON标签大小写）
	TxJson      string `json:"txJson"`      // 交易json序列化数据
	ReceiptJson string `json:"receiptJson"` // 收据json序列化数据
}

// ArbiResult 对应 Flask 的 ArbiResult（单个响应结果）
type ArbiResult struct {
	Success          bool    `json:"success"`          // 是否处理成功
	TxHash           string  `json:"txHash"`           // 对应的交易Hash
	ArbitragePercent float64 `json:"arbitragePercent"` // 套利百分比
}

// BatchArbiRequest 批量请求参数
type BatchArbiRequest []ArbiRequest

// BatchArbiResult 对应 Flask 的 BatchArbiResult（批量响应结果）
type BatchArbiResult struct {
	SuccessItems int          `json:"successItems"` // 成功处理的项目数
	Results      []ArbiResult `json:"results"`      // 批量处理的详细结果
}

// DetectSingleArbi 发送单个套利检测请求（/detectArbi）
func DetectSingleArbi(req ArbiRequest) (*ArbiResult, error) {
	// 1. 将请求结构体序列化为 JSON 字节流
	jsonData, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("JSON序列化失败: %v", err)
	}

	// 2. 创建 HTTP POST 请求（设置请求体、Content-Type 为 JSON）
	httpReq, err := http.NewRequest("POST", "http://localhost:5000/detectArbi", bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("创建请求失败: %v", err)
	}
	httpReq.Header.Set("Content-Type", "application/json") // 必须设置，否则Flask无法解析JSON

	// 3. 发送请求（设置超时时间为5秒，避免无限等待）
	client := &http.Client{Timeout: 5 * time.Second}
	resp, err := client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("发送请求失败: %v", err)
	}
	defer resp.Body.Close() // 确保响应体被关闭，避免资源泄漏

	// 4. 检查响应状态码（200表示成功，400为参数错误，其他为服务端错误）
	if resp.StatusCode != http.StatusOK {
		// 读取错误响应的内容，便于排查问题
		errBody, _ := ioutil.ReadAll(resp.Body)
		return nil, fmt.Errorf("请求失败 [状态码: %d], 错误信息: %s", resp.StatusCode, string(errBody))
	}

	// 5. 解析响应体为 ArbiResult 结构体
	var result ArbiResult
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("解析响应失败: %v", err)
	}

	return &result, nil
}

// DetectBatchArbi 发送批量套利检测请求（/batchDetectArbi）
func DetectBatchArbi(req BatchArbiRequest) (*BatchArbiResult, error) {
	// 逻辑与单个请求类似，仅序列化和响应解析的结构体不同
	jsonData, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("JSON序列化失败: %v", err)
	}

	httpReq, err := http.NewRequest("POST", "http://localhost:5000/batchDetectArbi", bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("创建请求失败: %v", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	client := &http.Client{Timeout: 10 * time.Second} // 批量请求超时时间可适当延长
	resp, err := client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("发送请求失败: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		errBody, _ := ioutil.ReadAll(resp.Body)
		return nil, fmt.Errorf("请求失败 [状态码: %d], 错误信息: %s", resp.StatusCode, string(errBody))
	}

	var result BatchArbiResult
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("解析响应失败: %v", err)
	}

	return &result, nil
}
