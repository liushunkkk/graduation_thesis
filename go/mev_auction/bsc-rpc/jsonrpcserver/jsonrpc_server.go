// Package jsonrpcserver allows exposing functions like:
// func Foo(context, int) (int, error)
// as a JSON RPC methods
//
// This implementation is similar to the one in go-ethereum, but the idea is to eventually replace it as a default
// JSON RPC server implementation in Flasbhots projects and for this we need to reimplement some of the quirks of existing API.
package jsonrpcserver

import (
	"context"
	"encoding/json"
	"net/http"
)

var (
	CodeParseError     = -32700
	CodeInvalidRequest = -32600
	CodeMethodNotFound = -32601
	CodeInvalidParams  = -32602
	CodeInternalError  = -32603
	CodeCustomError    = -32000
)

const (
	maxOriginIDLength = 255
)

type (
	highPriorityKey struct{}
	signerKey       struct{}
	originKey       struct{}
)

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

// Handler 自己封装的一个 handler，他可以根据请求体的内容，去掉用 map 中的对应的函数
// 他实现了 ServeHTTP 接口，就可以注册到 http 中，提供 http 接口服务
type Handler struct {
	methods map[string]methodHandler
}

type Methods map[string]interface{}

// NewHandler creates JSONRPC http.Handler from the map that maps method names to method functions
// each method function must:
// - have context as a first argument
// - return error as a last argument
// - have argument types that can be unmarshalled from JSON
// - have return types that can be marshalled to JSON
func NewHandler(methods Methods) (*Handler, error) {
	m := make(map[string]methodHandler)
	for name, fn := range methods {
		method, err := getMethodTypes(fn)
		if err != nil {
			return nil, err
		}
		m[name] = method
	}
	return &Handler{
		methods: m,
	}, nil
}

func writeJSONRPCError(w http.ResponseWriter, id any, code int, msg string) {
	res := JSONRPCResponse{
		JSONRPC: "2.0",
		ID:      id,
		Result:  nil,
		Error: &JSONRPCError{
			Code:    code,
			Message: msg,
			Data:    nil,
		},
	}
	if err := json.NewEncoder(w).Encode(res); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
}

// ServeHTTP 自实现的http处理逻辑
// 读取req中的body信息 -> JSONRPCRequest，然后判断相关字段是否服务要求
// 然后 mev share 也对请求头中的几个字段做了相关要求，需要带有signer和origin信息，
// 最后就是读取 JSONRPCRequest 中的 Method 字段，通过反射调用他得到结果
// 最后封装成 JSONRPCResponse 返回给 searcher
func (h *Handler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	// read request
	var req JSONRPCRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSONRPCError(w, nil, CodeParseError, err.Error())
		return
	}

	if req.JSONRPC != "2.0" {
		writeJSONRPCError(w, req.ID, CodeParseError, "invalid jsonrpc version")
		return
	}
	if req.ID != nil {
		// id must be string or number
		switch req.ID.(type) {
		case string, int, int8, int16, int32, int64, uint, uint8, uint16, uint32, uint64, float32, float64:
		default:
			writeJSONRPCError(w, req.ID, CodeParseError, "invalid id type")
		}
	}

	ctx := context.Background()

	// get method
	method, ok := h.methods[req.Method]
	if !ok {
		writeJSONRPCError(w, req.ID, CodeMethodNotFound, "method not found")
		return
	}

	// call method
	result, err := method.call(ctx, req.Params)
	if err != nil {
		writeJSONRPCError(w, req.ID, CodeCustomError, err.Error())
		return
	}

	marshaledResult, err := json.Marshal(result)
	if err != nil {
		writeJSONRPCError(w, req.ID, CodeInternalError, err.Error())
		return
	}

	// write response
	rawMessageResult := json.RawMessage(marshaledResult)
	res := JSONRPCResponse{
		JSONRPC: "2.0",
		ID:      req.ID,
		Result:  &rawMessageResult,
		Error:   nil,
	}
	if err := json.NewEncoder(w).Encode(res); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
}
