package scutum

import (
	"context"
	"errors"
	"fmt"
	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/core"
	"github.com/ethereum/go-ethereum/core/state"
	"github.com/ethereum/go-ethereum/core/types"
	"github.com/ethereum/go-ethereum/core/vm"
	"github.com/ethereum/go-ethereum/eth/tracers"
	"github.com/ethereum/go-ethereum/eth/tracers/logger"
	"github.com/ethereum/go-ethereum/internal/ethapi"
	"github.com/ethereum/go-ethereum/invalid_tx"
	"github.com/ethereum/go-ethereum/params"
	"github.com/ethereum/go-ethereum/portal"
	"github.com/ethereum/go-ethereum/portal/zrpc_client/model/rpc_portal/pb/rpcpb"
	"github.com/ethereum/go-ethereum/rpc"
	"github.com/holiman/uint256"
	"github.com/spf13/cast"
	"strconv"
	"time"
)

const (
	// defaultTraceTimeout is the amount of time a single transaction can execute
	// by default before being forcefully aborted.
	defaultTraceTimeout = 5 * time.Second

	// defaultTraceReexec is the number of blocks the tracer is willing to go back
	// and reexecute to produce missing historical state necessary to run a specific
	// trace.
	defaultTraceReexec = uint64(128)
)

var errTxNotFound = errors.New("transaction not found")

// Backend interface provides the common API services (that are provided by
// both full and light clients) with access to necessary functions.
type Backend interface {
	HeaderByHash(ctx context.Context, hash common.Hash) (*types.Header, error)
	GetReceipts(ctx context.Context, hash common.Hash) (types.Receipts, error)
	BlockByNumber(ctx context.Context, number rpc.BlockNumber) (*types.Block, error)
	BlockByHash(ctx context.Context, hash common.Hash) (*types.Block, error)
	GetTransaction(ctx context.Context, txHash common.Hash) (bool, *types.Transaction, common.Hash, uint64, uint64, error)
	ChainConfig() *params.ChainConfig
	StateAtTransaction(ctx context.Context, block *types.Block, txIndex int, reexec uint64) (*core.Message, vm.BlockContext, *state.StateDB, tracers.StateReleaseFunc, error)
}

// API is the collection of tracing APIs exposed over the private debugging endpoint.
type API struct {
	backend Backend
}

// NewAPI creates a new API definition for the tracing methods of the Ethereum service.
func NewAPI(backend Backend) *API {
	return &API{backend: backend}
}

// APIs return the collection of RPC services the tracer package offers.
func APIs(backend Backend) []rpc.API {
	// Append all the local APIs and return
	return []rpc.API{
		{
			Namespace: "scutum",
			Service:   NewAPI(backend),
		},
	}
}

// PushRpcInfo accept the portal's updated user information
func (api *API) PushRpcInfo(ctx context.Context, userInfo *rpcpb.GetRpcInfoResponse) error {
	if userInfo == nil {
		return errors.New("userInfo is nil")
	}
	if len(userInfo.Url) == 0 || len(userInfo.RpcId) == 0 {
		return errors.New("the Url or RpcId is none")
	}
	if portal.UserServer == nil {
		return errors.New("userServer is nil, error occurred in the node service")
	}
	//fmt.Println("accept userInfo from portal: ", portal.UserServer)
	portal.UserServer.RpcInfos.Store(userInfo.Url, userInfo)
	portal.UserServer.RpcInfos.Store(userInfo.RpcId, userInfo)
	return nil
}

// GetRpcTransaction returns the transaction information for the given transaction hash in the rpc system.
func (api *API) GetRpcTransaction(ctx context.Context, hash common.Hash) (*types.RpcTransactionResult, error) {
	found, tx, blockHash, blockNumber, index, err := api.backend.GetTransaction(ctx, hash)
	if err != nil {
		return nil, ethapi.NewTxIndexingError() // transaction is not fully indexed
	}
	if !found {
		if ret := invalid_tx.Server.Get(hash); ret != nil {
			return &types.RpcTransactionResult{Status: "0", Time: cast.ToString(ret.Time)}, nil
		}
		return nil, ethapi.NewTxIndexingError() // transaction is not existent or reachable
	}
	header, err := api.backend.HeaderByHash(ctx, blockHash)
	if err != nil {
		return nil, err
	}

	signer := types.MakeSigner(api.backend.ChainConfig(), header.Number, header.Time)
	from, _ := types.Sender(signer, tx)

	receipts, err := api.backend.GetReceipts(ctx, blockHash)
	if err != nil || uint64(len(receipts)) <= index {
		return &types.RpcTransactionResult{
			BlockHash:   blockHash.Hex(),
			BlockNumber: strconv.FormatUint(blockNumber, 10),
			GasPrice:    tx.GasPrice().String(),
			Time:        strconv.FormatUint(header.Time, 10),
			From:        from.Hex(),
			Status:      "2", // 2 pending
			GasUsed:     "0",
		}, nil
	}
	receipt := receipts[index]

	return &types.RpcTransactionResult{
		BlockHash:   blockHash.Hex(),
		BlockNumber: strconv.FormatUint(blockNumber, 10),
		GasPrice:    strconv.FormatUint(receipt.EffectiveGasPrice.Uint64(), 10),
		Time:        strconv.FormatUint(header.Time, 10),
		From:        from.Hex(),
		Status:      strconv.FormatUint(receipt.Status, 10), // 0 fail 1 success
		GasUsed:     strconv.FormatUint(receipt.GasUsed, 10),
		Data:        tx.Data(),
		Logs:        receipt.Logs,
	}, nil
}

// TxTraceResult is the result of a single transaction trace.
type TxTraceResult struct {
	TxHash common.Hash `json:"txHash"`           // transaction hash
	Result interface{} `json:"result,omitempty"` // Trace results produced by the tracer
	Error  string      `json:"error,omitempty"`  // Trace failure produced by the tracer
}

// BalanceChangeByTx 通过交易哈希查询余额变化。
// 参数:
// - ctx: 上下文，包含请求的上下文信息。
// - hash: 交易的哈希值。
// 返回值:
// - interface{}: 包含交易的余额变化结果。
// - error: 如果查询过程中发生错误，返回错误信息。
func (api *API) BalanceChangeByTx(ctx context.Context, hash common.Hash, address common.Address) (interface{}, error) {
	// 检索交易信息，包括是否找到、交易位置等信息。
	found, _, blockHash, blockNumber, index, err := api.backend.GetTransaction(ctx, hash)
	if err != nil {
		// 如果检索失败，返回错误。
		return nil, ethapi.NewTxIndexingError()
	}
	// 只支持已挖掘的交易
	if !found {
		return nil, errTxNotFound
	}
	// 块号为0的情况不应该发生。
	if blockNumber == 0 {
		return nil, errors.New("genesis is not traceable")
	}
	// 设置重新执行次数，默认值。
	reexec := defaultTraceReexec
	// 根据块号和哈希获取块信息。
	block, err := api.blockByNumberAndHash(ctx, rpc.BlockNumber(blockNumber), blockHash)
	if err != nil {
		// 如果获取块信息失败，返回错误。
		return nil, err
	}
	// 获取交易状态和虚拟机上下文等信息。
	msg, vmctx, statedb, release, err := api.backend.StateAtTransaction(ctx, block, int(index), reexec)
	if err != nil {
		// 如果获取状态失败，返回错误。
		return nil, err
	}
	// 释放资源的延迟操作。
	defer release()

	// 构建交易上下文。
	txctx := &tracers.Context{
		BlockHash:   blockHash,
		BlockNumber: block.Number(),
		TxIndex:     int(index),
		TxHash:      hash,
	}

	// 定义变量，包括追踪器、超时设置和交易上下文。
	var (
		tracer    tracers.Tracer
		timeout   = defaultTraceTimeout
		txContext = core.NewEVMTxContext(msg)
	)
	// 配置追踪设置。
	config := &tracers.TraceConfig{}
	// 默认追踪器是结构日志记录器。
	tracer = logger.NewStructLogger(config.Config)
	// 如果配置了特定的追踪器，使用它。
	if config.Tracer != nil {
		tracer, err = tracers.DefaultDirectory.New(*config.Tracer, txctx, config.TracerConfig)
		if err != nil {
			// 如果追踪器初始化失败，返回错误。
			return nil, err
		}
	}
	// 创建 EVM 环境。
	vmenv := vm.NewEVM(vmctx, txContext, statedb, api.backend.ChainConfig(), vm.Config{Tracer: tracer, NoBaseFee: true, EnablePreimageRecording: true})

	// 设置超时时间。
	if config.Timeout != nil {
		if timeout, err = time.ParseDuration(*config.Timeout); err != nil {
			// 如果解析超时时间失败，返回错误。
			return nil, err
		}
	}
	// 创建带有超时的上下文。
	deadlineCtx, cancel := context.WithTimeout(ctx, timeout)
	// 超时处理函数。
	go func() {
		<-deadlineCtx.Done()
		if errors.Is(deadlineCtx.Err(), context.DeadlineExceeded) {
			tracer.Stop(errors.New("execution timeout"))
			vmenv.Cancel()
		}
	}()
	// 确保超时结束。
	defer cancel()

	addressChange := &state.RpcBalanceChange{}
	if (address != common.Address{}) {
		beforeBalance := statedb.GetBalance(address)
		if beforeBalance != nil {
			addressChange.BeforeBalance = beforeBalance
		} else {
			addressChange.BeforeBalance = uint256.NewInt(0)
		}
	}

	// 清除状态数据库访问列表。
	statedb.SetTxContext(txctx.TxHash, txctx.TxIndex)
	// 应用交易消息。
	if _, err := core.ApplyMessage(vmenv, msg, new(core.GasPool).AddGas(msg.GasLimit)); err != nil {
		// 如果应用交易失败，返回错误。
		return nil, fmt.Errorf("tracing failed: %w", err)
	}

	if (address != common.Address{}) {
		afterBalance := statedb.GetBalance(address)
		if afterBalance != nil {
			addressChange.AfterBalance = afterBalance
		} else {
			addressChange.AfterBalance = uint256.NewInt(0)
		}
		addressChange.BalanceChange = state.SubtractUint256(addressChange.AfterBalance, addressChange.BeforeBalance)
		accounts := make(map[string][]*state.RpcBalanceChange)
		accounts[address.String()] = []*state.RpcBalanceChange{addressChange}
		result := struct {
			BalanceChanges map[string][]*state.RpcBalanceChange
		}{
			BalanceChanges: accounts,
		}
		// 返回交易追踪结果。
		return &TxTraceResult{
			TxHash: txctx.TxHash,
			Result: result,
			Error:  "",
		}, nil
	}

	// 获取余额变化账户信息。
	accounts := statedb.GetRpcBalanceChange()
	// 构建结果。
	result := struct {
		BalanceChanges map[string][]*state.RpcBalanceChange
	}{
		BalanceChanges: accounts,
	}
	// 返回交易追踪结果。
	return &TxTraceResult{
		TxHash: txctx.TxHash,
		Result: result,
		Error:  "",
	}, nil
}

// blockByNumber is the wrapper of the chain access function offered by the backend.
// It will return an error if the block is not found.
func (api *API) blockByNumber(ctx context.Context, number rpc.BlockNumber) (*types.Block, error) {
	block, err := api.backend.BlockByNumber(ctx, number)
	if err != nil {
		return nil, err
	}
	if block == nil {
		return nil, fmt.Errorf("block #%d not found", number)
	}
	return block, nil
}

// blockByHash is the wrapper of the chain access function offered by the backend.
// It will return an error if the block is not found.
func (api *API) blockByHash(ctx context.Context, hash common.Hash) (*types.Block, error) {
	block, err := api.backend.BlockByHash(ctx, hash)
	if err != nil {
		return nil, err
	}
	if block == nil {
		return nil, fmt.Errorf("block %s not found", hash.Hex())
	}
	return block, nil
}

func (api *API) blockByNumberAndHash(ctx context.Context, number rpc.BlockNumber, hash common.Hash) (*types.Block, error) {
	block, err := api.blockByNumber(ctx, number)
	if err != nil {
		return nil, err
	}
	if block.Hash() == hash {
		return block, nil
	}
	return api.blockByHash(ctx, hash)
}
