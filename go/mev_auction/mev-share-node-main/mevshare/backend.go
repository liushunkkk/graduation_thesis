package mevshare

import (
	"context"
	"encoding/json"
	"github.com/ethereum/go-ethereum/common/hexutil"
	"github.com/ethereum/go-ethereum/core/types"
	"github.com/flashbots/mev-share-node/zap_logger"
	"go.uber.org/zap"
	"math/big"
	"math/rand"
	"time"

	"github.com/ethereum/go-ethereum/common"
	"github.com/redis/go-redis/v9"
	"github.com/ybbus/jsonrpc/v3"
)

type HintBackend interface {
	NotifyHint(ctx context.Context, hint *Hint) error
}

type BuilderBackend interface {
	String() string
	SendMatchedShareBundle(ctx context.Context, bundle *SendMevBundleArgs) error
	CancelBundleByHash(ctx context.Context, hash common.Hash) error
}

// SimulationBackend is an interface for simulating transactions
// There should be one simulation backend per worker node
type SimulationBackend interface {
	SimulateBundle(ctx context.Context, bundle *SendMevBundleArgs, aux *SimMevBundleAuxArgs) (*SimMevBundleResponse, error)
}

type JSONRPCSimulationBackend struct {
}

// NewJSONRPCSimulationBackend 创建 jsonrpc 的连客户端连接
func NewJSONRPCSimulationBackend(url string) *JSONRPCSimulationBackend {
	return &JSONRPCSimulationBackend{
		// todo here use optsx
	}
}

func replaceRevertModeForSimulation(bundle *SendMevBundleArgs) *SendMevBundleArgs {
	newB := &SendMevBundleArgs{
		Version:         bundle.Version,
		ReplacementUUID: "",
		Inclusion:       bundle.Inclusion,
		Body:            nil,
		Validity:        bundle.Validity,
		Privacy:         bundle.Privacy,
		Metadata:        bundle.Metadata,
	}

	for _, el := range bundle.Body {
		var newEl MevBundleBody
		if el.Tx != nil {
			if el.RevertMode == "drop" {
				newEl.CanRevert = true
				newEl.Tx = el.Tx
			} else {
				newEl.CanRevert = el.CanRevert
				newEl.Tx = el.Tx
			}
		}
		if el.Bundle != nil {
			newEl.Bundle = replaceRevertModeForSimulation(el.Bundle)
		}
		newB.Body = append(newB.Body, newEl)
	}
	return newB
}

// SimulateBundle 调用别的服务的 mev_simBundle rpc接口
func (b *JSONRPCSimulationBackend) SimulateBundle(ctx context.Context, bundle *SendMevBundleArgs, aux *SimMevBundleAuxArgs) (*SimMevBundleResponse, error) {
	var result SimMevBundleResponse
	// we need a hack here until mev_simBundle supports revertMode, we will treat revertMode=drop as canRevert=true so that bundle passes simulation
	_ = replaceRevertModeForSimulation(bundle)
	if len(bundle.Body) >= 2 {
		time.Sleep(30 * time.Millisecond)
		result.Profit = hexutil.Big(*big.NewInt(int64(200 + rand.Intn(100))))
	} else {
		time.Sleep(20 * time.Millisecond)
		result.Profit = hexutil.Big(*big.NewInt(int64(100 + rand.Intn(100))))
	}
	result.Success = true
	result.StateBlock = hexutil.Uint64(CurrentHeader.Number.Uint64())
	result.GasUsed = hexutil.Uint64(rand.Uint64())
	result.MevGasPrice = hexutil.Big(*big.NewInt(int64(rand.Intn(1000000))))
	if len(bundle.Body) > 1 {
		result.BodyLogs = []SimMevBodyLogs{
			{
				BundleLogs: make([]SimMevBodyLogs, 0),
			},
			{
				TxLogs: []*types.Log{
					{
						Address: common.HexToAddress("0x0000000000000000000000000000000000000000"),
					},
				},
			},
		}
		if len(bundle.Body[0].Bundle.Body) > 1 {
			result.BodyLogs[0].BundleLogs = []SimMevBodyLogs{
				{
					BundleLogs: []SimMevBodyLogs{
						{
							TxLogs: []*types.Log{
								{
									Address: common.HexToAddress("0x0000000000000000000000000000000000000000"),
								},
							},
						},
					},
				},
				{
					TxLogs: []*types.Log{
						{
							Address: common.HexToAddress("0x0000000000000000000000000000000000000000"),
						},
					},
				},
			}
		} else {
			result.BodyLogs[0].BundleLogs = []SimMevBodyLogs{
				{
					TxLogs: []*types.Log{
						{
							Address: common.HexToAddress("0x0000000000000000000000000000000000000000"),
						},
					},
				},
			}
		}
	} else {
		result.BodyLogs = []SimMevBodyLogs{
			{
				TxLogs: []*types.Log{
					{
						Address: common.HexToAddress("0x0000000000000000000000000000000000000000"),
					},
				},
			},
		}
	}
	//err := b.client.CallFor(ctx, &result, "mev_simBundle", newBundle, aux)
	return &result, nil
}

type RedisHintBackend struct {
	client     *redis.Client
	pubChannel string
}

func NewRedisHintBackend(redisClient *redis.Client, pubChannel string) *RedisHintBackend {
	return &RedisHintBackend{
		client:     redisClient,
		pubChannel: pubChannel,
	}
}

func (b *RedisHintBackend) NotifyHint(ctx context.Context, hint *Hint) error {
	data, err := json.Marshal(hint)
	if err != nil {
		return err
	}
	zap_logger.Zap.Info("push to redis", zap.Any("channel", b.pubChannel))
	return b.client.Publish(ctx, b.pubChannel, data).Err()
}

// JSONRPCBuilder 下面这部分没有用到
type JSONRPCBuilder struct {
	url    string
	client jsonrpc.RPCClient
}

func NewJSONRPCBuilder(url string) *JSONRPCBuilder {
	return &JSONRPCBuilder{
		url:    url,
		client: jsonrpc.NewClient(url),
	}
}

func (b *JSONRPCBuilder) String() string {
	return b.url
}

func (b *JSONRPCBuilder) SendMatchedShareBundle(ctx context.Context, bundle *SendMevBundleArgs) error {
	res, err := b.client.Call(ctx, "mev_sendBundle", []*SendMevBundleArgs{bundle})
	if err != nil {
		return err
	}
	if res.Error != nil {
		return res.Error
	}
	return nil
}

func (b *JSONRPCBuilder) CancelBundleByHash(ctx context.Context, hash common.Hash) error {
	res, err := b.client.Call(ctx, "mev_cancelBundleByHash", []common.Hash{hash})
	if err != nil {
		return err
	}
	if res.Error != nil {
		return res.Error
	}
	return nil
}
