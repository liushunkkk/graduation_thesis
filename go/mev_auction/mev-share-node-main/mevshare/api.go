package mevshare

import (
	"context"
	"errors"
	"math/rand"
	"time"

	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/common/hexutil"
	"github.com/ethereum/go-ethereum/common/lru"
	"github.com/ethereum/go-ethereum/core/types"
	"github.com/flashbots/mev-share-node/jsonrpcserver"
	"go.uber.org/zap"
	"golang.org/x/time/rate"
)

var (
	ErrInvalidInclusion      = errors.New("invalid inclusion")
	ErrInvalidBundleBodySize = errors.New("invalid bundle body size")
	ErrInvalidBundleBody     = errors.New("invalid bundle body")
	ErrBackrunNotFound       = errors.New("backrun not found")
	ErrBackrunInvalidBundle  = errors.New("backrun invalid bundle")
	ErrBackrunInclusion      = errors.New("backrun invalid inclusion")

	ErrInternalServiceError = errors.New("mev-share service error")

	simBundleTimeout    = 500 * time.Millisecond
	cancelBundleTimeout = 3 * time.Second
	bundleCacheSize     = 1000
)

type SimScheduler interface {
	ScheduleBundleSimulation(ctx context.Context, bundle *SendMevBundleArgs, highPriority bool) error
}

type BundleStorage interface {
	GetBundleByMatchingHash(ctx context.Context, hash common.Hash) (*SendMevBundleArgs, error)
	CancelBundleByHash(ctx context.Context, hash common.Hash, signer common.Address) error
}

type EthClient interface {
	BlockNumber(ctx context.Context) (uint64, error)
}

type API struct {
	log *zap.Logger

	scheduler     SimScheduler
	bundleStorage BundleStorage
	signer        types.Signer
	simBackends   []SimulationBackend
	// rate.NewLimiter(simRateLimit, 1)
	// 	- r Limit：定义速率限制器的速率，表示每秒允许发生的事件数。它是 Limit 类型，这通常是一个浮点数。
	//	- b int：定义突发容量，表示可以在短时间内突发处理的最大事件数。
	// MEV_SIM_BUNDLE_RATE_LIMIT 默认是 5，也就是同时只允许每秒最多5个
	simRateLimiter *rate.Limiter
	builders       BuildersBackend

	knownBundleCache *lru.Cache[common.Hash, SendMevBundleArgs]
}

func NewAPI(
	log *zap.Logger,
	scheduler SimScheduler, signer types.Signer,
	simBackends []SimulationBackend, simRateLimit rate.Limit, builders BuildersBackend,
	sbundleValidDuration time.Duration,
) *API {
	return &API{
		log: log,

		scheduler:        scheduler,
		signer:           signer,
		simBackends:      simBackends,
		simRateLimiter:   rate.NewLimiter(simRateLimit, 1),
		builders:         builders,
		knownBundleCache: lru.NewCache[common.Hash, SendMevBundleArgs](bundleCacheSize),
	}
}

func findAndReplace(strs []common.Hash, old, replacer common.Hash) bool {
	var found bool
	for i, str := range strs {
		if str == old {
			strs[i] = replacer
			found = true
		}
	}
	return found
}

// SendBundle mev_sendBundle 的处理逻辑，
// 只是做了一些校验，设置了一些值，就加入模拟队列了
// 返回样例
//
//	{
//	 "bundleHash": "0x7d6e491ab67aee5f4b75321c936bf05664d2d9b234fd67083e46bd43bb42f383"
//	}
func (m *API) SendBundle(ctx context.Context, bundle SendMevBundleArgs) (_ SendMevBundleResponse, err error) {
	logger := m.log
	startAt := time.Now()

	// bundle hash
	hash, hasUnmatchedHash, err := ValidateBundle(&bundle, 0, m.signer)
	if err != nil {
		logger.Warn("failed to validate bundle", zap.Error(err))
		return SendMevBundleResponse{}, err
	}
	logger.Debug("received bundle", zap.String("bundle", hash.String()), zap.Time("receivedAt", startAt), zap.Int64("timestamp", startAt.Unix()))

	if oldBundle, ok := m.knownBundleCache.Get(hash); ok {
		if !newerInclusion(&oldBundle, &bundle) {
			logger.Debug("bundle already known, ignoring", zap.String("hash", hash.Hex()))
			return SendMevBundleResponse{hash}, nil
		}
	}
	m.knownBundleCache.Add(hash, bundle)

	// 拿到请求头中的信息
	signerAddress := jsonrpcserver.GetSigner(ctx)
	origin := jsonrpcserver.GetOrigin(ctx)
	if bundle.Metadata == nil {
		bundle.Metadata = &MevBundleMetadata{}
	}
	bundle.Metadata.Signer = signerAddress
	bundle.Metadata.ReceivedAt = hexutil.Uint64(uint64(time.Now().UnixMicro()))
	bundle.Metadata.OriginID = origin
	bundle.Metadata.Prematched = !hasUnmatchedHash

	// unmatched 表示不支持 simulation，包含未签名交易
	if hasUnmatchedHash {
		bundle.Body[0].Hash = nil
		// send 90 % of the refund to the unmatched bundle or the suggested refund if set
		refundPercent := RefundPercent
		bundle.Validity.Refund = []RefundConstraint{{0, refundPercent}}
		MergePrivacyBuilders(&bundle)
	}

	//highPriority := jsonrpcserver.GetPriority(ctx)
	// 所有的 metadata 设置好了，主要是用于模拟
	// 加入交易模拟队列
	err = m.scheduler.ScheduleBundleSimulation(ctx, &bundle, false)
	if err != nil {
		logger.Error("Failed to schedule bundle simulation", zap.Error(err))
		return SendMevBundleResponse{}, ErrInternalServiceError
	}

	return SendMevBundleResponse{
		BundleHash: hash,
	}, nil
}

// SimBundle mev_simBundle api的实现逻辑
// uses a new bundle format to simulate matched bundles on MEV-Share
//
//	返回样例：{
//	 "success": true,
//	 "stateBlock": "0x8b8da8",
//	 "mevGasPrice": "0x74c7906005",
//	 "profit": "0x4bc800904fc000",
//	 "refundableValue": "0x4bc800904fc000",
//	 "gasUsed": "0xa620",
//	 "logs": [{}, {}]
//	}
func (m *API) SimBundle(ctx context.Context, bundle SendMevBundleArgs, aux SimMevBundleAuxArgs) (_ *SimMevBundleResponse, err error) {

	if len(m.simBackends) == 0 {
		return nil, ErrInternalServiceError
	}
	ctx, cancel := context.WithTimeout(ctx, simBundleTimeout)
	defer cancel()

	simTimeout := int64(simBundleTimeout / time.Millisecond)
	aux.Timeout = &simTimeout

	err = m.simRateLimiter.Wait(ctx)
	if err != nil {
		return nil, err
	}

	// select random backend
	// 随机选择一个提供商进行模拟，他没有节点数据，他不可能进行模拟的
	idx := rand.Intn(len(m.simBackends)) //nolint:gosec
	backend := m.simBackends[idx]
	return backend.SimulateBundle(ctx, &bundle, &aux)
}
