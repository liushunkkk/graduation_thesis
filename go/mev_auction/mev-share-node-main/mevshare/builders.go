package mevshare

import (
	"context"
	"errors"
	"strings"
	"sync"
	"time"

	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/common/hexutil"
	"github.com/ybbus/jsonrpc/v3"
	"go.uber.org/zap"
)

var ErrInvalidBuilder = errors.New("invalid builder specification")

type BuilderAPI uint8

const (
	BuilderAPIRefundRecipient BuilderAPI = iota
	BuilderAPIMevShareBeta1
	BuilderAPIMevShareBeta1Replacement

	OrderflowHeaderName = "x-orderflow-origin"
)

func parseBuilderAPI(api string) (BuilderAPI, error) {
	switch api {
	case "refund-recipient":
		return BuilderAPIRefundRecipient, nil
		// 现在这个是默认值
	case "v0.1":
		return BuilderAPIMevShareBeta1, nil
	case "v0.1-replacement":
		return BuilderAPIMevShareBeta1Replacement, nil
	default:
		return 0, ErrInvalidBuilder
	}
}

type BuildersConfig struct {
	Builders []struct {
		Name     string `yaml:"name"`
		URL      string `yaml:"url"`
		API      string `yaml:"api"`
		Internal bool   `yaml:"internal,omitempty"`
		Disabled bool   `yaml:"disabled,omitempty"`
		Delay    bool   `yaml:"delay,omitempty"`
	} `yaml:"builders"`
	OrderflowHeader      bool   `yaml:"orderflowHeader,omitempty"`
	OrderflowHeaderValue string `yaml:"orderflowHeaderValue,omitempty"`
	RestrictedAddress    string `yaml:"restrictedAddress"`
}

// LoadBuilderConfig parses a builder config from a file
// 构造者的信息都是从 builders.yaml 配置文件中读取的
// 现在的配置中其实只有一个internal builder，没有其他的builder了
func LoadBuilderConfig(file string) (BuildersBackend, error) {
	return BuildersBackend{
		externalBuilders: map[string]JSONRPCBuilderBackend{},
		internalBuilders: []JSONRPCBuilderBackend{
			{
				Name:  "blockrazor",
				API:   BuilderAPIRefundRecipient,
				Delay: false,
			},
			{
				Name:  "48club",
				API:   BuilderAPIRefundRecipient,
				Delay: false,
			},
			{
				Name:  "blocksmith",
				API:   BuilderAPIRefundRecipient,
				Delay: false,
			},
			{
				Name:  "blockxroute",
				API:   BuilderAPIRefundRecipient,
				Delay: false,
			},
		},
		RestrictedAddress: "",
	}, nil
}

type JSONRPCBuilderBackend struct {
	Name   string
	Client jsonrpc.RPCClient
	API    BuilderAPI
	Delay  bool
}

func (b *JSONRPCBuilderBackend) SendBundle(ctx context.Context, bundle *SendMevBundleArgs) (err error) {
	switch b.API {
	case BuilderAPIRefundRecipient:
		// eth_sendBundle 只有一个refund结果，那就是第一个交易的第一个sender
		// 返回的 refund.percent 或 refund.percent * refundConfig[0].percent
		// 这个是最开始的 api 版本
		_, err := ConvertBundleToRefundRecipient(bundle)
		if err != nil {
			return err
		}
		time.Sleep(20 * time.Millisecond)
		//res, err := b.Client.Call(ctx, "eth_sendBundle", []SendRefundRecBundleArgs{refRec})
		//if err != nil {
		//	return err
		//}
		//if res.Error != nil {
		//	return res.Error
		//}
		// 后面这两个应该是新的 api 现在是这个为标准
	case BuilderAPIMevShareBeta1:
		//res, err := b.Client.Call(ctx, "mev_sendBundle", []SendMevBundleArgs{*bundle})
		//if err != nil {
		//	return err
		//}
		//if res.Error != nil {
		//	return res.Error
		//}
	case BuilderAPIMevShareBeta1Replacement:
		//res, err := b.Client.Call(ctx, "mev_sendBundle", []SendMevBundleArgs{*bundle})
		//if err != nil {
		//	return err
		//}
		//if res.Error != nil {
		//	return res.Error
		//}
	}
	return nil
}

func (b *JSONRPCBuilderBackend) CancelBundleByHash(ctx context.Context, hash common.Hash) error {
	res, err := b.Client.Call(ctx, "mev_cancelBundleByHash", []common.Hash{hash})
	if err != nil {
		return err
	}
	if res.Error != nil {
		return res.Error
	}
	return nil
}

type BuildersBackend struct {
	externalBuilders  map[string]JSONRPCBuilderBackend
	internalBuilders  []JSONRPCBuilderBackend
	RestrictedAddress string
}

// SendBundle sends a bundle to all builders.
// Bundles are sent to all builders in parallel.
func (b *BuildersBackend) SendBundle(ctx context.Context, logger *zap.Logger, bundle *SendMevBundleArgs, targetBlock uint64, shouldCancel bool) { //nolint:gocognit
	var wg sync.WaitGroup
	isFirstBlock := uint64(bundle.Inclusion.BlockNumber) == targetBlock

	isReplaceable := bundle.ReplacementUUID != ""
	// clean metadata, privacy, inclusion
	args := *bundle
	args.Inclusion.BlockNumber = hexutil.Uint64(targetBlock)
	args.Inclusion.MaxBlock = hexutil.Uint64(targetBlock)
	var signingAddress common.Address
	if args.Metadata != nil {
		signingAddress = args.Metadata.Signer
	}
	if signingAddress == (common.Address{}) {
		logger.Warn("No signing address provided for bundle")
	}
	logger = logger.With(zap.Bool("shouldCancel", shouldCancel))
	// 如果他设置了 builders，说明是外部的 builder
	var builders []string
	if args.Privacy != nil {
		// it should already be cleaned while matching, but just in case we do it again here
		MergePrivacyBuilders(&args)
		builders = args.Privacy.Builders
	}
	cleanBundle(&args)

	// for internal builders send signing_address
	iArgs := &SendMevBundleArgs{
		Version:         args.Version,
		Inclusion:       args.Inclusion,
		Body:            args.Body,
		Validity:        args.Validity,
		Privacy:         args.Privacy,
		ReplacementUUID: bundle.ReplacementUUID,
		Metadata: &MevBundleMetadata{
			Signer:           signingAddress,
			ReplacementNonce: bundle.Metadata.ReplacementNonce,
			Cancelled:        shouldCancel,
		},
	}
	// always send to internal builders
	internalBuildersSuccess := make([]bool, len(b.internalBuilders))
	for idx, builder := range b.internalBuilders {
		// if bundle needs to be replaceable, only send to builders that support replacement
		if isReplaceable && builder.API != BuilderAPIMevShareBeta1Replacement {
			internalBuildersSuccess[idx] = true
			continue
		}
		// if address only allows sending to replacement supporting builders we skip
		if strings.ToLower(iArgs.Metadata.Signer.String()) == strings.ToLower(b.RestrictedAddress) && builder.API != BuilderAPIMevShareBeta1Replacement {
			logger.Debug("Skipping restricted address", zap.String("restrictedAddress", b.RestrictedAddress))
			internalBuildersSuccess[idx] = true
			continue
		}
		wg.Add(1)
		go func(builder JSONRPCBuilderBackend, idx int) {
			defer wg.Done()
			if builder.Delay && isFirstBlock {
				// mark as success
				logger.Debug("Skipping builder due to delay", zap.String("builder", builder.Name), zap.Uint64("blockNumber", uint64(bundle.Inclusion.BlockNumber)), zap.Uint64("targetBlock", targetBlock))
				internalBuildersSuccess[idx] = true
				return
			}

			start := time.Now()
			err := builder.SendBundle(ctx, iArgs)
			now := time.Now()
			logger.Debug("Sent bundle to internal builder", zap.String("builder", builder.Name), zap.Duration("duration", time.Since(start)), zap.Error(err), zap.Time("time", now), zap.Int64("timestamp", now.Unix()))

			if err != nil {
				logger.Warn("Failed to send bundle to internal builder", zap.Error(err), zap.String("builder", builder.Name), zap.Time("time", now), zap.Int64("timestamp", now.Unix()))
			} else {
				internalBuildersSuccess[idx] = true
			}
		}(builder, idx)
	}

	if len(builders) > 0 {
		buildersUsed := make(map[string]struct{})
		for _, target := range builders {
			// if bundle needs to be replaceable, only send to builders that support replacement

			target = strings.ToLower(target)

			if target == "default" || target == "flashbots" {
				// right now we always send to flashbots and default means flashbots
				continue
			}
			if _, ok := buildersUsed[target]; ok {
				continue
			}
			buildersUsed[target] = struct{}{}
			if builder, ok := b.externalBuilders[target]; ok {
				if isReplaceable && builder.API != BuilderAPIMevShareBeta1Replacement {
					continue
				}
				wg.Add(1)
				go func(builder JSONRPCBuilderBackend) {
					if builder.Delay && isFirstBlock {
						logger.Debug("Skipping builder due to delay", zap.String("builder", builder.Name), zap.Uint64("blockNumber", uint64(bundle.Inclusion.BlockNumber)), zap.Uint64("targetBlock", targetBlock))
						return
					}
					defer wg.Done()
					start := time.Now()
					err := builder.SendBundle(ctx, &args)
					now := time.Now()
					logger.Debug("Sent bundle to external builder", zap.String("builder", builder.Name), zap.Duration("duration", time.Since(start)), zap.Error(err), zap.Time("time", now), zap.Int64("timestamp", now.Unix()))

					if err != nil {
						logger.Warn("Failed to send bundle to external builder", zap.Error(err), zap.String("builder", builder.Name), zap.Time("time", now), zap.Int64("timestamp", now.Unix()))
					}
				}(builder)
			} else {
				logger.Warn("Unknown external builder", zap.String("builder", target))
			}
		}
	}

	wg.Wait()

	sentToInternal := false
	for _, success := range internalBuildersSuccess {
		if success {
			sentToInternal = true
			break
		}
	}
	if !sentToInternal {
		logger.Error("Failed to send bundle to any of the internal builders")
	}
}

func (b *BuildersBackend) CancelBundleByHash(ctx context.Context, logger *zap.Logger, hash common.Hash) {
	var wg sync.WaitGroup
	// we cancel bundle only in the internal builders, external cancellations are not supported
	for _, builder := range b.internalBuilders {
		wg.Add(1)
		go func(builder JSONRPCBuilderBackend) {
			err := builder.CancelBundleByHash(ctx, hash)
			if err != nil {
				logger.Warn("Failed to cancel bundle on the internal builder", zap.Error(err), zap.String("builder", builder.Name))
			}
		}(builder)
	}
	wg.Wait()
}

// cleanBundle 将 Privacy 和 Metadata 设置为 nil
func cleanBundle(bundle *SendMevBundleArgs) {
	for _, el := range bundle.Body {
		if el.Bundle != nil {
			cleanBundle(el.Bundle)
		}
	}
	bundle.Privacy = nil
	bundle.Metadata = nil
}
