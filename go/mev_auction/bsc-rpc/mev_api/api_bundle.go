package mev_api

import (
	"context"
	"errors"
	"fmt"
	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/common/limit"
	"github.com/ethereum/go-ethereum/core/types"
	"github.com/ethereum/go-ethereum/portal"
	portalRpc "github.com/ethereum/go-ethereum/portal/zrpc_client/typed/rpc_portal/rpcv2"
	"strings"
	"time"
)

const InvalidBundleParamError = -38000

// PrivateTxBundleAPI offers an API for accepting bundled transactions
type PrivateTxBundleAPI struct {
	b ethapi.Backend
}

// NewPrivateTxBundleAPI creates a new Tx Bundle API instance.
func NewPrivateTxBundleAPI(b ethapi.Backend) *PrivateTxBundleAPI {
	return &PrivateTxBundleAPI{b}
}

func newBundleError(err error) *bundleError {
	return &bundleError{
		error: err,
	}
}

// bundleError is an API error that encompasses an invalid bundle with JSON error
// code and a binary data blob.
type bundleError struct {
	error
}

// SendMevBundle rpc core interface
func (s *PrivateTxBundleAPI) SendMevBundle(ctx context.Context, args types.SendMevBundleArgs) (common.Hash, error) {
	remoteAddr := ctx.Value("remote").(string)
	split := strings.Split(remoteAddr, ":")
	if len(split) == 2 {
		limiter := limitip.GetLimiter(split[0])
		if !limiter.Allow() {
			return common.Hash{}, newBundleError(errors.New("too many requests"))
		}
	}

	//fmt.Println(time.Now(), " SendMevBundle:", args)
	if len(args.Txs) == 0 {
		return common.Hash{}, newBundleError(errors.New("bundle missing txs"))
	}

	if (args.Hash != common.Hash{}) && len(args.Txs) > 1 {
		return common.Hash{}, newBundleError(errors.New("the number of back run tx should not be lager than 1"))
	}

	if (args.Hash == common.Hash{}) && len(args.Txs) > types.MaxTxPerBundle {
		return common.Hash{}, newBundleError(errors.New(fmt.Sprintf("the number of bundle txs should not be lager than %d", types.MaxTxPerBundle)))
	}
	currentHeader := s.b.CurrentHeader()

	if args.MaxBlockNumber != 0 && args.MaxBlockNumber > currentHeader.Number.Uint64()+types.MaxBundleAliveBlock {
		return common.Hash{}, newBundleError(errors.New("the maxBlockNumber should not be lager than currentBlockNum + 100"))
	}
	if args.MaxBlockNumber != 0 && args.MaxBlockNumber <= currentHeader.Number.Uint64() {
		return common.Hash{}, newBundleError(errors.New("the maxBlockNumber should be lager than currentBlockNum"))
	}

	// If the maxBlockNumber is not set, set max ddl of bundle as types.MaxBundleAliveBlock
	if args.MaxBlockNumber == 0 {
		args.MaxBlockNumber = currentHeader.Number.Uint64() + types.MaxBundleAliveBlock
	}

	var txs types.Transactions
	for _, encodedTx := range args.Txs {
		tx := new(types.Transaction)
		if err := tx.UnmarshalBinary(encodedTx); err != nil {
			return common.Hash{}, err
		}
		txs = append(txs, tx)
	}

	// check if the revertTxHashes are included in Txs
	for _, revertTxHash := range args.RevertingTxHashes {
		exist := false
		for _, tx := range txs {
			if revertTxHash == tx.Hash() {
				exist = true
			}
		}
		if !exist {
			return common.Hash{}, newBundleError(errors.New("revertingTxHashes should be included in Txs"))
		}
	}

	////////////////////

	bundle := &types.Bundle{
		Txs:               txs,
		MaxBlockNumber:    args.MaxBlockNumber,
		RevertingTxHashes: args.RevertingTxHashes,
		ParentHash:        args.Hash,
		Counter:           1,
		Hint:              args.Hint,
		RefundAddress:     args.RefundAddress,
		RefundPercent:     args.RefundPercent,
		From:              common.Address{},

		PrivacyPeriod:    0,
		PrivacyBuilder:   []string{"blockrazor"},
		BroadcastBuilder: []string{"blockrazor", "48club", "bloxroute"},
		ArrivalTime:      time.Now(),
	}

	path := ctx.Value("URL").(string)
	host := ctx.Value("Host").(string)
	hostSlice := strings.Split(host, ".")
	if len(hostSlice) > 4 || len(hostSlice) < 3 {
		return common.Hash{}, errors.New("invalid host")
	}
	var userInfo *portalRpc.GetAllRpcInfoResponse
	if len(hostSlice) > 3 {
		userInfo = portal.UserServer.GetAllRpcInfoList(hostSlice[0])
	}
	if userInfo == nil && len(path) > 15 {
		split := strings.Split(path, "/")
		userInfo = portal.UserServer.GetAllRpcInfoList(split[len(split)-1])
	}
	if userInfo == nil {
		bundle.RPCID = "default"
		userInfo = portal.UserServer.GetAllRpcInfoList(bundle.RPCID)
	}
	if userInfo != nil {
		bundle.RPCID = userInfo.RpcId
		bundle.PrivacyPeriod = userInfo.PrivacyPeriod
		bundle.PrivacyBuilder = userInfo.PrivacyBuilder
		bundle.BroadcastBuilder = userInfo.BroadcastBuilder
	}
	////////////////////

	for hint, _ := range args.Hint {
		if hint != types.HintHash &&
			hint != types.HintFrom &&
			hint != types.HintTo &&
			hint != types.HintCallData &&
			hint != types.HintFunctionSelector &&
			hint != types.HintGasLimit &&
			hint != types.HintLogs &&
			hint != types.HintNonce &&
			hint != types.HintValue &&
			hint != types.HintGasPrice {
			return common.Hash{}, newBundleError(errors.New(fmt.Sprintf("hint[%s] is illegal", hint)))
		}
	}

	isShare := false
	for _, v := range args.Hint {
		if v {
			isShare = true
			break
		}
	}
	if isShare {
		if (args.RefundAddress == common.Address{}) {
			return common.Hash{}, newBundleError(errors.New("bundle missing refundRecipient when hints exist"))
		}
		if args.RefundPercent < 0 || args.RefundPercent >= 100 {
			return common.Hash{}, newBundleError(errors.New("refundPercent must be between 0 and 99 when hints exist"))
		}
	} else {
		if args.RefundPercent < 0 || args.RefundPercent >= 100 {
			return common.Hash{}, newBundleError(errors.New("refundPercent must be between 0 and 99 even though hints not exist"))
		}
	}

	// This bundle is not yet complete and needs to be further constructed in the addBundle.
	// Since bundles depend on parent bundle, txs from parent bundle need to be combined
	// and the bundle counter needs to be updated.
	err := s.b.SendBundle(ctx, bundle)
	if err != nil {
		return common.Hash{}, newBundleError(err)
	}

	return bundle.Hash(), nil
}
