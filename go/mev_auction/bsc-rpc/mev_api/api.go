// Copyright 2015 The go-ethereum Authors
// This file is part of the go-ethereum library.
//
// The go-ethereum library is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// The go-ethereum library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with the go-ethereum library. If not, see <http://www.gnu.org/licenses/>.

package mev_api

import (
	"context"
	"errors"
	"fmt"
	"github.com/ethereum/go-ethereum-test/base"
	"github.com/ethereum/go-ethereum-test/bundlepool"
	"github.com/spf13/cast"
	"math/big"
	"time"

	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/common/hexutil"
	"github.com/ethereum/go-ethereum/core/types"
	"github.com/ethereum/go-ethereum/crypto"
	"github.com/ethereum/go-ethereum/log"
	"github.com/ethereum/go-ethereum/params"
)

type API struct {
	pool *bundlepool.BundlePool
}

func NewAPI(pool *bundlepool.BundlePool) *API {
	return &API{
		pool: pool,
	}
}

type SendRawTransactionArgs struct {
	Input          hexutil.Bytes
	UserId         int
	MaxBlockNumber uint64
}

type SendRawTransactionResponse struct {
	TxHash common.Hash `json:"txHash"`
}

var EmptyRawResponse = SendRawTransactionResponse{
	TxHash: common.Hash{},
}

type ResetHeaderArgs struct {
	HeaderNumber uint64
	Time         uint64
}

type ResetHeaderResponse struct {
	HeaderNumber uint64 `json:"headerNumber"`
}

// ResetHeader 客户端定时调用即可
func (a *API) ResetHeader(ctx context.Context, args ResetHeaderArgs) (ResetHeaderResponse, error) {
	curr := &types.Header{
		Number: big.NewInt(int64(args.HeaderNumber)),
		Time:   args.Time,
	}
	base.CurrentHeader = curr
	a.pool.Reset(nil, curr)
	return ResetHeaderResponse{
		HeaderNumber: args.HeaderNumber,
	}, nil
}

// SendRawTransaction will add the signed transaction to the transaction pool.
// The sender is responsible for signing the transaction and using the correct nonce.
func (a *API) SendRawTransaction(ctx context.Context, input SendRawTransactionArgs) (SendRawTransactionResponse, error) {
	tx := new(types.Transaction)
	if err := tx.UnmarshalBinary(input.Input); err != nil {
		return EmptyRawResponse, err
	}
	return SubmitTransaction(ctx, a, tx, input.MaxBlockNumber, input.UserId)
}

// SubmitTransaction is a helper function that submits tx to txPool and logs a message.
func SubmitTransaction(ctx context.Context, a *API, tx *types.Transaction, maxBlockNumber uint64, userId int) (SendRawTransactionResponse, error) {
	//remoteAddr := ctx.Value("remote").(string)
	//split := strings.Split(remoteAddr, ":")
	//if len(split) == 2 {
	//	limiter := limitip.GetLimiter(split[0])
	//	if !limiter.Allow() {
	//		return EmptyRawResponse, newBundleError(errors.New("too many requests"))
	//	}
	//}

	//if relay.SubServer.IsPublic(tx.Hash()) {
	//	TransactionFilterPublicGauge.Inc(1)
	//	return SendRawTransactionResponse{
	//		TxHash: tx.Hash(),
	//	}, nil
	//}
	// If the transaction fee cap is already specified, ensure the
	// fee of the given transaction is _reasonable_.
	if err := checkTxFee(tx.GasPrice(), tx.Gas(), 0); err != nil {
		return EmptyRawResponse, err
	}

	signer := types.LatestSignerForChainID(big.NewInt(56))
	from, err := types.Sender(signer, tx)
	if err != nil {
		return EmptyRawResponse, err
	}

	bundle := &base.Bundle{
		UserId:            userId,
		ParentHash:        common.Hash{},
		Txs:               []*types.Transaction{tx},
		MaxBlockNumber:    maxBlockNumber,
		RevertingTxHashes: nil,
		Counter:           0,
		RefundAddress:     from,
		RefundPercent:     99,
		Hint:              make(map[string]bool),
		From:              from,
		PrivacyPeriod:     0,
		PrivacyBuilder:    []string{"blockrazor"},
		BroadcastBuilder:  []string{"blockrazor", "48club", "bloxroute", "smith"},
		ArrivalTime:       time.Now(),
	}

	//path := ctx.Value("URL").(string)
	//host := ctx.Value("Host").(string)
	//hostSlice := strings.Split(host, ".")
	//if len(hostSlice) > 4 || len(hostSlice) < 3 {
	//	return EmptyRawResponse, errors.New("invalid host")
	//}
	//var userInfo *portalRpc.GetAllRpcInfoResponse
	//if len(hostSlice) > 3 {
	//	userInfo = portal.UserServer.GetAllRpcInfoList(hostSlice[0])
	//}
	//if userInfo == nil {
	//	if path == "/fullprivacy" {
	//		bundle.RPCID = "fullprivacy"
	//	} else if path == "/maxbackrun" {
	//		bundle.Hint[types.HintHash] = true
	//		bundle.Hint[types.HintTo] = true
	//		bundle.Hint[types.HintCallData] = true
	//		bundle.Hint[types.HintFunctionSelector] = true
	//		bundle.Hint[types.HintLogs] = true
	//		bundle.RPCID = "maxbackrun"
	//	} else if len(path) > 15 {
	//		split := strings.Split(path, "/")
	//		userInfo = portal.UserServer.GetAllRpcInfoList(split[len(split)-1])
	//		if userInfo == nil {
	//			bundle.Hint[types.HintHash] = true
	//			bundle.Hint[types.HintLogs] = true
	//			bundle.RevertingTxHashes = []common.Hash{tx.Hash()}
	//			bundle.RPCID = "default"
	//		}
	//	} else {
	//		bundle.Hint[types.HintHash] = true
	//		bundle.Hint[types.HintLogs] = true
	//		bundle.RevertingTxHashes = []common.Hash{tx.Hash()}
	//		bundle.RPCID = "default"
	//	}
	//}
	//if userInfo == nil {
	//	userInfo = portal.UserServer.GetAllRpcInfoList(bundle.RPCID)
	//}

	bundle.Hint[types.HintHash] = true
	bundle.Hint[types.HintTo] = true
	bundle.Hint[types.HintFrom] = true
	bundle.Hint[types.HintCallData] = true
	bundle.Hint[types.HintFunctionSelector] = true
	bundle.Hint[types.HintLogs] = true
	bundle.Hint[types.HintValue] = true
	bundle.Hint[types.HintGasLimit] = true
	bundle.Hint[types.HintGasPrice] = true
	bundle.RPCID = "maxbackrun"

	//if userInfo != nil {
	//	bundle.RPCID = userInfo.RpcId
	//	bundle.Hint[types.HintHash] = userInfo.HintHash
	//	bundle.Hint[types.HintFrom] = userInfo.HintFrom
	//	bundle.Hint[types.HintTo] = userInfo.HintTo
	//	bundle.Hint[types.HintValue] = userInfo.HintValue
	//	bundle.Hint[types.HintNonce] = userInfo.HintNonce
	//	bundle.Hint[types.HintCallData] = userInfo.HintCalldata
	//	bundle.Hint[types.HintFunctionSelector] = userInfo.HintFunctionSelector
	//	bundle.Hint[types.HintGasLimit] = userInfo.HintGasLimit
	//	bundle.Hint[types.HintGasPrice] = userInfo.HintGasPrice
	//	bundle.Hint[types.HintLogs] = userInfo.HintLogs
	//	bundle.PrivacyPeriod = userInfo.PrivacyPeriod
	//	bundle.PrivacyBuilder = userInfo.PrivacyBuilder
	//	bundle.BroadcastBuilder = userInfo.BroadcastBuilder
	//
	//	if userInfo.RefundPercent >= 1 && userInfo.RefundPercent <= 99 {
	//		bundle.RefundPercent = int(userInfo.RefundPercent)
	//	}
	//	if userInfo.RefundRecipient != "" && userInfo.RefundRecipient != "tx.origin" {
	//		bundle.RefundAddress = common.HexToAddress(userInfo.RefundRecipient)
	//	}
	//
	//	if !userInfo.IsProtected {
	//		bundle.RevertingTxHashes = []common.Hash{tx.Hash()}
	//	}
	//}

	if ctx.Value("RefundPercent") != nil {
		p := cast.ToInt(ctx.Value("RefundPercent"))
		if p >= 1 && p <= 99 {
			bundle.RefundPercent = p
		} else {
			return EmptyRawResponse, errors.New("the RefundPercent in url is illegal")
		}
	}

	//every transaction must have tx tip
	tip, err := tx.EffectiveGasTip(big.NewInt(0))
	if err != nil {
		return EmptyRawResponse, err
	}
	if tip.Uint64() < params.GWei {
		return EmptyRawResponse, errors.New("the gas tip fee of transaction must be greater than or equal to 1gwei")
	}

	//invalid_tx.Server.Delete(tx.Hash())

	if err := a.pool.AddBundle(bundle); err != nil {
		return EmptyRawResponse, err
	}
	// Print a log with full tx details for manual investigations and interventions
	xForward := ctx.Value("X-Forwarded-For")

	if tx.To() == nil {
		addr := crypto.CreateAddress(from, tx.Nonce())
		log.Info("Submitted contract creation", "hash", tx.Hash().Hex(), "from", from, "nonce", tx.Nonce(), "contract", addr.Hex(), "value", tx.Value(), "x-forward-ip", xForward)
	} else {
		log.Info("Submitted transaction", "hash", tx.Hash().Hex(), "from", from, "nonce", tx.Nonce(), "recipient", tx.To(), "value", tx.Value(), "x-forward-ip", xForward)
	}
	return SendRawTransactionResponse{
		TxHash: tx.Hash(),
	}, nil
}

// checkTxFee is an internal function used to check whether the fee of
// the given transaction is _reasonable_(under the cap).
func checkTxFee(gasPrice *big.Int, gas uint64, cap float64) error {
	// Short circuit if there is no cap for transaction fee at all.
	if cap == 0 {
		return nil
	}
	feeEth := new(big.Float).Quo(new(big.Float).SetInt(new(big.Int).Mul(gasPrice, new(big.Int).SetUint64(gas))), new(big.Float).SetInt(big.NewInt(params.Ether)))
	feeFloat, _ := feeEth.Float64()
	if feeFloat > cap {
		return fmt.Errorf("tx fee (%.2f ether) exceeds the configured cap (%.2f ether)", feeFloat, cap)
	}
	return nil
}
