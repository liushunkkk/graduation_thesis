package miner

import (
	"errors"
	"fmt"
	. "github.com/ethereum/go-ethereum/log/zap"
	"github.com/ethereum/go-ethereum/push/define"
	"go.uber.org/zap"
	"math/big"
	"sort"
	"sync"
	"time"

	"github.com/holiman/uint256"

	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/consensus/misc/eip4844"
	"github.com/ethereum/go-ethereum/core"
	"github.com/ethereum/go-ethereum/core/state"
	"github.com/ethereum/go-ethereum/core/txpool"
	"github.com/ethereum/go-ethereum/core/types"
	"github.com/ethereum/go-ethereum/log"
	"github.com/ethereum/go-ethereum/params"
)

const smallBundleGas = 10 * params.TxGas

var (
	errNonRevertingTxInBundleFailed = errors.New("execution failed,but non-reverting tx in bundle")
	errBundlePriceTooLow            = errors.New("bundle price too low")
	errNoBribe                      = errors.New("no bribe")
	errInvalidBundleCounter         = errors.New("invalid bundle counter")
	errBundleParentMissing          = errors.New("bundle parent missing")
)

// fillTransactions retrieves the pending bundles and transactions from the txpool and fills them
// into the given sealing block. The selection and ordering strategy can be extended in the future.
func (w *worker) fillTransactionsAndBundles(interruptCh chan int32, env *environment, stopTimer *time.Timer) error {
	env.state.StopPrefetcher() // no need to prefetch txs for a builder

	var (
		localPlainTxs  map[common.Address][]*txpool.LazyTransaction
		remotePlainTxs map[common.Address][]*txpool.LazyTransaction
		localBlobTxs   map[common.Address][]*txpool.LazyTransaction
		remoteBlobTxs  map[common.Address][]*txpool.LazyTransaction
		bundles        []*types.Bundle
	)

	// commit bundles
	{
		bundles = w.eth.TxPool().PendingBundles(env.header.Number.Uint64(), env.header.Time)

		// if no bundles, not necessary to fill transactions
		if len(bundles) == 0 {
			return errors.New("no bundles in bundle pool")
		}

		txs, bundle, err := w.generateOrderedBundles(env, bundles)
		if err != nil {
			log.Error("fail to generate ordered bundles", "err", err)
			return err
		}

		if err = w.commitBundles(env, txs, interruptCh, stopTimer); err != nil {
			log.Error("fail to commit bundles", "err", err)
			return err
		}

		env.profit.Add(env.profit, bundle.EthSentToSystem)
		log.Info("fill bundles", "bundles_count", len(bundles))
	}

	// commit normal transactions
	{
		w.mu.RLock()
		tip := w.tip
		w.mu.RUnlock()

		// Retrieve the pending transactions pre-filtered by the 1559/4844 dynamic fees
		filter := txpool.PendingFilter{
			MinTip: tip,
		}
		if env.header.BaseFee != nil {
			filter.BaseFee = uint256.MustFromBig(env.header.BaseFee)
		}
		if env.header.ExcessBlobGas != nil {
			filter.BlobFee = uint256.MustFromBig(eip4844.CalcBlobFee(*env.header.ExcessBlobGas))
		}
		filter.OnlyPlainTxs, filter.OnlyBlobTxs = true, false
		pendingPlainTxs := w.eth.TxPool().Pending(filter)

		filter.OnlyPlainTxs, filter.OnlyBlobTxs = false, true
		pendingBlobTxs := w.eth.TxPool().Pending(filter)

		// Split the pending transactions into locals and remotes
		// Fill the block with all available pending transactions.
		localPlainTxs, remotePlainTxs = make(map[common.Address][]*txpool.LazyTransaction), pendingPlainTxs
		localBlobTxs, remoteBlobTxs = make(map[common.Address][]*txpool.LazyTransaction), pendingBlobTxs

		for _, account := range w.eth.TxPool().Locals() {
			if txs := remotePlainTxs[account]; len(txs) > 0 {
				delete(remotePlainTxs, account)
				localPlainTxs[account] = txs
			}
			if txs := remoteBlobTxs[account]; len(txs) > 0 {
				delete(remoteBlobTxs, account)
				localBlobTxs[account] = txs
			}
		}
		log.Info("fill transactions", "plain_txs_count", len(localPlainTxs)+len(remotePlainTxs), "blob_txs_count", len(localBlobTxs)+len(remoteBlobTxs))
	}

	// Fill the block with all available pending transactions.
	// we will abort when:
	//   1.new block was imported
	//   2.out of Gas, no more transaction can be added.
	//   3.the mining timer has expired, stop adding transactions.
	//   4.interrupted resubmit timer, which is by default 10s.
	//     resubmit is for PoW only, can be deleted for PoS consensus later
	if len(localPlainTxs) > 0 || len(localBlobTxs) > 0 {
		plainTxs := newTransactionsByPriceAndNonce(env.signer, localPlainTxs, env.header.BaseFee)
		blobTxs := newTransactionsByPriceAndNonce(env.signer, localBlobTxs, env.header.BaseFee)

		if err := w.commitTransactions(env, plainTxs, blobTxs, interruptCh, stopTimer); err != nil {
			return err
		}
	}
	if len(remotePlainTxs) > 0 || len(remoteBlobTxs) > 0 {
		plainTxs := newTransactionsByPriceAndNonce(env.signer, remotePlainTxs, env.header.BaseFee)
		blobTxs := newTransactionsByPriceAndNonce(env.signer, remoteBlobTxs, env.header.BaseFee)

		if err := w.commitTransactions(env, plainTxs, blobTxs, interruptCh, stopTimer); err != nil {
			return err
		}
	}
	log.Info("fill bundles and transactions done", "total_txs_count", len(env.txs))
	return nil
}

func (w *worker) commitBundles(
	env *environment,
	txs types.Transactions,
	interruptCh chan int32,
	stopTimer *time.Timer,
) error {
	if env.gasPool == nil {
		env.gasPool = prepareGasPool(env.header.GasLimit)
	}

	var coalescedLogs []*types.Log
	signal := commitInterruptNone
LOOP:
	for _, tx := range txs {
		// In the following three cases, we will interrupt the execution of the transaction.
		// (1) new head block event arrival, the reason is 1
		// (2) worker start or restart, the reason is 1
		// (3) worker recreate the sealing block with any newly arrived transactions, the reason is 2.
		// For the first two cases, the semi-finished work will be discarded.
		// For the third case, the semi-finished work will be submitted to the consensus engine.
		if interruptCh != nil {
			select {
			case signal, ok := <-interruptCh:
				if !ok {
					// should never be here, since interruptCh should not be read before
					log.Warn("commit transactions stopped unknown")
				}
				return signalToErr(signal)
			default:
			}
		} // If we don't have enough gas for any further transactions then we're done
		if env.gasPool.Gas() < params.TxGas {
			log.Trace("Not enough gas for further transactions", "have", env.gasPool, "want", params.TxGas)
			signal = commitInterruptOutOfGas
			break
		}
		if tx == nil {
			log.Error("Unexpected nil transaction in bundle")
			return signalToErr(commitInterruptBundleTxNil)
		}
		if stopTimer != nil {
			select {
			case <-stopTimer.C:
				log.Info("Not enough time for further transactions", "txs", len(env.txs))
				stopTimer.Reset(0) // re-active the timer, in case it will be used later.
				signal = commitInterruptTimeout
				break LOOP
			default:
			}
		}

		// Error may be ignored here. The error has already been checked
		// during transaction acceptance is the transaction pool.
		//
		// We use the eip155 signer regardless of the current hf.
		from, _ := types.Sender(env.signer, tx)
		// Check whether the tx is replay protected. If we're not in the EIP155 hf
		// phase, start ignoring the sender until we do.
		if tx.Protected() && !w.chainConfig.IsEIP155(env.header.Number) {
			log.Debug("Unexpected protected transaction in bundle")
			return signalToErr(commitInterruptBundleTxProtected)
		}
		// Start executing the transaction
		env.state.SetTxContext(tx.Hash(), env.tcount)

		logs, err := w.commitTransaction(env, tx, core.NewReceiptBloomGenerator())
		switch err {
		case core.ErrGasLimitReached:
			// Pop the current out-of-gas transaction without shifting in the next from the account
			log.Error("Unexpected gas limit exceeded for current block in the bundle", "sender", from)
			return signalToErr(commitInterruptBundleCommit)

		case core.ErrNonceTooLow:
			// New head notification data race between the transaction pool and miner, shift
			log.Error("Transaction with low nonce in the bundle", "sender", from, "nonce", tx.Nonce())
			return signalToErr(commitInterruptBundleCommit)

		case core.ErrNonceTooHigh:
			// Reorg notification data race between the transaction pool and miner, skip account =
			log.Error("Account with high nonce in the bundle", "sender", from, "nonce", tx.Nonce())
			return signalToErr(commitInterruptBundleCommit)

		case nil:
			// Everything ok, collect the logs and shift in the next transaction from the same account
			coalescedLogs = append(coalescedLogs, logs...)
			env.tcount++
			continue

		default:
			// Strange error, discard the transaction and get the next in line (note, the
			// nonce-too-high clause will prevent us from executing in vain).
			log.Error("Transaction failed in the bundle", "hash", tx.Hash(), "err", err)
			return signalToErr(commitInterruptBundleCommit)
		}
	}

	if !w.isRunning() && len(coalescedLogs) > 0 {
		// We don't push the pendingLogsEvent while we are mining. The reason is that
		// when we are mining, the worker will regenerate a mining block every 3 seconds.
		// In order to avoid pushing the repeated pendingLog, we disable the pending log pushing.

		// make a copy, the state caches the logs and these logs get "upgraded" from pending to mined
		// logs by filling in the block hash when the block was mined by the local miner. This can
		// cause a race condition if a log was "upgraded" before the PendingLogsEvent is processed.
		cpy := make([]*types.Log, len(coalescedLogs))
		for i, l := range coalescedLogs {
			cpy[i] = new(types.Log)
			*cpy[i] = *l
		}
		w.pendingLogsFeed.Send(cpy)
	}
	return signalToErr(signal)
}

// generateOrderedBundles generates ordered txs from the given bundles.
// 1. sort bundles according to computed gas price when received.
// 2. simulate bundles based on the same state, resort.
// 3. merge resorted simulateBundles based on the iterative state.
func (w *worker) generateOrderedBundles(
	env *environment,
	bundles []*types.Bundle,
) (types.Transactions, *types.SimulatedBundle, error) {
	// sort bundles according to gas price computed when received
	sort.SliceStable(bundles, func(i, j int) bool {
		priceI, priceJ := bundles[i].Price, bundles[j].Price

		return priceI.Cmp(priceJ) >= 0
	})

	// recompute bundle gas price based on the same state and current env
	simulatedBundles, _, err := w.simulateBundles(env, bundles, common.Address{})
	if err != nil {
		log.Error("fail to simulate bundles base on the same state", "err", err)
		return nil, nil, err
	}

	// sort bundles according to fresh gas price
	sort.SliceStable(simulatedBundles, func(i, j int) bool {
		priceI, priceJ := simulatedBundles[i].BundleGasPrice, simulatedBundles[j].BundleGasPrice

		return priceI.Cmp(priceJ) >= 0
	})

	// merge bundles based on iterative state
	includedTxs, mergedBundle, err := w.mergeBundles(env, simulatedBundles)
	if err != nil {
		log.Error("fail to merge bundles", "err", err)
		return nil, nil, err
	}

	return includedTxs, mergedBundle, nil
}

// Deprecated
// simulateBundles
func (w *worker) simulateBundles(env *environment, bundles []*types.Bundle, rpcBribeAddress common.Address) ([]*types.SimulatedBundle, []*define.SseBundleData, error) {
	simResult := make(map[common.Hash]*types.SimulatedBundle)
	simBundleData := make([]*define.SseBundleData, 0)
	var wg sync.WaitGroup
	var mu sync.Mutex
	var err error
	for i, bundle := range bundles {
		wg.Add(1)
		go func(idx int, bundle *types.Bundle, state *state.StateDB) {
			defer wg.Done()

			gasPool := prepareGasPool(env.header.GasLimit)
			var tempGasUsed uint64
			var currentTxIndex int
			var simulateResult *types.SimulatedBundle
			var sseBundleData *define.SseBundleData
			simulateResult, sseBundleData, err = w.simulateBundle(env, bundle, state, gasPool, &currentTxIndex, false, false, &tempGasUsed, rpcBribeAddress)
			if err != nil {
				log.Trace("Error computing gas for a simulateBundle", "error", err)
				return
			}

			mu.Lock()
			defer mu.Unlock()
			simResult[bundle.Hash()] = simulateResult
			simBundleData = append(simBundleData, sseBundleData)
		}(i, bundle, env.state.Copy())
	}
	wg.Wait()
	simulatedBundles := make([]*types.SimulatedBundle, 0)
	for _, bundle := range simResult {
		if bundle == nil {
			continue
		}
		simulatedBundles = append(simulatedBundles, bundle)
	}
	return simulatedBundles, simBundleData, err
}

// mergeBundles merges the given simulateBundle into the given environment.
// It returns the merged simulateBundle and the number of transactions that were merged.
func (w *worker) mergeBundles(
	env *environment,
	bundles []*types.SimulatedBundle,
) (types.Transactions, *types.SimulatedBundle, error) {
	currentState := env.state.Copy()
	gasPool := prepareGasPool(env.header.GasLimit)

	includedTxs := types.Transactions{}
	mergedBundle := types.SimulatedBundle{
		BundleGasFees:   new(big.Int),
		BundleGasUsed:   0,
		BundleGasPrice:  new(big.Int),
		EthSentToSystem: new(big.Int),
	}

	for _, bundle := range bundles {
		// if we don't have enough gas for any further transactions then we're done
		if gasPool.Gas() < smallBundleGas {
			break
		}

		prevState := currentState.Copy()
		prevGasPool := new(core.GasPool).AddGas(gasPool.Gas())

		// the floor gas price is 99/100 what was simulated at the top of the block
		floorGasPrice := new(big.Int).Mul(bundle.BundleGasPrice, big.NewInt(99))
		floorGasPrice = floorGasPrice.Div(floorGasPrice, big.NewInt(100))
		var tempGasUsed uint64

		var currentTxIndex int
		simulatedBundle, _, err := w.simulateBundle(env, bundle.OriginalBundle, currentState, gasPool, &currentTxIndex, true, false, &tempGasUsed, common.Address{})

		if err != nil || simulatedBundle.BundleGasPrice.Cmp(floorGasPrice) <= 0 {
			currentState = prevState
			gasPool = prevGasPool

			log.Error("failed to merge bundle", "floorGasPrice", floorGasPrice, "err", err)
			continue
		}

		log.Info("included bundle",
			"gasUsed", simulatedBundle.BundleGasUsed,
			"gasPrice", simulatedBundle.BundleGasPrice,
			"txcount", len(simulatedBundle.OriginalBundle.Txs))

		includedTxs = append(includedTxs, bundle.OriginalBundle.Txs...)

		mergedBundle.BundleGasFees.Add(mergedBundle.BundleGasFees, simulatedBundle.BundleGasFees)
		mergedBundle.BundleGasUsed += simulatedBundle.BundleGasUsed

		for _, tx := range includedTxs {
			if !containsHash(bundle.OriginalBundle.RevertingTxHashes, tx.Hash()) {
				env.UnRevertible = append(env.UnRevertible, tx.Hash())
			}
		}
	}

	if len(includedTxs) == 0 {
		return nil, nil, errors.New("include no txs when merge bundles")
	}

	mergedBundle.BundleGasPrice.Div(mergedBundle.BundleGasFees, new(big.Int).SetUint64(mergedBundle.BundleGasUsed))

	return includedTxs, &mergedBundle, nil
}

// Deprecated
// simulateBundle computes the gas price for a whole simulateBundle based on the same ctx
// named computeBundleGas in flashbots
func (w *worker) simulateBundle(
	env *environment, bundle *types.Bundle, state *state.StateDB, gasPool *core.GasPool, currentTxCount *int,
	prune, pruneGasExceed bool, tempGasUsed *uint64, rpcBribeAddress common.Address,
) (*types.SimulatedBundle, *define.SseBundleData, error) {
	var (
		rpcTransactionFees  = new(big.Int) // the transaction fees of this bundler's txs spend
		rpcSimulator        = w.rpcSimulator
		rpcBundlePrice      = new(big.Int) // bundler price
		sseBundleData       = &define.SseBundleData{}
		parentSimBundle     *types.SimulatedBundle
		parentSseBundleData *define.SseBundleData
	)

	// simulate parent bundle
	if bundle.Parent != nil {
		var err error
		parentSimBundle, parentSseBundleData, err = w.simulateBundle(env, bundle.Parent, state, gasPool, currentTxCount, prune, pruneGasExceed, tempGasUsed, rpcBribeAddress)
		if err != nil {
			log.Warn("fail to simulate bundle's parent bundle", "hash", bundle.Parent.Hash().String(), "err", err)
			return nil, nil, err
		}
	}
	if bundle.Parent != nil && parentSseBundleData != nil {
		// merge parent txs, keep the parent's txs ahead of this bundle's txs
		sseBundleData.SseTxs = append(sseBundleData.SseTxs, parentSseBundleData.SseTxs...)
	}

	var rpcBalanceBefore, refundRecipientBalanceBefore *uint256.Int

	rpcBalanceBefore = state.GetBalance(rpcBribeAddress)
	if bundle.Parent != nil {
		refundRecipientBalanceBefore = state.GetBalance(bundle.Parent.RefundAddress)
	}

	for _, tx := range bundle.Txs {
		state.SetTxContext(tx.Hash(), *currentTxCount)
		*currentTxCount++

		receipt, err := core.ApplyTransaction(w.chainConfig, w.chain, &w.coinbase, gasPool, state, env.header, tx,
			tempGasUsed, *w.chain.GetVMConfig())
		if err != nil {
			log.Warn("fail to simulate bundle", "hash", bundle.Hash().String(), "err", err)
			return nil, nil, err
		}

		if receipt.Status == types.ReceiptStatusFailed && !containsHash(bundle.RevertingTxHashes, receipt.TxHash) {
			err = errors.New(fmt.Sprintf("transaction execution failed:%s", receipt.GetReturnData()))
			log.Warn("fail to simulate bundle", "hash", bundle.Hash().String(), "err", err)
			return nil, nil, err
		}

		txGasUsed := new(big.Int).SetUint64(receipt.GasUsed)
		effectiveTip, err := tx.EffectiveGasTip(env.header.BaseFee)
		if err != nil {
			return nil, nil, err
		}
		txGasFees := new(big.Int).Mul(txGasUsed, effectiveTip)

		if tx.Type() == types.BlobTxType {
			blobFee := new(big.Int).SetUint64(receipt.BlobGasUsed)
			blobFee.Mul(blobFee, receipt.BlobGasPrice)
			txGasFees.Add(txGasFees, blobFee)
		}
		// add transaction fees
		rpcTransactionFees.Add(rpcTransactionFees, txGasFees)
		// build sse txData
		curSseTxData, err := rpcSimulator.BuildTxData(bundle, tx, receipt)
		if err != nil {
			return nil, nil, err
		}
		sseBundleData.SseTxs = append(sseBundleData.SseTxs, curSseTxData)
	}

	// add bribe
	rpcBalanceAfter := state.GetBalance(rpcBribeAddress)
	rpcBribeDelta := new(uint256.Int).Sub(rpcBalanceAfter, rpcBalanceBefore).ToBig()

	rpcBundlePrice.Add(rpcBundlePrice, rpcTransactionFees)
	if bundle.Parent != nil {
		refundRecipientBalanceAfter := state.GetBalance(bundle.Parent.RefundAddress)
		refundRecipientDelta := big.NewInt(0).Sub(refundRecipientBalanceAfter.ToBig(), refundRecipientBalanceBefore.ToBig())

		if refundRecipientDelta.Cmp(big.NewInt(0)) < 0 {
			return nil, nil, errors.New("bribe value is illegal")
		}

		if rpcBribeDelta.Cmp(big.NewInt(0)) > 0 && refundRecipientDelta.Cmp(big.NewInt(0)) > 0 {
			a := big.NewInt(0).Mul(refundRecipientDelta,
				big.NewInt(0).Add(
					big.NewInt(int64(100*100*types.RpcBribePercent)),
					big.NewInt(int64((100-types.RpcBribePercent)*(100-bundle.Parent.RefundPercent)*(100-types.RpcBuilderProfitPercent)))))
			b := big.NewInt(0).Mul(rpcBribeDelta, big.NewInt(int64(100*(100-types.RpcBribePercent)*bundle.Parent.RefundPercent)))

			a1, _ := new(big.Rat).SetString(a.String())
			b1, _ := new(big.Rat).SetString(b.String())
			c := new(big.Rat)
			c.Quo(a1, b1)

			rate1, _ := new(big.Rat).SetString("0.99")
			rate2, _ := new(big.Rat).SetString("1.01")

			if c.Cmp(rate1) <= 0 || c.Cmp(rate2) >= 0 {
				return nil, nil, errors.New("refund is illegal")
			}
		}

		// calculate bundle price
		rpcBundlePrice.Add(rpcBundlePrice, parentSimBundle.RpcBundlePrice)
		// to builder price
		t := big.NewInt(0).Mul(refundRecipientDelta, big.NewInt(int64(100-bundle.Parent.RefundPercent)))
		t.Div(t, big.NewInt(int64(bundle.Parent.RefundPercent)))
		t.Mul(t, big.NewInt(int64(types.RpcBuilderProfitPercent)))
		t.Div(t, big.NewInt(100))
		rpcBundlePrice.Add(rpcBundlePrice, t)
	}

	// set chainId, bundleHash and MaxBlockNumber
	sseBundleData.Hash = bundle.Hash().Hex()
	sseBundleData.ChainID = w.chain.Config().ChainID.String()
	sseBundleData.MaxBlockNumber = bundle.MaxBlockNumber

	return &types.SimulatedBundle{
		OriginalBundle: bundle,
		RpcBundlePrice: rpcBundlePrice,
	}, sseBundleData, nil
}

func (w *worker) simulateGaslessBundle(env *environment, bundle *types.Bundle) (*types.SimulateGaslessBundleResp, error) {
	result := make([]types.GaslessTxSimResult, 0)

	txIdx := 0
	for _, tx := range bundle.Txs {
		env.state.SetTxContext(tx.Hash(), txIdx)

		var (
			snap = env.state.Snapshot()
			gp   = env.gasPool.Gas()
		)

		receipt, err := core.ApplyTransaction(w.chainConfig, w.chain, &w.coinbase, env.gasPool, env.state, env.header, tx,
			&env.header.GasUsed, *w.chain.GetVMConfig())
		if err != nil {
			env.state.RevertToSnapshot(snap)
			env.gasPool.SetGas(gp)
			log.Error("fail to simulate gasless tx, skipped", "hash", tx.Hash(), "err", err)
		} else {
			txIdx++

			result = append(result, types.GaslessTxSimResult{
				Hash:    tx.Hash(),
				GasUsed: receipt.GasUsed,
			})
		}
	}

	return &types.SimulateGaslessBundleResp{
		ValidResults:     result,
		BasedBlockNumber: env.header.Number.Int64(),
	}, nil
}

func containsHash(arr []common.Hash, match common.Hash) bool {
	for _, elem := range arr {
		if elem == match {
			return true
		}
	}
	return false
}

func prepareGasPool(gasLimit uint64) *core.GasPool {
	gasPool := new(core.GasPool).AddGas(gasLimit)
	gasPool.SubGas(params.SystemTxsGas) // reserve gas for system txs(keep align with mainnet)
	return gasPool
}

// WorkerExecuteBundle computes the gas price for a whole simulateBundle based on the same ctx
// named computeBundleGas in flashbots
func (w *worker) WorkerExecuteBundle(
	env *environment, bundle *types.Bundle, state *state.StateDB, gasPool *core.GasPool, currentTxCount *int,
	prune, pruneGasExceed bool, tempGasUsed *uint64, rpcBribeAddress common.Address, calllayer int,
) (*types.SimulatedBundle, *define.SseBundleData, error) {
	var (
		rpcTransactionFees  = new(big.Int) // the transaction fees of this bundler's txs spend
		rpcSimulator        = w.rpcSimulator
		rpcBundlePrice      = new(big.Int) // bundler price
		sseBundleData       = &define.SseBundleData{}
		parentSimBundle     *types.SimulatedBundle
		parentSseBundleData *define.SseBundleData
		bundleGasUsed       = uint64(0)
		bundleGasFees       = new(big.Int)
	)

	// simulate parent bundle
	if bundle.Parent != nil {
		var err error
		parentSimBundle, parentSseBundleData, err = w.WorkerExecuteBundle(env, bundle.Parent, state, gasPool, currentTxCount, prune, pruneGasExceed, tempGasUsed, rpcBribeAddress, calllayer+1)
		if err != nil {
			log.Warn("fail to simulate bundle's parent bundle", "hash", bundle.Parent.Hash().String(), "err", err)
			return nil, nil, err
		}
	}
	if bundle.Parent != nil && parentSimBundle != nil && parentSseBundleData != nil {
		// merge parent txs, keep the parent's txs ahead of this bundle's txs
		sseBundleData.SseTxs = append(sseBundleData.SseTxs, parentSseBundleData.SseTxs...)
		bundleGasUsed += parentSimBundle.BundleGasUsed
		bundleGasFees.Add(bundleGasFees, parentSimBundle.BundleGasFees)
	}

	var rpcBalanceBefore, refundRecipientBalanceBefore *uint256.Int

	rpcBalanceBefore = state.GetBalance(rpcBribeAddress)
	if bundle.Parent != nil {
		refundRecipientBalanceBefore = state.GetBalance(bundle.Parent.RefundAddress)
	}

	for _, tx := range bundle.Txs {
		state.SetTxContext(tx.Hash(), *currentTxCount)
		*currentTxCount++

		receipt, err := core.ApplyTransaction(w.chainConfig, w.chain, &w.coinbase, gasPool, state, env.header, tx,
			tempGasUsed, *w.chain.GetVMConfig())
		if err != nil {
			log.Warn("fail to simulate bundle", "hash", bundle.Hash().String(), "err", err)
			return nil, nil, err
		}

		if receipt.Status == types.ReceiptStatusFailed && !containsHash(bundle.RevertingTxHashes, receipt.TxHash) {
			err = errors.New(fmt.Sprintf("transaction execution failed:%s", receipt.GetReturnData()))
			log.Warn("fail to simulate bundle", "hash", bundle.Hash().String(), "err", err)
			return nil, nil, err
		}

		txGasUsed := new(big.Int).SetUint64(receipt.GasUsed)
		effectiveTip, err := tx.EffectiveGasTip(env.header.BaseFee)
		bundleGasUsed += receipt.GasUsed
		if err != nil {
			return nil, nil, err
		}
		txGasFees := new(big.Int).Mul(txGasUsed, effectiveTip)

		if tx.Type() == types.BlobTxType {
			blobFee := new(big.Int).SetUint64(receipt.BlobGasUsed)
			blobFee.Mul(blobFee, receipt.BlobGasPrice)
			txGasFees.Add(txGasFees, blobFee)
			bundleGasUsed += receipt.BlobGasUsed
		}
		bundleGasFees.Add(bundleGasFees, txGasFees)
		// add transaction fees
		rpcTransactionFees.Add(rpcTransactionFees, txGasFees)
		// build sse txData
		curSseTxData, err := rpcSimulator.BuildTxData(bundle, tx, receipt)
		if err != nil {
			return nil, nil, err
		}
		sseBundleData.SseTxs = append(sseBundleData.SseTxs, curSseTxData)
	}

	// add bribe
	rpcBalanceAfter := state.GetBalance(rpcBribeAddress)
	rpcBribeDelta := new(uint256.Int).Sub(rpcBalanceAfter, rpcBalanceBefore).ToBig()

	rpcBundlePrice.Add(rpcBundlePrice, rpcTransactionFees)
	if bundle.Parent != nil {
		refundRecipientBalanceAfter := state.GetBalance(bundle.Parent.RefundAddress)
		refundRecipientDelta := new(uint256.Int).Sub(refundRecipientBalanceAfter, refundRecipientBalanceBefore).ToBig()

		if rpcBribeDelta.Cmp(big.NewInt(0)) < 0 || refundRecipientDelta.Cmp(big.NewInt(0)) < 0 {
			return nil, nil, errors.New("bribe value is too small")
		}

		builderPercent := 0
		if percent, ok := types.BuilderPercentMap[bundle.Parent.RPCID]; ok {
			builderPercent = percent
		} else {
			builderPercent = types.RpcBuilderProfitPercent
		}

		if rpcBribeDelta.Cmp(big.NewInt(0)) > 0 {
			if refundRecipientDelta.Cmp(big.NewInt(0)) <= 0 {
				return nil, nil, errors.New("refund value to recipient is too small")
			}
			a := big.NewInt(0).Mul(refundRecipientDelta,
				big.NewInt(0).Add(
					big.NewInt(int64(100*100*types.RpcBribePercent)),
					big.NewInt(int64((100-types.RpcBribePercent)*(100-bundle.Parent.RefundPercent)*(100-builderPercent)))))
			b := big.NewInt(0).Mul(rpcBribeDelta, big.NewInt(int64(100*(100-types.RpcBribePercent)*bundle.Parent.RefundPercent)))

			a1, _ := new(big.Rat).SetString(a.String())
			b1, _ := new(big.Rat).SetString(b.String())
			c := new(big.Rat)
			c.Quo(a1, b1)

			//Zap.Info("rate", zap.Any("a1", a1), zap.Any("b1", b1), zap.Any("c", c), zap.Any("bundleHash", bundle.Hash()))

			rate1, _ := new(big.Rat).SetString("0.999")
			rate2, _ := new(big.Rat).SetString("1.001")

			if c.Cmp(rate1) <= 0 || c.Cmp(rate2) >= 0 {
				Zap.Error("refund is illegal", zap.String("bundleHash", bundle.Hash().Hex()))
				return nil, nil, errors.New("refund is illegal")
			}

			if calllayer == 1 {
				f, _ := rpcBribeDelta.Float64()
				f = f * 100 / float64(types.RpcBribePercent) / 1e12
				Zap.Info("bribe price",
					zap.String("bundleHash", bundle.Hash().Hex()),
					zap.String("parentHash", bundle.Parent.Hash().Hex()),
					zap.Any("price(1e-6 BNB)", f))
			}
		}

		// calculate bundle price
		rpcBundlePrice.Add(rpcBundlePrice, parentSimBundle.RpcBundlePrice)

		// to builder price
		t := big.NewInt(0).Mul(refundRecipientDelta, big.NewInt(int64(100-bundle.Parent.RefundPercent)))
		t.Div(t, big.NewInt(int64(bundle.Parent.RefundPercent)))
		t.Mul(t, big.NewInt(int64(builderPercent)))
		t.Div(t, big.NewInt(100))

		// to rpc + refund price
		t.Add(t, refundRecipientDelta)
		t.Add(t, rpcBribeDelta)

		if bundle.Counter == 1 {
			t.Mul(t, big.NewInt(2))
		}
		rpcBundlePrice.Add(rpcBundlePrice, t)
	}

	// set chainId, bundleHash and MaxBlockNumber
	sseBundleData.Hash = bundle.Hash().Hex()
	sseBundleData.ChainID = w.chain.Config().ChainID.String()
	sseBundleData.MaxBlockNumber = bundle.MaxBlockNumber

	return &types.SimulatedBundle{
		OriginalBundle: bundle,
		RpcBundlePrice: rpcBundlePrice,
		BundleGasUsed:  bundleGasUsed,
		BundleGasFees:  bundleGasFees,
	}, sseBundleData, nil
}

func (w *worker) WorkerExecuteBundles(env *environment, bundles []*types.Bundle, rpcBribeAddress common.Address) ([]*types.SimulatedBundle, []*define.SseBundleData, error) {
	simResult := make(map[common.Hash]*types.SimulatedBundle)
	simBundleData := make([]*define.SseBundleData, 0)
	var wg sync.WaitGroup
	var mu sync.Mutex
	var err error
	for i, bundle := range bundles {
		wg.Add(1)
		go func(idx int, bundle *types.Bundle, state *state.StateDB) {
			defer wg.Done()

			gasPool := prepareGasPool(env.header.GasLimit)
			var tempGasUsed uint64
			var currentTxIndex int
			var simulateResult *types.SimulatedBundle
			var sseBundleData *define.SseBundleData
			simulateResult, sseBundleData, err = w.WorkerExecuteBundle(env, bundle, state, gasPool, &currentTxIndex, false, false, &tempGasUsed, rpcBribeAddress, 1)
			if err != nil {
				log.Trace("Error computing gas for a simulateBundle", "error", err)
				return
			}
			avg := simulateResult.BundleGasFees.Div(simulateResult.BundleGasFees, big.NewInt(int64(simulateResult.BundleGasUsed)))
			Zap.Info("bundle avg gas price", zap.Any("bundleHash", bundle.Hash()), zap.Any("value", avg))
			if avg.Cmp(big.NewInt(params.GWei)) < 0 {
				err = errors.New("effective bundle gas price too low")
				return
			}

			states := state.GetAllStates()

			if bundle.Hint[types.HintLogs] &&
				(bundle.Parent == nil || bundle.Parent.Hint[types.HintLogs]) &&
				(bundle.Parent == nil || bundle.Parent.Parent == nil || bundle.Parent.Parent.Hint[types.HintLogs]) {
				sseBundleData.State = states
			}

			mu.Lock()
			defer mu.Unlock()
			simResult[bundle.Hash()] = simulateResult
			simBundleData = append(simBundleData, sseBundleData)
		}(i, bundle, env.state.Copy())
	}
	wg.Wait()
	simulatedBundles := make([]*types.SimulatedBundle, 0)
	for _, bundle := range simResult {
		if bundle == nil {
			continue
		}
		simulatedBundles = append(simulatedBundles, bundle)
	}
	return simulatedBundles, simBundleData, err
}
