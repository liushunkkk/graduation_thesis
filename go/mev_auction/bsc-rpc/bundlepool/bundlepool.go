package bundlepool

import (
	"context"
	"errors"
	"github.com/ethereum/go-ethereum-test/arbi_detector"
	"github.com/ethereum/go-ethereum-test/base"
	"github.com/ethereum/go-ethereum-test/push"
	"github.com/ethereum/go-ethereum-test/zap_logger"
	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/common/ms"
	"github.com/ethereum/go-ethereum/core"
	"github.com/ethereum/go-ethereum/core/txpool"
	"github.com/ethereum/go-ethereum/core/types"
	"github.com/ethereum/go-ethereum/event"
	"go.uber.org/zap"
	"math/big"
	"strings"
	"sync"
	"time"
)

var (
	// ErrSimulatorMissing is returned if the bundle simulator is missing.
	ErrSimulatorMissing = errors.New("bundle simulator is missing")

	// ErrBundleTimestampTooHigh is returned if the bundle's MinTimestamp is too high.
	ErrBundleTimestampTooHigh = errors.New("bundle MinTimestamp is too high")

	// ErrBundleGasPriceLow is returned if the bundle gas price is too low.
	ErrBundleGasPriceLow = errors.New("bundle gas price is too low")

	// ErrBundleAlreadyExist is returned if the bundle is already contained
	// within the pool.
	ErrBundleAlreadyExist = errors.New("bundle already exist")

	ErrParentBundleNotFound = errors.New("parent bundle not found")

	ErrBundleTooDeep = errors.New("bundle nesting is too deep")
)

// BlockChain defines the minimal set of methods needed to back a tx pool with
// a chain. Exists to allow mocking the live chain out of tests.
//type BlockChain interface {
//	// Config retrieves the chain's fork configuration.
//	Config() *params.ChainConfig
//
//	// CurrentBlock returns the current head of the chain.
//	CurrentBlock() *types.Header
//
//	// GetBlock retrieves a specific block, used during pool resets.
//	GetBlock(hash common.Hash, number uint64) *types.Block
//
//	// StateAt returns a state database for a given root hash (generally the head).
//	StateAt(root common.Hash) (*state.StateDB, error)
//}

//------------------------------------------------------------------------------------------------------------

type BundlePool struct {
	originalSet  map[common.Hash]struct{}     // map of the original bundle
	bundleGroups map[common.Hash]*BundleGroup // Bundles with the same original transaction are stored in a group
	mu           sync.RWMutex

	simulator base.BundleSimulator
	//blockchain    *core.BlockChain
	sseServer     *push.SSEServer
	builderServer *push.BuilderServer
	txQueue       *TxQueue
}

func New(pushServer *push.SSEServer, blockchain *core.BlockChain) *BundlePool {
	pool := &BundlePool{
		bundleGroups: make(map[common.Hash]*BundleGroup),
		originalSet:  make(map[common.Hash]struct{}),
		//blockchain:    blockchain,
		sseServer:     pushServer,
		builderServer: push.NewBuilderServer(),
		txQueue:       NewTxQueue(),
	}
	pool.builderServer.Start()

	//portal.NewSaver().Start()

	//for _, v := range types.RpcIdList {
	//	bundleLiveSummaryMetricsMap[v] = metrics.NewRegisteredTimer("bundlepool/bundle/live/summary/"+v, nil)
	//}

	pool.SetBundleSimulator(base.NewBundleSimulator())

	return pool
}

func (p *BundlePool) SetBundleSimulator(simulator base.BundleSimulator) {
	p.simulator = simulator
}

func (p *BundlePool) Init(gasTip uint64, head *types.Header, reserve txpool.AddressReserver) error {
	return nil
}

func (p *BundlePool) FilterBundle(bundle *types.Bundle) bool {
	for _, tx := range bundle.Txs {
		if !p.filter(tx) {
			return false
		}
	}
	return true
}

func (p *BundlePool) IsCancelTx(from1 common.Address, tx1 *types.Transaction) (bundlue0Hash common.Hash, tx0 common.Hash, ok bool) {
	if !IsSelfTransfer0Value(from1, tx1) {
		return common.Hash{}, common.Hash{}, false
	}

	p.mu.RLock()
	defer p.mu.RUnlock()

	for _, group := range p.bundleGroups {
		if group.Original.Counter == 0 {
			if NonceIsEqual(group.Original.From, group.Original.Txs[0], from1, tx1) {
				return group.Original.Hash(), group.Original.Txs[0].Hash(), true
			}
		}
	}

	b := p.txQueue.DeleteBundle(from1, tx1.Nonce())
	if b != nil {
		return b.Hash(), b.Txs[0].Hash(), true
	}
	return common.Hash{}, common.Hash{}, false
}

func IsSelfTransfer0Value(from1 common.Address, tx1 *types.Transaction) bool {
	if len(tx1.Data()) == 0 && from1.Hex() == tx1.To().Hex() && (tx1.Value() == nil || tx1.Value().Int64() == 0) {
		return true
	}
	return false
}

func NonceIsEqual(from0 common.Address, tx0 *types.Transaction, from1 common.Address, tx1 *types.Transaction) bool {
	if tx0.Nonce() == tx1.Nonce() && from0.Hex() == from1.Hex() {
		return true
	}
	return false
}

// AddBundle adds a mev bundle to the pool
func (p *BundlePool) AddBundle(bundle *base.Bundle) error {
	//if bundle.Counter == 0 {
	//	if bundleHash, txHash, ok := p.IsCancelTx(bundle.From, bundle.Txs[0]); ok {
	//
	//		p.mu.RLock()
	//		if g, exist := p.bundleGroups[bundleHash]; exist {
	//			g.SetClosed()
	//		}
	//		p.mu.RUnlock()
	//
	//		p.PruneBundle(bundleHash, nil)
	//		invalid_tx.Server.Put(txHash)
	//		invalid_tx.Server.Put(bundle.Txs[0].Hash())
	//		return nil
	//	}
	//}

	if p.simulator == nil {
		return ErrSimulatorMissing
	}

	// Group bundles with the same original transaction
	p.mu.RLock()
	group, ok := p.bundleGroups[bundle.ParentHash]
	if !ok && (bundle.ParentHash != common.Hash{}) {
		p.mu.RUnlock()
		return ErrParentBundleNotFound
	}
	if !ok {
		_, exist := p.originalSet[bundle.Hash()]
		if exist {
			p.mu.RUnlock()
			return ErrBundleAlreadyExist
		}
	}
	p.mu.RUnlock()

	//header := p.blockchain.CurrentBlock()
	header := base.CurrentHeader
	if !ok {
		group = &BundleGroup{
			Header:        header,
			Original:      bundle,
			Bundles:       make(map[common.Hash]*base.Bundle),
			Slots:         numSlots(bundle),
			pool:          p,
			builderServer: p.builderServer,
			//blockchain:    p.blockchain,
			sseServer: p.sseServer,
		}
		group.CreatedNumber = group.Header.Number.Uint64()
	} else {
		parentBundle := group.GetBundle(bundle.ParentHash)
		if parentBundle == nil {
			return ErrParentBundleNotFound
		}
		bundle.RPCID = parentBundle.RPCID
		bundle.Parent = parentBundle
		bundle.Counter = parentBundle.Counter + 1
		bundle.PrivacyPeriod = parentBundle.PrivacyPeriod
		bundle.PrivacyBuilder = parentBundle.PrivacyBuilder
		bundle.BroadcastBuilder = parentBundle.BroadcastBuilder
		bundle.UserId = parentBundle.UserId

		if bundle.Counter > types.MaxBundleCounter {
			return ErrBundleTooDeep
		}
		bundle.MaxBlockNumber = min(parentBundle.MaxBlockNumber, bundle.MaxBlockNumber)
	}

	hash := bundle.Hash()
	if ok && group.Exist(hash) {
		return ErrBundleAlreadyExist
	}

	bundleData, err := group.Simulate(bundle)
	if bundle.Parent != nil && bundleData != nil {
		var arbiReq []arbi_detector.ArbiRequest
		for _, t := range bundleData.SseTxs {
			arbiReq = append(arbiReq, arbi_detector.ArbiRequest{
				TxHash:      t.Hash,
				TxJson:      t.Tx,
				ReceiptJson: t.ReceiptJson,
			})
		}
		arbiResult, _ := arbi_detector.DetectBatchArbi(arbiReq)
		isArbi := false
		if arbiResult != nil {
			for _, arbi := range arbiResult.Results {
				if arbi.ArbitragePercent > 0.4 { // 适当放宽
					isArbi = true
				}
			}
			if len(arbiResult.Results) > 1 && !isArbi { // 如果判断了，但是没有符合要求就跳过
				return errors.New("not arbi")
			}
		}
	}

	if err != nil {
		//zap_logger.Zap.Info("Receive Bundle", zap.Any("bundleHash", bundle.Hash()), zap.Any("bundle", bundle))
		zap_logger.Zap.Info("Receive Bundle", zap.Any("bundleHash", bundle.Hash()), zap.Any("blockNumber", bundle.MaxBlockNumber), zap.Any("txs", bundle.GetTxHashes()))
		zap_logger.Zap.Error("simulate failed", zap.String("bundleHash", bundle.Hash().Hex()), zap.String("err", err.Error()))
		return err
	}

	if bundle.State == types.BundleNonceTooHigh {
		// count === 0
		zap_logger.Zap.Info("Receive Bundle", zap.Any("bundleHash", bundle.Hash()), zap.Any("bundle", bundle))
		p.txQueue.InsertNonceTooHighTxToQueue(bundle.From, bundle)
		return nil
	}

	bundle.Erc20Tx = false
	if bundle.Counter == 0 && bundleData != nil && len(bundleData.SseTxs) == 1 {
		b := functionSelectors[strings.ToLower(bundleData.SseTxs[0].Selector)]
		if b {
			bundle.Erc20Tx = true
		}
	}
	//zap_logger.Zap.Info("Receive Bundle", zap.Any("bundleHash", bundle.Hash()), zap.Any("bundle", bundle))
	zap_logger.Zap.Info("Receive Bundle", zap.Any("bundleHash", bundle.Hash()), zap.Any("blockNumber", bundle.MaxBlockNumber), zap.Any("txs", bundle.GetTxHashes()))

	if !ok {
		// original bundle handle logic,need to start ms server
		group.bidServer, err = ms.NewSvr(bundle.Hash().String(), group.Send, nil)
		if err != nil {
			// bundle has existed
			return nil
		}
		group.bidServer.ActionGoroutineNum = 1
		group.bidServer.Go()

		group.bidServer.PushMsgToServer(context.Background(), group.Header)
	}

	p.mu.Lock()
	p.bundleGroups[hash] = group
	if !ok {
		// original bundle
		p.originalSet[hash] = struct{}{}
	}
	p.mu.Unlock()

	if p.sseServer != nil && bundleData != nil && bundle.Counter < types.MaxBundleCounter && !bundle.IsPrivate() {
		go group.SendSseData(bundleData, bundle, header)
	}

	return nil
}

func (p *BundlePool) GetBundle(hash common.Hash) *base.Bundle {
	p.mu.RLock()
	defer p.mu.RUnlock()

	g := p.bundleGroups[hash]
	if g == nil {
		return nil
	}

	return p.bundleGroups[hash].GetBundle(hash)
}

func (p *BundlePool) PruneBundle(hash common.Hash, newHead *types.Header) {
	p.mu.Lock()
	p.deleteBundle(hash, newHead)
	p.mu.Unlock()
}

func (p *BundlePool) PendingBundles(blockNumber uint64, blockTimestamp uint64) []*types.Bundle {
	//p.mu.Lock()
	//defer p.mu.Unlock()
	//
	//ret := make([]*types.Bundle, 0)
	//for hash, bundle := range p.bundles {
	//	// Prune outdated bundles
	//	if (bundle.MaxTimestamp != 0 && blockTimestamp > bundle.MaxTimestamp) ||
	//		(bundle.MaxBlockNumber != 0 && blockNumber > bundle.MaxBlockNumber) {
	//		p.deleteBundle(hash)
	//		continue
	//	}
	//
	//	// Roll over future bundles
	//	if bundle.MinTimestamp != 0 && blockTimestamp < bundle.MinTimestamp {
	//		continue
	//	}
	//
	//	// return the ones that are in time
	//	ret = append(ret, bundle)
	//}
	//
	//bundleGauge.Update(int64(len(p.bundles)))
	//slotsGauge.Update(int64(p.slots))
	//return ret
	return nil
}

// AllBundles returns all the bundles currently in the pool
func (p *BundlePool) AllBundles() []*types.Bundle {
	return nil
}

func (p *BundlePool) Filter(tx *types.Transaction) bool {
	return false
}

func (p *BundlePool) Close() error {
	zap_logger.Zap.Info("Bundle pool stopped")
	p.builderServer.Stop()
	//portal.BundleSaver.Stop()
	return nil
}

func (p *BundlePool) Reset(oldHead, newHead *types.Header) {
	//if oldHead == newHead {
	//	return
	//}
	//go func() {
	//	var bundleSize int64
	//	var bundleSlots int64
	//	p.mu.RLock()
	//	defer p.mu.RUnlock()
	//
	//	for k, _ := range p.originalSet {
	//		if g, ok := p.bundleGroups[k]; ok {
	//			bundleSize += g.Len()
	//			bundleSlots += int64(g.GetSlots())
	//		}
	//	}
	//
	//	bundleGauge.Update(bundleSize)
	//	slotsGauge.Update(bundleSlots)
	//}()

	//go p.txQueue.CreateBundles(p, newHead)

	var mtx sync.Mutex
	var closeHash []common.Hash

	common.HeadTime.Store(newHead.Time)

	zap_logger.Zap.Info("Receive New Block", zap.Uint64("number", newHead.Number.Uint64()))

	var wg sync.WaitGroup
	p.mu.RLock()
	for hash, _ := range p.originalSet {
		g := p.bundleGroups[hash]
		wg.Add(1)
		go func(group *BundleGroup) {
			defer wg.Done()
			closed, delHash := group.Reset(newHead)
			mtx.Lock()
			if closed {
				closeHash = append(closeHash, group.GetOriginal())
				//if group.Original.Counter == 0 {
				//	invalid_tx.Server.Put(group.Original.Txs[0].Hash())
				//}
			}
			closeHash = append(closeHash, delHash...)
			mtx.Unlock()
		}(g)
	}

	p.mu.RUnlock()
	wg.Wait()

	for _, hash := range closeHash {
		p.PruneBundle(hash, newHead)
	}

	//go daemon.SdNotify(false, daemon.SdNotifyWatchdog)
}

// SetGasTip updates the minimum price required by the subpool for a new
// transaction, and drops all transactions below this threshold.
func (p *BundlePool) SetGasTip(tip *big.Int) {}

func (p *BundlePool) SetMaxGas(maxGas uint64) {
}

// Has returns an indicator whether subpool has a transaction cached with the
// given hash.
func (p *BundlePool) Has(hash common.Hash) bool {
	return p.GetBundle(hash) != nil
}

// Get returns a transaction if it is contained in the pool, or nil otherwise.
func (p *BundlePool) Get(hash common.Hash) *types.Transaction {
	return nil
}

// Add enqueues a batch of transactions into the pool if they are valid. Due
// to the large transaction churn, add may postpone fully integrating the tx
// to a later point to batch multiple ones together.
func (p *BundlePool) Add(txs []*types.Transaction, local bool, sync bool) []error {
	return nil
}

// Pending retrieves all currently processable transactions, grouped by origin
// account and sorted by nonce.
func (p *BundlePool) Pending(filter txpool.PendingFilter) map[common.Address][]*txpool.LazyTransaction {
	return nil
}

// SubscribeTransactions subscribes to new transaction events.
func (p *BundlePool) SubscribeTransactions(ch chan<- core.NewTxsEvent, reorgs bool) event.Subscription {
	return nil
}

// SubscribeReannoTxsEvent should return an event subscription of
// ReannoTxsEvent and send events to the given channel.
func (p *BundlePool) SubscribeReannoTxsEvent(chan<- core.ReannoTxsEvent) event.Subscription {
	return nil
}

// Nonce returns the next nonce of an account, with all transactions executable
// by the pool already applied on topool.
func (p *BundlePool) Nonce(addr common.Address) uint64 {
	return 0
}

// Stats retrieves the current pool stats, namely the number of pending and the
// number of queued (non-executable) transactions.
func (p *BundlePool) Stats() (int, int) {
	return 0, 0
}

// Content retrieves the data content of the transaction pool, returning all the
// pending as well as queued transactions, grouped by account and sorted by nonce.
func (p *BundlePool) Content() (map[common.Address][]*types.Transaction, map[common.Address][]*types.Transaction) {
	return make(map[common.Address][]*types.Transaction), make(map[common.Address][]*types.Transaction)
}

// ContentFrom retrieves the data content of the transaction pool, returning the
// pending as well as queued transactions of this address, grouped by nonce.
func (p *BundlePool) ContentFrom(addr common.Address) ([]*types.Transaction, []*types.Transaction) {
	return []*types.Transaction{}, []*types.Transaction{}
}

// Locals retrieves the accounts currently considered local by the pool.
func (p *BundlePool) Locals() []common.Address {
	return []common.Address{}
}

// Status returns the known status (unknown/pending/queued) of a transaction
// identified by their hashes.
func (p *BundlePool) Status(hash common.Hash) txpool.TxStatus {
	return txpool.TxStatusUnknown
}

func (p *BundlePool) filter(tx *types.Transaction) bool {
	switch tx.Type() {
	case types.LegacyTxType, types.AccessListTxType, types.DynamicFeeTxType:
		return true
	default:
		return false
	}
}

// deleteBundle Delete the specified bundle. If the bundle is original, the entire bundleGroup is deleted
// It assumes that the caller holds the pool's lock.
func (p *BundlePool) deleteBundle(hash common.Hash, newHead *types.Header) {
	group := p.bundleGroups[hash]
	if group == nil {
		return
	}

	if group.GetClosed() {
		if group.Original.Hash() == hash {
			zap_logger.Zap.Info("Delete Original Bundle", zap.String("hash", hash.Hex()))
		} else {
			zap_logger.Zap.Info("由于group需要被释放，Delete Bundle", zap.String("hash", hash.Hex()))
		}
		delete(p.originalSet, group.Original.Hash())

		delete(p.bundleGroups, group.Original.Hash()) // 删除原始的
		//if newHead != nil {                           // 取消交易不进行记录
		//	go updateBundleLiveMetrics(group.Original.ArrivalTime, newHead.Time, group.Original.RPCID)
		//}
		for key, _ := range group.Bundles {
			delete(p.bundleGroups, key)
		}

		go group.bidServer.Stop()

	} else {
		zap_logger.Zap.Info("Delete Bundle", zap.String("hash", hash.Hex()))

		group.DeleteBundle(hash)
		delete(p.bundleGroups, hash)
	}
}

func (p *BundlePool) PrintState() {
	ticker := time.NewTicker(1 * time.Second)
	defer ticker.Stop()

	for range ticker.C {
		zap_logger.Zap.Info("[pool-state]", zap.Any("group", len(p.originalSet)), zap.Any("bundles", len(p.bundleGroups)))
	}
}

// minimalBundleGasPrice return the lowest gas price from the pool.
//func (p *BundlePool) minimalBundleGasPrice() *big.Int {
//	for len(p.bundleHeap) != 0 {
//		leastPriceBundleHash := p.bundleHeap[0].Hash()
//		if bundle, ok := p.bundles[leastPriceBundleHash]; ok {
//			return bundle.Price
//		}
//		heap.Pop(&p.bundleHeap)
//	}
//	return new(big.Int)
//}

// =====================================================================================================================

// numSlots calculates the number of slots needed for a single bundle.
func numSlots(bundle *base.Bundle) uint64 {
	//return (bundle.Size() + bundleSlotSize - 1) / bundleSlotSize
	if bundle == nil {
		return 0
	}
	return uint64(bundle.Txs.Len()) + numSlots(bundle.Parent)
}

// =====================================================================================================================

type BundleHeap []*base.Bundle

func (h *BundleHeap) Len() int { return len(*h) }

func (h *BundleHeap) Less(i, j int) bool {
	return (*h)[i].Price.Cmp((*h)[j].Price) == -1
}

func (h *BundleHeap) Swap(i, j int) { (*h)[i], (*h)[j] = (*h)[j], (*h)[i] }

func (h *BundleHeap) Push(x interface{}) {
	*h = append(*h, x.(*base.Bundle))
}

func (h *BundleHeap) Pop() interface{} {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[0 : n-1]
	return x
}
