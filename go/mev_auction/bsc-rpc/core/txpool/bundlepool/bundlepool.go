package bundlepool

import (
	"context"
	"errors"
	"fmt"
	"github.com/coreos/go-systemd/v22/daemon"
	"github.com/duke-git/lancet/v2/random"
	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/common/ms"
	"github.com/ethereum/go-ethereum/core"
	"github.com/ethereum/go-ethereum/core/state"
	"github.com/ethereum/go-ethereum/core/txpool"
	"github.com/ethereum/go-ethereum/core/types"
	"github.com/ethereum/go-ethereum/event"
	"github.com/ethereum/go-ethereum/internal/ethapi"
	"github.com/ethereum/go-ethereum/invalid_tx"
	. "github.com/ethereum/go-ethereum/log/zap"
	"github.com/ethereum/go-ethereum/metrics"
	"github.com/ethereum/go-ethereum/params"
	"github.com/ethereum/go-ethereum/portal"
	"github.com/ethereum/go-ethereum/push"
	"github.com/ethereum/go-ethereum/push/define"
	"github.com/ethereum/go-ethereum/relay"
	"go.uber.org/zap"
	"math/big"
	"strings"
	"sync"
	"time"
)

const (
	// TODO: decide on a good default value
	// bundleSlotSize is used to calculate how many data slots a single bundle
	// takes up based on its size. The slots are used as DoS protection, ensuring
	// that validating a new bundle remains a constant operation (in reality
	// O(maxslots), where max slots are 4 currently).
	bundleSlotSize = 128 * 1024 // 128KB

	maxMinTimestampFromNow = int64(300) // 5 minutes
)

var (
	bundleGauge              = metrics.NewRegisteredGauge("bundlepool/bundles", nil)
	slotsGauge               = metrics.NewRegisteredGauge("bundlepool/slots", nil)
	resetHeaderGauge         = metrics.NewRegisteredGauge("bundlepool/reset/header", nil)
	resetHeaderTimeGauge     = metrics.NewRegisteredGauge("bundlepool/reset/header/time", nil)
	waitingQueueGauge        = metrics.NewRegisteredGauge("bundlepool/waiting/queue", nil)
	bundleLiveSummaryMetrics = metrics.NewRegisteredTimer("bundlepool/bundle/live/summary", nil)
	// 2048 大概最近两分钟的交易
	bundleLiveHistogramMetrics = metrics.GetOrRegisterMyHistogram("bundlepool/bundle/live/histogram", nil, 2048, []float64{3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 60, 120, 240})

	otherTimer                  = metrics.NewRegisteredTimer("bundlepool/bundle/live/summary/other", nil)
	bundleLiveSummaryMetricsMap = map[string]metrics.Timer{}
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
type BlockChain interface {
	// Config retrieves the chain's fork configuration.
	Config() *params.ChainConfig

	// CurrentBlock returns the current head of the chain.
	CurrentBlock() *types.Header

	// GetBlock retrieves a specific block, used during pool resets.
	GetBlock(hash common.Hash, number uint64) *types.Block

	// StateAt returns a state database for a given root hash (generally the head).
	StateAt(root common.Hash) (*state.StateDB, error)
}

type BundleSimulator interface {
	ExecuteBundle(parent *types.Header, bundle *types.Bundle, rpcBribeAddress common.Address) (*big.Int, *define.SseBundleData, error)
}

// BundleGroup Grouping bundles according to the original transaction
type BundleGroup struct {
	Closed        bool
	CreatedNumber uint64
	Header        *types.Header
	Original      *types.Bundle
	Bundles       map[common.Hash]*types.Bundle // Not contains the original bundle
	bidServer     *ms.Server
	rwMtx         sync.RWMutex
	Slots         uint64
	pool          *BundlePool
	builderServer *push.BuilderServer
	blockchain    *core.BlockChain
	sseServer     *push.SSEServer
}

var functionSelectors = map[string]bool{
	"0x095ea7b3": true, // approve
	"0xa9059cbb": true, // transfer
	"0x23b872dd": true, // transferFrom
	"0xa457c2d7": true, // decreaseAllowance
	"0x39509351": true, // increaseAllowance
}

func updateBundleLiveMetrics(arrival time.Time, newHeadTime uint64, rpcId string) {
	newTime := time.Unix(int64(newHeadTime), 0)
	d := newTime.Sub(arrival)
	bundleLiveSummaryMetrics.Update(d)
	bundleLiveHistogramMetrics.Add(d.Seconds())
	if v, ok := bundleLiveSummaryMetricsMap[rpcId]; ok {
		v.Update(d)
	} else {
		otherTimer.Update(d)
	}
}

func (bg *BundleGroup) SendSseData(sseData *define.SseBundleData, bundle *types.Bundle, header *types.Header) {
	if len(sseData.SseTxs) == 1 {
		if len(sseData.SseTxs[0].Logs) == 0 {
			ethapi.TransactionFilterPureGauge.Inc(1)
			return
		}
		if _, ok := functionSelectors[sseData.SseTxs[0].Selector]; ok {
			ethapi.TransactionFilterPureGauge.Inc(1)
			return
		}
	}

	// 需要是非公开交易
	if bundle.Counter == 1 || (bundle.Counter == 0 && !relay.SubServer.IsPublic(bundle.Txs[0].Hash())) {
		bg.sseServer.Send(sseData)
	} else {
		ethapi.TransactionFilterPublicGauge.Inc(1)
	}
}

// Len get len
func (bg *BundleGroup) Len() int64 {
	bg.rwMtx.RLock()
	defer bg.rwMtx.RUnlock()
	return int64(len(bg.Bundles)) + 1
}

func (bg *BundleGroup) GetBundle(hash common.Hash) *types.Bundle {
	bg.rwMtx.RLock()
	defer bg.rwMtx.RUnlock()

	if bg.Closed {
		return nil
	}

	if hash == bg.Original.Hash() {
		return bg.Original
	}
	return bg.Bundles[hash]
}

func (bg *BundleGroup) Exist(hash common.Hash) bool {

	if bg.Closed || bg.GetBundle(hash) == nil {
		return false
	}

	return true
}

func (bg *BundleGroup) GetClosed() bool {
	bg.rwMtx.RLock()
	defer bg.rwMtx.RUnlock()
	return bg.Closed
}

func (bg *BundleGroup) GetSlots() uint64 {
	bg.rwMtx.RLock()
	defer bg.rwMtx.RUnlock()
	return bg.Slots
}

// DeleteBundle delete bundle
func (bg *BundleGroup) DeleteBundle(hash common.Hash) {
	bg.rwMtx.Lock()
	defer bg.rwMtx.Unlock()

	bundle, ok := bg.Bundles[hash]
	if ok {
		delete(bg.Bundles, hash)
		bg.Slots -= numSlots(bundle)
		return
	}
	return
}

func (bg *BundleGroup) SetClosed() {
	bg.rwMtx.Lock()
	defer bg.rwMtx.Unlock()

	bg.Closed = true
}

func (bg *BundleGroup) GetSlot(hash common.Hash) uint64 {
	return numSlots(bg.GetBundle(hash))
}

func (bg *BundleGroup) GetOriginal() common.Hash {
	bg.rwMtx.RLock()
	defer bg.rwMtx.RUnlock()
	return bg.Original.Hash()
}

func (bg *BundleGroup) Reset(header *types.Header) (closed bool, delHash []common.Hash) {

	bg.rwMtx.RLock()
	if bg.Closed ||
		bg.blockchain.GetBlockByHash(header.Hash()).Transaction(bg.Original.Txs[0].Hash()) != nil ||
		bg.Original.MaxBlockNumber <= header.Number.Uint64() {
		bg.rwMtx.RUnlock()
		bg.Closed = true
		return true, nil
	}
	bg.rwMtx.RUnlock()

	var delHashMtx sync.Mutex
	var wg sync.WaitGroup

	var sseData *define.SseBundleData

	bg.rwMtx.Lock()
	wg.Add(1)
	go func() {
		defer wg.Done()
		price, bd, err := bg.pool.simulator.ExecuteBundle(header, bg.Original, types.BidContractAddress)
		if err != nil {
			closed = true
			return
		}
		if bd != nil {
			bd.ProxyBidContract = types.ProxyContractAddress.Hex()
			bd.RefundAddress = bg.Original.RefundAddress.Hex()

			if percent, ok := types.BuilderPercentMap[bg.Original.RPCID]; ok {
				bd.RefundCfg = types.RpcBribePercent*1000_000 + random.RandInt(0, 9)*100_000 + bg.Original.RefundPercent*1000 + percent
			} else {
				bd.RefundCfg = types.RpcBribePercent*1000_000 + random.RandInt(0, 9)*100_000 + bg.Original.RefundPercent*1000 + types.RpcBuilderProfitPercent
			}
			sseData = bd
		}
		bg.Original.Price = price
	}()
	for hash, bundle := range bg.Bundles {
		if bundle.MaxBlockNumber <= header.Number.Uint64() {
			delHashMtx.Lock()
			delHash = append(delHash, hash)
			delHashMtx.Unlock()
			continue
		}
		wg.Add(1)
		go func(hash common.Hash, bundle *types.Bundle) {
			defer wg.Done()
			price, _, err := bg.pool.simulator.ExecuteBundle(header, bundle, types.BidContractAddress)
			if err != nil {
				delHashMtx.Lock()
				delHash = append(delHash, hash)
				delHashMtx.Unlock()
				return
			}
			bundle.Price = price

		}(hash, bundle)
	}
	bg.Header = header
	bg.rwMtx.Unlock()

	wg.Wait()

	if closed {
		bg.SetClosed()
		return true, nil
	}

	bg.bidServer.PushMsgToServer(context.Background(), header)

	if bg.sseServer != nil && sseData != nil && bg.Original.Counter < types.MaxBundleCounter && !bg.Original.IsPrivate() {
		go bg.SendSseData(sseData, bg.Original, header)
	}

	return false, delHash
}

func (bg *BundleGroup) Simulate(bundle *types.Bundle) (*define.SseBundleData, error) {

	bg.rwMtx.RLock()
	if bg.Closed {
		bg.rwMtx.RUnlock()
		return nil, errors.New("bundle group is closed")
	}
	bg.rwMtx.RUnlock()

	for {
		bg.rwMtx.RLock()
		header := bg.Header
		bg.rwMtx.RUnlock()

		price, sseBundleData, err := bg.pool.simulator.ExecuteBundle(header, bundle, types.BidContractAddress)
		if err != nil {
			return nil, err
		}
		bundle.Price = price

		if sseBundleData != nil {
			sseBundleData.ProxyBidContract = types.ProxyContractAddress.Hex()
			sseBundleData.RefundAddress = bundle.RefundAddress.Hex()

			if percent, ok := types.BuilderPercentMap[bundle.RPCID]; ok {
				sseBundleData.RefundCfg = types.RpcBribePercent*1000_000 + random.RandInt(0, 9)*100_000 + bundle.RefundPercent*1000 + percent
			} else {
				sseBundleData.RefundCfg = types.RpcBribePercent*1000_000 + random.RandInt(0, 9)*100_000 + bundle.RefundPercent*1000 + types.RpcBuilderProfitPercent
			}
		}

		bg.rwMtx.Lock()
		if bg.Header == header {
			if bg.Closed {
				bg.rwMtx.Unlock()
				return nil, fmt.Errorf("bundle group is closed")
			}
			if !bg.Closed && bundle != bg.Original {
				if _, exist := bg.Bundles[bundle.Hash()]; exist {
					bg.rwMtx.Unlock()
					return nil, ErrBundleAlreadyExist
				}
				bg.Bundles[bundle.Hash()] = bundle
				bg.Slots += numSlots(bundle)
			}
			bg.rwMtx.Unlock()
			return sseBundleData, nil
		}
		bg.rwMtx.Unlock()
	}
}

func (bg *BundleGroup) GetMaxBundle(header *types.Header) (bundle *types.Bundle) {
	bg.rwMtx.RLock()
	defer bg.rwMtx.RUnlock()

	if header.Number.Cmp(bg.Header.Number) != 0 || bg.Original.State != types.BundleOK {
		return nil
	}

	maxBundle := bg.Original
	for _, bd := range bg.Bundles {
		if bd.State != types.BundleOK {
			continue
		}
		if bd.Price.Cmp(maxBundle.Price) > 0 {
			maxBundle = bd
		} else if bd.Price.Cmp(maxBundle.Price) == 0 {
			if bd.Counter > maxBundle.Counter {
				maxBundle = bd
			}
		}
	}
	//if bg.lastMaxBundle != nil && bg.lastMaxBundle == maxBundle {
	//	return nil
	//} else {
	//	bg.lastMaxBundle = maxBundle
	//}

	return maxBundle
}

func (bg *BundleGroup) Send(ctx context.Context, msg interface{}, num int) (resp interface{}, err error) {
	header := msg.(*types.Header)

	if bg.Original.IsPrivate() {
		bg.builderServer.Send(header, bg.Original, bg.CreatedNumber)
		return nil, nil
	}

	blockTime := time.Duration(header.Time * (1e9))
	now := time.Duration(time.Now().UnixNano())
	if now-blockTime > 2400*time.Millisecond {
		// send original bundle immediately
		bg.builderServer.Send(header, bg.Original, bg.CreatedNumber)
	}

	times := []struct {
		timeOut  time.Duration
		waitTime time.Duration
	}{
		{1000 * time.Millisecond, 1000 * time.Millisecond},
		{2400 * time.Millisecond, 2600 * time.Millisecond}, // global 3 endpoint
		{2600 * time.Millisecond, 2800 * time.Millisecond}, // for us
		{2800 * time.Millisecond, 2880 * time.Millisecond}, // for us
	}
	try := false
	for _, t := range times {
		try = bg.send(header, t.timeOut, t.waitTime)
		if try {
			break
		}
	}

	return nil, nil
}

func (bg *BundleGroup) send(header *types.Header, timeOut time.Duration, waitTime time.Duration) bool {

	try := false

	blockTime := time.Duration(header.Time * (1e9))
	now := time.Duration(time.Now().UnixNano())
	if blockTime > now {
		return try
	}

	longBlock := false
	if header.Number.Uint64()-bg.CreatedNumber >= 2 {
		longBlock = true
	}
	if now-blockTime < timeOut {
		if timeOut == 1000*time.Millisecond && !longBlock {
			return try
		}
		time.Sleep(waitTime - (now - blockTime))

		try = true
		maxBundle := bg.GetMaxBundle(header)
		if maxBundle == nil {
			return try
		}
		if bg.builderServer != nil {
			bg.builderServer.Send(header, maxBundle, bg.CreatedNumber)
		}

		if header.Number.Uint64()-bg.CreatedNumber >= 5 && maxBundle != bg.Original {
			if bg.builderServer != nil {
				bg.builderServer.Send(header, bg.Original, bg.CreatedNumber)
			}
		}
		//if longBlock {
		//	time.Sleep(2998*time.Millisecond - waitTime)
		//}
	}
	return try
}

//------------------------------------------------------------------------------------------------------------

type BundlePool struct {
	config Config

	originalSet  map[common.Hash]struct{}     // map of the original bundle
	bundleGroups map[common.Hash]*BundleGroup // Bundles with the same original transaction are stored in a group
	mu           sync.RWMutex

	simulator     BundleSimulator
	blockchain    *core.BlockChain
	sseServer     *push.SSEServer
	builderServer *push.BuilderServer
	txQueue       *TxQueue
}

func New(config Config, pushServer *push.SSEServer, blockchain *core.BlockChain) *BundlePool {
	// Sanitize the input to ensure no vulnerable gas prices are set
	config = (&config).sanitize()

	pool := &BundlePool{
		config:        config,
		bundleGroups:  make(map[common.Hash]*BundleGroup),
		originalSet:   make(map[common.Hash]struct{}),
		blockchain:    blockchain,
		sseServer:     pushServer,
		builderServer: push.NewBuilderServer(),
		txQueue:       NewTxQueue(),
	}
	pool.builderServer.Start()

	portal.NewSaver().Start()

	for _, v := range types.RpcIdList {
		bundleLiveSummaryMetricsMap[v] = metrics.NewRegisteredTimer("bundlepool/bundle/live/summary/"+v, nil)
	}

	return pool
}

func (p *BundlePool) SetBundleSimulator(simulator BundleSimulator) {
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
func (p *BundlePool) AddBundle(bundle *types.Bundle) error {
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

	header := p.blockchain.CurrentBlock()
	if !ok {
		group = &BundleGroup{
			Header:        header,
			Original:      bundle,
			Bundles:       make(map[common.Hash]*types.Bundle),
			Slots:         numSlots(bundle),
			pool:          p,
			builderServer: p.builderServer,
			blockchain:    p.blockchain,
			sseServer:     p.sseServer,
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

	if err != nil {
		Zap.Info("Receive Bundle", zap.Any("bundleHash", bundle.Hash()), zap.Any("bundle", bundle))
		Zap.Error("simulate failed", zap.String("bundleHash", bundle.Hash().Hex()), zap.String("err", err.Error()))
		return err
	}

	if bundle.State == types.BundleNonceTooHigh {
		// count === 0
		Zap.Info("Receive Bundle", zap.Any("bundleHash", bundle.Hash()), zap.Any("bundle", bundle))
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
	Zap.Info("Receive Bundle", zap.Any("bundleHash", bundle.Hash()), zap.Any("bundle", bundle))

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

func (p *BundlePool) GetBundle(hash common.Hash) *types.Bundle {
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
	Zap.Info("Bundle pool stopped")
	p.builderServer.Stop()
	portal.BundleSaver.Stop()
	return nil
}

func (p *BundlePool) Reset(oldHead, newHead *types.Header) {
	if oldHead == newHead {
		return
	}
	go func() {
		var bundleSize int64
		var bundleSlots int64
		p.mu.RLock()
		defer p.mu.RUnlock()

		for k, _ := range p.originalSet {
			if g, ok := p.bundleGroups[k]; ok {
				bundleSize += g.Len()
				bundleSlots += int64(g.GetSlots())
			}
		}

		bundleGauge.Update(bundleSize)
		slotsGauge.Update(bundleSlots)
	}()

	go p.txQueue.CreateBundles(p, newHead)
	waitingQueueGauge.Update(int64(p.txQueue.queue.Len()))

	var mtx sync.Mutex
	var closeHash []common.Hash

	resetHeaderGauge.Update(newHead.Number.Int64())
	resetHeaderTimeGauge.Update(int64(newHead.Time))
	common.HeadTime.Store(newHead.Time)

	Zap.Info("Receive New Block", zap.Uint64("number", newHead.Number.Uint64()))

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
				if group.Original.Counter == 0 {
					invalid_tx.Server.Put(group.Original.Txs[0].Hash())
				}
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

	go daemon.SdNotify(false, daemon.SdNotifyWatchdog)
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
			Zap.Info("Delete Original Bundle", zap.String("hash", hash.Hex()))
		} else {
			Zap.Info("由于group需要被释放，Delete Bundle", zap.String("hash", hash.Hex()))
		}
		delete(p.originalSet, group.Original.Hash())

		delete(p.bundleGroups, group.Original.Hash()) // 删除原始的
		if newHead != nil {                           // 取消交易不进行记录
			go updateBundleLiveMetrics(group.Original.ArrivalTime, newHead.Time, group.Original.RPCID)
		}
		for key, _ := range group.Bundles {
			delete(p.bundleGroups, key)
		}

		go group.bidServer.Stop()

	} else {
		Zap.Info("Delete Bundle", zap.String("hash", hash.Hex()))

		group.DeleteBundle(hash)
		delete(p.bundleGroups, hash)
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
func numSlots(bundle *types.Bundle) uint64 {
	//return (bundle.Size() + bundleSlotSize - 1) / bundleSlotSize
	if bundle == nil {
		return 0
	}
	return uint64(bundle.Txs.Len()) + numSlots(bundle.Parent)
}

// =====================================================================================================================

type BundleHeap []*types.Bundle

func (h *BundleHeap) Len() int { return len(*h) }

func (h *BundleHeap) Less(i, j int) bool {
	return (*h)[i].Price.Cmp((*h)[j].Price) == -1
}

func (h *BundleHeap) Swap(i, j int) { (*h)[i], (*h)[j] = (*h)[j], (*h)[i] }

func (h *BundleHeap) Push(x interface{}) {
	*h = append(*h, x.(*types.Bundle))
}

func (h *BundleHeap) Pop() interface{} {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[0 : n-1]
	return x
}
