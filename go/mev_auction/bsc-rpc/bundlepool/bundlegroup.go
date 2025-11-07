package bundlepool

import (
	"context"
	"errors"
	"fmt"
	"github.com/duke-git/lancet/v2/random"
	"github.com/ethereum/go-ethereum-test/base"
	"github.com/ethereum/go-ethereum-test/push"
	"github.com/ethereum/go-ethereum-test/push/define"
	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/common/ms"
	"github.com/ethereum/go-ethereum/core/types"
	"sync"
	"time"
)

// BundleGroup Grouping bundles according to the original transaction
type BundleGroup struct {
	Closed        bool
	CreatedNumber uint64
	Header        *types.Header
	Original      *base.Bundle
	Bundles       map[common.Hash]*base.Bundle // Not contains the original bundle
	bidServer     *ms.Server
	rwMtx         sync.RWMutex
	Slots         uint64
	pool          *BundlePool
	builderServer *push.BuilderServer
	//blockchain    *core.BlockChain
	sseServer *push.SSEServer
}

var functionSelectors = map[string]bool{
	"0x095ea7b3": true, // approve
	"0xa9059cbb": true, // transfer
	"0x23b872dd": true, // transferFrom
	"0xa457c2d7": true, // decreaseAllowance
	"0x39509351": true, // increaseAllowance
}

func (bg *BundleGroup) SendSseData(sseData *define.SseBundleData, bundle *base.Bundle, header *types.Header) {
	// 需要是非公开交易
	if bundle.Counter == 1 || bundle.Counter == 0 {
		bg.sseServer.Send(sseData)
	}
}

// Len get len
func (bg *BundleGroup) Len() int64 {
	bg.rwMtx.RLock()
	defer bg.rwMtx.RUnlock()
	return int64(len(bg.Bundles)) + 1
}

func (bg *BundleGroup) GetBundle(hash common.Hash) *base.Bundle {
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
			bd.RefundCfg = types.RpcBribePercent*1000_000 + random.RandInt(0, 9)*100_000 + bg.Original.RefundPercent*1000 + types.RpcBuilderProfitPercent
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
		go func(hash common.Hash, bundle *base.Bundle) {
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

func (bg *BundleGroup) Simulate(bundle *base.Bundle) (*define.SseBundleData, error) {

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
			sseBundleData.RefundCfg = types.RpcBribePercent*1000_000 + random.RandInt(0, 9)*100_000 + bundle.RefundPercent*1000 + types.RpcBuilderProfitPercent
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

func (bg *BundleGroup) GetMaxBundle(header *types.Header) (bundle *base.Bundle) {
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
		{2000 * time.Millisecond, 2600 * time.Millisecond}, // global 3 endpoint
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
	}
	return try
}
