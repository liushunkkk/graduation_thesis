package bundlepool

import (
	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/core/types"
	. "github.com/ethereum/go-ethereum/log/zap"
	lru "github.com/hashicorp/golang-lru/v2"
	"go.uber.org/zap"
	"slices"
	"sync"
)

var QueueCount = 1000
var queueSize = 50

type TxQueue struct {
	queue    *lru.Cache[common.Address, []*types.Bundle]
	queueMtx sync.Mutex
}

func NewTxQueue() *TxQueue {
	t := &TxQueue{}
	t.queue, _ = lru.New[common.Address, []*types.Bundle](QueueCount)
	return t
}

// InsertNonceTooHighTxToQueue must insert count == 0's bundle
func (txQueue *TxQueue) InsertNonceTooHighTxToQueue(from common.Address, bundle *types.Bundle) {
	txQueue.queueMtx.Lock()
	defer txQueue.queueMtx.Unlock()

	list, exist := txQueue.queue.Get(from)
	if exist {
		i, b := slices.BinarySearchFunc(list, bundle, func(bundle1 *types.Bundle, bundle2 *types.Bundle) int {
			return int(bundle1.Txs[0].Nonce() - bundle2.Txs[0].Nonce())
		})
		if b {
			// tx repeated
			list[i] = bundle
		} else {
			txQueue.queue.Add(from, slices.Insert(list, i, bundle))
			arr, _ := txQueue.queue.Get(from)
			if len(arr) > queueSize {
				txQueue.queue.Add(from, arr[0:len(arr)-1])
			}
		}
	} else {
		txQueue.queue.Add(from, []*types.Bundle{bundle})
	}

	Zap.Info("Insert NonceTooHighTx To Queue", zap.String("from", from.Hex()), zap.String("bundleHash", bundle.Hash().Hex()))
}

func (txQueue *TxQueue) TakeBundles(from common.Address) []*types.Bundle {
	txQueue.queueMtx.Lock()
	defer txQueue.queueMtx.Unlock()

	list, ok := txQueue.queue.Get(from)
	if ok {
		return append([]*types.Bundle{}, list...)
	}
	return nil
}

// DeleteBundle delete one bundle which bundle's nonce == nonce
func (txQueue *TxQueue) DeleteBundle(from common.Address, nonce uint64) *types.Bundle {
	txQueue.queueMtx.Lock()
	defer txQueue.queueMtx.Unlock()
	list, ok := txQueue.queue.Get(from)
	if !ok {
		return nil
	}
	var i int
	for i = 0; i < len(list); i++ {
		if list[i].Txs[0].Nonce() == nonce {
			break
		}
	}

	if i == len(list) {
		return nil
	}

	ans := list[i]

	tmp := slices.Delete(list, i, i+1)
	if len(tmp) > 0 {
		txQueue.queue.Add(from, tmp)
	} else {
		txQueue.queue.Remove(from)
	}

	Zap.Info("delete from txQueue", zap.String("bundleHash", ans.Hash().Hex()), zap.Any("bundle", ans))

	return ans
}

// DeleteBundles delete bundles than bundle's nonce <= nonce
func (txQueue *TxQueue) DeleteBundles(from common.Address, nonce uint64) {
	txQueue.queueMtx.Lock()
	defer txQueue.queueMtx.Unlock()
	list, ok := txQueue.queue.Get(from)
	if !ok {
		return
	}

	var i int
	for i = 0; i < len(list); i++ {
		if list[i].Txs[0].Nonce() > nonce {
			break
		}
	}

	if i >= 1 {
		Zap.Info("delete from txQueue", zap.String("bundleHash", list[i-1].Hash().Hex()))
	}
	tmp := slices.Delete(list, 0, i)
	if len(tmp) > 0 {
		txQueue.queue.Add(from, tmp)
	} else {
		txQueue.queue.Remove(from)
	}
}

func (txQueue *TxQueue) CreateBundles(p *BundlePool, header *types.Header) {
	txQueue.queueMtx.Lock()

	for _, from := range txQueue.queue.Keys() {
		v, _ := txQueue.queue.Get(from)
		bundles := append([]*types.Bundle{}, v...)
		go func() {
			for _, bundle := range bundles {
				if header.Number.Uint64() >= bundle.MaxBlockNumber {
					txQueue.DeleteBundles(from, bundle.Txs[0].Nonce())
					continue
				}
				_, _, err := p.simulator.ExecuteBundle(header, bundle, types.BidContractAddress)
				if err != nil {
					txQueue.DeleteBundles(from, bundle.Txs[0].Nonce())
					continue
				} else {
					if bundle.State != types.BundleNonceTooHigh {
						txQueue.DeleteBundles(from, bundle.Txs[0].Nonce())

						Zap.Info("take bundle from txQueue,then AddBundle to bundlePool", zap.String("bundleHash", bundle.Hash().Hex()))
						p.AddBundle(bundle)
					}
					return
				}
			}
		}()
	}
	txQueue.queueMtx.Unlock()
}
