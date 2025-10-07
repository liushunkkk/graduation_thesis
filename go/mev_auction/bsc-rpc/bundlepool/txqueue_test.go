package bundlepool

import (
	"fmt"
	"github.com/agiledragon/gomonkey/v2"
	"github.com/duke-git/lancet/v2/random"
	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/core/types"
	"github.com/ethereum/go-ethereum/push/define"
	. "github.com/smartystreets/goconvey/convey"
	"math/big"
	"reflect"
	"testing"
	"time"
)

func TestTxQueue_InsertNonceTooHighTxToQueue(t *testing.T) {

	Convey("insert too many", t, func() {
		queue := NewTxQueue()
		for i := 0; i < 1100; i++ {
			from := common.BytesToAddress(random.RandBytes(20))
			bd := &types.Bundle{
				Txs: types.Transactions{types.NewTx(&types.DynamicFeeTx{Nonce: uint64(0)})},
			}
			queue.InsertNonceTooHighTxToQueue(from, bd)
		}

		So(len(queue.queue.Keys()), ShouldEqual, QueueCount)
	})
	Convey("insert too many1", t, func() {
		queue := NewTxQueue()
		from := common.HexToAddress("0x420b72ACE04A0cd54516e081BA6ACE885d53287A")

		for i := 0; i < 100; i++ {
			bd := &types.Bundle{
				Txs: types.Transactions{types.NewTx(&types.DynamicFeeTx{Nonce: uint64(i)})},
			}
			queue.InsertNonceTooHighTxToQueue(from, bd)

			if i < 50 {
				So(len(queue.TakeBundles(from)), ShouldEqual, i+1)
			} else {
				So(len(queue.TakeBundles(from)), ShouldEqual, queueSize)
			}
		}
	})

	Convey("insert successfully", t, func() {
		queue := NewTxQueue()

		from := common.HexToAddress("0x420b72ACE04A0cd54516e081BA6ACE885d53287A")
		tx := types.NewTx(&types.DynamicFeeTx{Nonce: 100})
		queue.InsertNonceTooHighTxToQueue(from, &types.Bundle{
			Txs: types.Transactions{tx},
		})

		bds := queue.TakeBundles(from)
		So(len(bds), ShouldEqual, 1)
		So(bds[0].Txs[0].Nonce(), ShouldEqual, 100)
	})

	Convey("insert repeated", t, func() {
		queue := NewTxQueue()

		from := common.HexToAddress("0x420b72ACE04A0cd54516e081BA6ACE885d53287A")
		tx := types.NewTx(&types.DynamicFeeTx{Nonce: 100})
		bundle := &types.Bundle{
			Txs: types.Transactions{tx},
		}
		queue.InsertNonceTooHighTxToQueue(from, bundle)
		queue.InsertNonceTooHighTxToQueue(from, bundle)

		bds := queue.TakeBundles(from)
		So(len(bds), ShouldEqual, 1)
		So(bds[0].Txs[0].Nonce(), ShouldEqual, 100)
	})

	Convey("insert three", t, func() {
		queue := NewTxQueue()

		from := common.HexToAddress("0x420b72ACE04A0cd54516e081BA6ACE885d53287A")
		tx3 := types.NewTx(&types.DynamicFeeTx{Nonce: 101})
		tx1 := types.NewTx(&types.DynamicFeeTx{Nonce: 99})
		tx2 := types.NewTx(&types.DynamicFeeTx{Nonce: 100})

		Convey("inset", func() {
			queue.InsertNonceTooHighTxToQueue(from, &types.Bundle{
				Txs: types.Transactions{tx1},
			})
			queue.InsertNonceTooHighTxToQueue(from, &types.Bundle{
				Txs: types.Transactions{tx2},
			})
			queue.InsertNonceTooHighTxToQueue(from, &types.Bundle{
				Txs: types.Transactions{tx3},
			})

			bds := queue.TakeBundles(from)
			So(len(queue.TakeBundles(from)), ShouldEqual, 3)
			So(bds[0].Txs[0].Nonce(), ShouldEqual, 99)
			So(bds[1].Txs[0].Nonce(), ShouldEqual, 100)
			So(bds[2].Txs[0].Nonce(), ShouldEqual, 101)
		})

		Convey("sort", func() {
			queue.InsertNonceTooHighTxToQueue(from, &types.Bundle{
				Txs: types.Transactions{tx1},
			})
			So(len(queue.TakeBundles(from)), ShouldEqual, 1)

			queue.InsertNonceTooHighTxToQueue(from, &types.Bundle{
				Txs: types.Transactions{tx3},
			})
			So(len(queue.TakeBundles(from)), ShouldEqual, 2)

			queue.InsertNonceTooHighTxToQueue(from, &types.Bundle{
				Txs: types.Transactions{tx2},
			})
			So(len(queue.TakeBundles(from)), ShouldEqual, 3)

			bds := queue.TakeBundles(from)
			So(bds[0].Txs[0].Nonce(), ShouldEqual, 99)
			So(bds[1].Txs[0].Nonce(), ShouldEqual, 100)
			So(bds[2].Txs[0].Nonce(), ShouldEqual, 101)

			queue.DeleteBundles(from, 100)
			bds = queue.TakeBundles(from)
			So(len(bds), ShouldEqual, 1)
			So(bds[0].Txs[0].Nonce(), ShouldEqual, 101)

			queue.DeleteBundles(from, 101)
			bds = queue.TakeBundles(from)
			So(len(bds), ShouldEqual, 0)
		})

		Convey("del all", func() {
			queue.InsertNonceTooHighTxToQueue(from, &types.Bundle{
				Txs: types.Transactions{tx1},
			})
			queue.InsertNonceTooHighTxToQueue(from, &types.Bundle{
				Txs: types.Transactions{tx3},
			})
			queue.InsertNonceTooHighTxToQueue(from, &types.Bundle{
				Txs: types.Transactions{tx2},
			})

			queue.DeleteBundles(from, 101)
			bds := queue.TakeBundles(from)
			So(len(bds), ShouldEqual, 0)
		})

	})
}

type TxQueueSimulator struct {
}

func (s *TxQueueSimulator) ExecuteBundle(parent *types.Header, bundle *types.Bundle, systemAddress common.Address) (*big.Int, *define.SseBundleData, error) {
	if bundle == ErrBundle {
		bundle.State = types.BundleErr
		return nil, nil, fmt.Errorf("error")
	} else if bundle == NonceTooHighBundle {
		bundle.State = types.BundleNonceTooHigh
		return nil, nil, nil
	} else if bundle == FeeCapTooLowBundle {
		bundle.State = types.BundleFeeCapTooLow
		return nil, nil, nil
	} else if bundle == BundleOK {
		bundle.State = types.BundleOK
		return big.NewInt(1), &define.SseBundleData{}, nil
	}
	return nil, nil, nil
}

var BundleOK = &types.Bundle{Txs: types.Transactions{types.NewTx(&types.DynamicFeeTx{Nonce: 99})}, MaxBlockNumber: 200}
var ErrBundle = &types.Bundle{Txs: types.Transactions{types.NewTx(&types.DynamicFeeTx{Nonce: 100})}, MaxBlockNumber: 200}
var NonceTooHighBundle = &types.Bundle{Txs: types.Transactions{types.NewTx(&types.DynamicFeeTx{Nonce: 101})}, MaxBlockNumber: 200}
var FeeCapTooLowBundle = &types.Bundle{Txs: types.Transactions{types.NewTx(&types.DynamicFeeTx{Nonce: 102})}, MaxBlockNumber: 200}
var TimeOutBundle = &types.Bundle{Txs: types.Transactions{types.NewTx(&types.DynamicFeeTx{Nonce: 98})}, MaxBlockNumber: 100}

func TestTxQueue_CreateBundles(t *testing.T) {

	pool := &BundlePool{
		simulator: &TxQueueSimulator{},
	}

	count := 0
	patch := gomonkey.ApplyMethod(reflect.TypeOf(pool), "AddBundle", func(p *BundlePool, bundle *types.Bundle) error {
		if bundle == ErrBundle {
			return fmt.Errorf("error")
		}
		if bundle == NonceTooHighBundle {
			return nil
		}
		if bundle == FeeCapTooLowBundle {
			count++
		}
		if bundle == BundleOK {
			count++
		}
		return nil
	})

	defer patch.Reset()

	Convey("Ok bundle", t, func() {
		queue := NewTxQueue()
		from := common.HexToAddress("0x420b72ACE04A0cd54516e081BA6ACE885d53287A")
		queue.InsertNonceTooHighTxToQueue(from, BundleOK)
		queue.InsertNonceTooHighTxToQueue(from, NonceTooHighBundle)

		queue.CreateBundles(pool, &types.Header{Number: big.NewInt(100)})
		time.Sleep(200 * time.Millisecond)
		bds := queue.TakeBundles(from)
		So(len(bds), ShouldEqual, 1)
		So(count, ShouldEqual, 1)

		queue.CreateBundles(pool, &types.Header{Number: big.NewInt(100)})
		time.Sleep(1000 * time.Millisecond)
		bds = queue.TakeBundles(from)
		So(len(bds), ShouldEqual, 1)
		So(count, ShouldEqual, 1)

		count = 0
	})

	Convey("Ok bundle", t, func() {
		queue := NewTxQueue()
		from := common.HexToAddress("0x420b72ACE04A0cd54516e081BA6ACE885d53287A")
		queue.InsertNonceTooHighTxToQueue(from, TimeOutBundle)
		queue.InsertNonceTooHighTxToQueue(from, ErrBundle)
		queue.InsertNonceTooHighTxToQueue(from, FeeCapTooLowBundle)

		queue.CreateBundles(pool, &types.Header{Number: big.NewInt(100)})
		time.Sleep(200 * time.Millisecond)
		bds := queue.TakeBundles(from)
		So(len(bds), ShouldEqual, 0)
		So(count, ShouldEqual, 1)

		count = 0
	})
}

func TestTxQueue_DeleteBundle(t *testing.T) {
	Convey("delete one bundle", t, func() {
		queue := NewTxQueue()
		from := common.HexToAddress("0x420b72ACE04A0cd54516e081BA6ACE885d53287A")
		queue.InsertNonceTooHighTxToQueue(from, TimeOutBundle)
		queue.InsertNonceTooHighTxToQueue(from, ErrBundle)
		queue.InsertNonceTooHighTxToQueue(from, FeeCapTooLowBundle)

		b := queue.DeleteBundle(from, 1)
		So(b, ShouldBeNil)
		b = queue.DeleteBundle(from, 100)
		So(b, ShouldEqual, ErrBundle)

		bds := queue.TakeBundles(from)
		So(len(bds), ShouldEqual, 2)

		bundle := queue.DeleteBundle(from, 98)
		So(bundle, ShouldEqual, TimeOutBundle)
		bds = queue.TakeBundles(from)
		So(len(bds), ShouldEqual, 1)

		bundle = queue.DeleteBundle(from, 102)
		So(bundle, ShouldEqual, FeeCapTooLowBundle)
		bds = queue.TakeBundles(from)
		So(len(bds), ShouldEqual, 0)
	})
}
