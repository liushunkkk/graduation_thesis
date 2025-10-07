package bundlepool

import (
	ctx "context"
	"fmt"
	gomonkey "github.com/agiledragon/gomonkey/v2"
	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/common/ms"
	"github.com/ethereum/go-ethereum/core"
	"github.com/ethereum/go-ethereum/core/state"
	"github.com/ethereum/go-ethereum/core/types"
	. "github.com/ethereum/go-ethereum/log/zap"
	"github.com/ethereum/go-ethereum/push"
	"github.com/ethereum/go-ethereum/push/define"
	. "github.com/smartystreets/goconvey/convey"
	"go.uber.org/zap"
	"math/big"
	"reflect"
	"testing"
	"time"
)

var SimulateFailedBundle = &types.Bundle{MaxBlockNumber: 100}
var OkBundle = &types.Bundle{Txs: []*types.Transaction{Tx}, MaxBlockNumber: 100, Price: big.NewInt(1)}
var Tx = types.NewTx(&types.LegacyTx{Nonce: 100})

type Simulator struct {
}

func (s *Simulator) ExecuteBundle(parent *types.Header, bundle *types.Bundle, rpcBribeAddress common.Address) (*big.Int, *define.SseBundleData, error) {
	if bundle == SimulateFailedBundle {
		return nil, nil, fmt.Errorf("simulate failed")
	}
	if bundle.Parent != nil {
		if bundle.Parent.Parent != nil {
			return big.NewInt(0).Add(big.NewInt(2), big.NewInt(int64(len(bundle.Txs)))), &define.SseBundleData{}, nil
		}
		return big.NewInt(0).Add(big.NewInt(1), big.NewInt(int64(len(bundle.Txs)))), &define.SseBundleData{}, nil
	}
	return big.NewInt(0).Add(big.NewInt(0), big.NewInt(int64(len(bundle.Txs)))), &define.SseBundleData{}, nil
}

func TestNew(t *testing.T) {
	add := common.HexToAddress("0x38493489292")

	Zap.Info("jfkdajd", zap.Any("address", add))
}

func TestBundleGroup_Reset(t *testing.T) {
	blockchain := &core.BlockChain{}
	b := &types.Block{}
	patch1 := gomonkey.ApplyMethod(reflect.TypeOf(blockchain), "GetBlockByHash", func(_ *core.BlockChain, hash common.Hash) *types.Block {
		return b
	})
	defer patch1.Reset()

	Convey("transaction is exist", t, func() {
		tx := &types.Transaction{}
		patch2 := gomonkey.ApplyMethod(reflect.TypeOf(b), "Transaction", func(_ *types.Block, hash common.Hash) *types.Transaction {
			return tx
		})
		defer patch2.Reset()

		h := &types.Header{Number: big.NewInt(1)}
		bg := &BundleGroup{Original: &types.Bundle{Txs: []*types.Transaction{Tx}}, pool: &BundlePool{simulator: &Simulator{}}}
		closed, _ := bg.Reset(h)
		So(closed, ShouldEqual, true)
	})
	//
	Convey("transaction is not exist,but group is closed ", t, func() {
		patch2 := gomonkey.ApplyMethod(reflect.TypeOf(b), "Transaction", func(_ *types.Block, hash common.Hash) *types.Transaction {
			return nil
		})
		defer patch2.Reset()
		h := &types.Header{Number: big.NewInt(1)}
		bg := &BundleGroup{Original: &types.Bundle{Txs: []*types.Transaction{Tx}}, pool: &BundlePool{simulator: &Simulator{}}}
		bg.Closed = true
		closed, _ := bg.Reset(h)
		So(closed, ShouldEqual, true)
	})

	Convey("orginal bundle simulate ", t, func() {
		h := &types.Header{Number: big.NewInt(1)}
		bg := &BundleGroup{Original: OkBundle, pool: &BundlePool{simulator: &Simulator{}}}
		bg.bidServer, _ = ms.NewSvr("1", nil, nil)
		closed, _ := bg.Reset(h)
		So(closed, ShouldEqual, false)
	})

	Convey("orginal bundle simulate ", t, func() {
		h := &types.Header{Number: big.NewInt(1)}
		bg := &BundleGroup{Original: OkBundle, pool: &BundlePool{simulator: &Simulator{}}}
		bg.bidServer, _ = ms.NewSvr("1", nil, nil)
		closed, _ := bg.Reset(h)
		So(closed, ShouldEqual, false)
	})
}

func TestBundlePool_AddBundle(t *testing.T) {
	Convey("add bundle", t, func() {

		blockchain := &core.BlockChain{}
		now := time.Now()
		for {
			now = time.Now()
			if now.UnixNano()%1e9 < 1e7 {
				break
			}
			time.Sleep(5 * time.Millisecond)
		}
		old := &types.Header{Number: big.NewInt(100), Time: uint64(now.Unix())}
		patch := gomonkey.ApplyMethod(reflect.TypeOf(blockchain), "CurrentBlock", func(_ *core.BlockChain) *types.Header {
			return old
		})
		defer patch.Reset()

		statedb := &state.StateDB{}
		patch1 := gomonkey.ApplyMethod(reflect.TypeOf(blockchain), "State", func(_ *core.BlockChain) (*state.StateDB, error) {
			return statedb, nil
		})
		defer patch1.Reset()

		patch2 := gomonkey.ApplyMethod(reflect.TypeOf(statedb), "GetNonce", func(_ *state.StateDB, addr common.Address) uint64 {
			return uint64(1)
		})
		defer patch2.Reset()

		p := New(DefaultConfig, nil, blockchain)
		bundle1 := &types.Bundle{MaxBlockNumber: 100, Txs: []*types.Transaction{Tx}}
		p.simulator = &Simulator{}

		p.AddBundle(bundle1)

		time.Sleep(1600 * time.Millisecond)
		bundle2 := &types.Bundle{ParentHash: bundle1.Hash(), MaxBlockNumber: 100, Txs: []*types.Transaction{types.NewTx(&types.LegacyTx{Nonce: 101})}}
		p.AddBundle(bundle2)

		time.Sleep(1400 * time.Millisecond)
		p.PruneBundle(bundle1.Hash(), nil)
		p.Close()

	})
}

func TestBundlePool_GenBuilderReq(t *testing.T) {
	Convey("gen builder request body, single bundleGroup", t, func() {
		blockchain := &core.BlockChain{}
		now := time.Now()
		for {
			now = time.Now()
			if now.UnixNano()%1e9 < 1e7 {
				break
			}
			time.Sleep(5 * time.Millisecond)
		}
		old := &types.Header{Number: big.NewInt(100), Time: uint64(now.Unix()), BaseFee: big.NewInt(0)}
		patch := gomonkey.ApplyMethod(reflect.TypeOf(blockchain), "CurrentBlock", func(_ *core.BlockChain) *types.Header {
			return old
		})
		defer patch.Reset()

		statedb := &state.StateDB{}
		patch1 := gomonkey.ApplyMethod(reflect.TypeOf(blockchain), "State", func(_ *core.BlockChain) (*state.StateDB, error) {
			return statedb, nil
		})
		defer patch1.Reset()

		patch2 := gomonkey.ApplyMethod(reflect.TypeOf(statedb), "GetNonce", func(_ *state.StateDB, addr common.Address) uint64 {
			return uint64(1)
		})
		defer patch2.Reset()

		Convey("No searcher, No RPC, No builder, No user", func() {
			p := New(DefaultConfig, nil, blockchain)
			bundle := &types.Bundle{
				Counter:        0,
				RPCID:          "1826863360814092288",
				MaxBlockNumber: 100,
				Txs:            []*types.Transaction{Tx},
			}
			p.simulator = &Simulator{}
			p.AddBundle(bundle)
			time.Sleep(3 * time.Second)
			p.PruneBundle(bundle.Hash(), nil)
			p.Close()
		})

		Convey("No searcher, No RPC, No builder", func() {
			p := New(DefaultConfig, nil, blockchain)
			p.simulator = &Simulator{}
			t1 := types.NewTx(&types.LegacyTx{Nonce: 101})
			parentBundle := &types.Bundle{
				RPCID:          "1826863360814092288",
				Counter:        0,
				MaxBlockNumber: 100,
				Txs:            []*types.Transaction{t1},

				RefundPercent: 80,
				RefundAddress: common.HexToAddress("0xEAe25CD03AB852C3d5F9662D85d857a8801C161f"),
			}
			p.AddBundle(parentBundle)
			time.Sleep(200 * time.Millisecond)
			t2 := types.NewTx(&types.LegacyTx{Nonce: 102})
			bundle := &types.Bundle{
				RPCID:          "1826863360814092288",
				ParentHash:     parentBundle.Hash(),
				MaxBlockNumber: 100,
				Counter:        1,
				Txs:            []*types.Transaction{t2},
			}
			p.AddBundle(bundle)
			time.Sleep(3 * time.Second)
			p.PruneBundle(parentBundle.Hash(), nil)
			p.Close()
		})

		Convey("No searcher, No RPC", func() {
			p := New(DefaultConfig, nil, blockchain)
			p.simulator = &Simulator{}
			t1 := types.NewTx(&types.LegacyTx{Nonce: 104})
			parentBundle := &types.Bundle{
				Counter:        0,
				RPCID:          "1826863360814092288",
				MaxBlockNumber: 100,
				Txs:            []*types.Transaction{t1},

				RefundPercent: 80,
				RefundAddress: common.HexToAddress("0xEAe25CD03AB852C3d5F9662D85d857a8801C161f"),
			}
			p.AddBundle(parentBundle)
			time.Sleep(200 * time.Millisecond)
			t2 := types.NewTx(&types.LegacyTx{Nonce: 105})
			bundle := &types.Bundle{
				RPCID:          "1826863360814092288",
				Counter:        1,
				ParentHash:     parentBundle.Hash(),
				MaxBlockNumber: 100,
				Txs:            []*types.Transaction{t2},
			}
			p.AddBundle(bundle)
			time.Sleep(3 * time.Second)
			p.PruneBundle(parentBundle.Hash(), nil)
			p.Close()
		})

		Convey("No searcher", func() {
			p := New(DefaultConfig, nil, blockchain)
			p.simulator = &Simulator{}
			t1 := types.NewTx(&types.LegacyTx{Nonce: 104})
			parentBundle := &types.Bundle{
				RPCID:          "1826863360814092288",
				MaxBlockNumber: 100,
				Counter:        0,
				Txs:            []*types.Transaction{t1},

				RefundPercent: 80,
				RefundAddress: common.HexToAddress("0xEAe25CD03AB852C3d5F9662D85d857a8801C161f"),
			}
			p.AddBundle(parentBundle)
			time.Sleep(200 * time.Millisecond)
			t2 := types.NewTx(&types.LegacyTx{Nonce: 105})
			bundle := &types.Bundle{
				RPCID:          "1826863360814092288",
				Counter:        1,
				ParentHash:     parentBundle.Hash(),
				MaxBlockNumber: 100,
				Txs:            []*types.Transaction{t2},
			}
			p.AddBundle(bundle)
			time.Sleep(3 * time.Second)
			p.PruneBundle(parentBundle.Hash(), nil)
			p.Close()
		})

		Convey("All", func() {
			p := New(DefaultConfig, nil, blockchain)
			p.simulator = &Simulator{}
			t1 := types.NewTx(&types.LegacyTx{Nonce: 104})
			parentBundle := &types.Bundle{
				Counter:        0,
				RPCID:          "1826863360814092288",
				MaxBlockNumber: 100,
				Txs:            []*types.Transaction{t1},

				RefundPercent: 80,
				RefundAddress: common.HexToAddress("0xEAe25CD03AB852C3d5F9662D85d857a8801C161f"),
			}
			p.AddBundle(parentBundle)
			time.Sleep(200 * time.Millisecond)
			t2 := types.NewTx(&types.LegacyTx{Nonce: 105})
			parentBundle1 := &types.Bundle{
				Counter:        1,
				ParentHash:     parentBundle.Hash(),
				RPCID:          "1826863360814092288",
				MaxBlockNumber: 100,
				Txs:            []*types.Transaction{t2},

				RefundPercent: 80,
				RefundAddress: common.HexToAddress("0xEAe25CD03AB852C3d5F9662D85d857a8801C161f"),
			}
			p.AddBundle(parentBundle1)
			time.Sleep(200 * time.Millisecond)
			t3 := types.NewTx(&types.LegacyTx{Nonce: 106})
			bundle := &types.Bundle{
				Counter:        2,
				RPCID:          "1826863360814092288",
				ParentHash:     parentBundle1.Hash(),
				MaxBlockNumber: 100,
				Txs:            []*types.Transaction{t3},
			}
			p.AddBundle(bundle)
			time.Sleep(3 * time.Second)
			p.PruneBundle(parentBundle.Hash(), nil)
			p.Close()
		})
	})

}

func TestBundlePool_MultiGroup(t *testing.T) {

	Convey("gen builder request body, multi bundleGroup", t, func() {
		blockchain := &core.BlockChain{}
		now := time.Now()
		for {
			now = time.Now()
			if now.UnixNano()%1e9 < 1e7 {
				break
			}
			time.Sleep(5 * time.Millisecond)
		}
		old := &types.Header{Number: big.NewInt(100), Time: uint64(now.Unix()), BaseFee: big.NewInt(0)}
		patch := gomonkey.ApplyMethod(reflect.TypeOf(blockchain), "CurrentBlock", func(_ *core.BlockChain) *types.Header {
			return old
		})
		defer patch.Reset()

		statedb := &state.StateDB{}
		patch1 := gomonkey.ApplyMethod(reflect.TypeOf(blockchain), "State", func(_ *core.BlockChain) (*state.StateDB, error) {
			return statedb, nil
		})
		defer patch1.Reset()

		patch2 := gomonkey.ApplyMethod(reflect.TypeOf(statedb), "GetNonce", func(_ *state.StateDB, addr common.Address) uint64 {
			return uint64(1)
		})
		defer patch2.Reset()

		p := New(DefaultConfig, nil, blockchain)
		p.simulator = &Simulator{}

		bundle := &types.Bundle{
			RPCID:          "1826863360814092288",
			MaxBlockNumber: 100,
			Counter:        0,
			Txs:            []*types.Transaction{Tx},
		}
		p.AddBundle(bundle)
		time.Sleep(10 * time.Millisecond)

		Tx = types.NewTx(&types.LegacyTx{Nonce: 101})
		parentBundle := &types.Bundle{
			RPCID:          "1826863360814092288",
			MaxBlockNumber: 100,
			Counter:        0,
			Txs:            []*types.Transaction{Tx},

			RefundPercent: 80,
			RefundAddress: common.HexToAddress("0xEAe25CD03AB852C3d5F9662D85d857a8801C161f"),
		}
		p.AddBundle(parentBundle)
		time.Sleep(200 * time.Millisecond)
		Tx = types.NewTx(&types.LegacyTx{Nonce: 102})
		bundle = &types.Bundle{
			RPCID:          "1826863360814092288",
			ParentHash:     parentBundle.Hash(),
			Counter:        1,
			MaxBlockNumber: 100,
			Txs:            []*types.Transaction{Tx},
		}
		p.AddBundle(bundle)
		time.Sleep(10 * time.Millisecond)

		Tx = types.NewTx(&types.LegacyTx{Nonce: 103})
		parentBundle = &types.Bundle{
			RPCID:          "1826863360814092288",
			MaxBlockNumber: 100,
			Txs:            []*types.Transaction{Tx},

			Counter:       0,
			RefundPercent: 80,
			RefundAddress: common.HexToAddress("0xEAe25CD03AB852C3d5F9662D85d857a8801C161f"),
		}
		p.AddBundle(parentBundle)
		time.Sleep(200 * time.Millisecond)
		Tx = types.NewTx(&types.LegacyTx{Nonce: 104})
		bundle = &types.Bundle{
			RPCID:          "1826863360814092288",
			ParentHash:     parentBundle.Hash(),
			Counter:        1,
			MaxBlockNumber: 100,
			Txs:            []*types.Transaction{Tx},
		}
		p.AddBundle(bundle)
		time.Sleep(10 * time.Millisecond)

		Tx = types.NewTx(&types.LegacyTx{Nonce: 105})
		parentBundle = &types.Bundle{
			RPCID:          "1826863360814092288",
			MaxBlockNumber: 100,
			Counter:        0,
			Txs:            []*types.Transaction{Tx},

			RefundPercent: 80,
			RefundAddress: common.HexToAddress("0xEAe25CD03AB852C3d5F9662D85d857a8801C161f"),
		}
		p.AddBundle(parentBundle)
		time.Sleep(200 * time.Millisecond)
		Tx = types.NewTx(&types.LegacyTx{Nonce: 106})
		bundle = &types.Bundle{
			RPCID:          "1826863360814092288",
			ParentHash:     parentBundle.Hash(),
			Counter:        1,
			MaxBlockNumber: 100,
			Txs:            []*types.Transaction{Tx},
		}
		p.AddBundle(bundle)
		time.Sleep(10 * time.Millisecond)

		Tx = types.NewTx(&types.LegacyTx{Nonce: 107})
		parentBundle = &types.Bundle{
			RPCID:          "1826863360814092288",
			MaxBlockNumber: 100,
			Txs:            []*types.Transaction{Tx},
			Counter:        0,

			RefundPercent: 80,
			RefundAddress: common.HexToAddress("0xEAe25CD03AB852C3d5F9662D85d857a8801C161f"),
		}
		p.AddBundle(parentBundle)
		time.Sleep(200 * time.Millisecond)
		Tx = types.NewTx(&types.LegacyTx{Nonce: 108})
		parentBundle1 := &types.Bundle{
			RPCID:          "1826863360814092288",
			MaxBlockNumber: 100,
			Counter:        1,
			ParentHash:     parentBundle.Hash(),
			Txs:            []*types.Transaction{Tx},

			RefundPercent: 80,
			RefundAddress: common.HexToAddress("0xEAe25CD03AB852C3d5F9662D85d857a8801C161f"),
		}
		p.AddBundle(parentBundle1)
		time.Sleep(200 * time.Millisecond)
		Tx = types.NewTx(&types.LegacyTx{Nonce: 109})
		bundle = &types.Bundle{
			RPCID:          "1826863360814092288",
			ParentHash:     parentBundle1.Hash(),
			MaxBlockNumber: 100,
			Txs:            []*types.Transaction{Tx},
			Counter:        2,
		}
		p.AddBundle(bundle)

		time.Sleep(3 * time.Second)

		p.Close()
	})
}

func TestBundleGroup_Send(t *testing.T) {
	patches := gomonkey.ApplyMethod(reflect.TypeOf(&push.BuilderServer{}), "Send", func(p *push.BuilderServer, header *types.Header, bundle *types.Bundle, createNumber uint64) {
		fmt.Println(time.Now(), "send", header.Number)
		return
	})
	defer patches.Reset()

	Convey("==0,500", t, func() {
		now := time.Now()
		now_3 := time.Unix(now.Unix()/3*3+3, 0)
		time.Sleep(now_3.Sub(now))

		fmt.Println(time.Now())

		h := &types.Header{Number: big.NewInt(0), Time: uint64(now_3.Unix())}
		bg := &BundleGroup{
			Closed:        false,
			CreatedNumber: 0,
			Header:        h,
			Original: &types.Bundle{
				Txs:               nil,
				MaxBlockNumber:    0,
				MinTimestamp:      0,
				MaxTimestamp:      0,
				RevertingTxHashes: nil,
				Price:             nil,
				ParentHash:        common.Hash{},
				Parent:            nil,
				Counter:           0,
				Hint:              map[string]bool{"hash": true},
				RefundAddress:     common.Address{},
				RefundPercent:     0,
				From:              common.Address{},
				RPCID:             "",
				State:             0,
				PrivacyPeriod:     0,
				PrivacyBuilder:    nil,
				BroadcastBuilder:  nil,
				ArrivalTime:       time.Time{},
			},
			Bundles:       nil,
			bidServer:     nil,
			Slots:         0,
			pool:          nil,
			builderServer: &push.BuilderServer{},
			blockchain:    nil,
			sseServer:     nil,
		}

		time.Sleep(500 * time.Millisecond)

		bg.Send(ctx.Background(), h, 0)
	})

	Convey("==0,1300", t, func() {
		now := time.Now()
		now_3 := time.Unix(now.Unix()/3*3+3, 0)
		time.Sleep(now_3.Sub(now))

		fmt.Println(time.Now())

		h := &types.Header{Number: big.NewInt(0), Time: uint64(now_3.Unix())}
		bg := &BundleGroup{
			Closed:        false,
			CreatedNumber: 0,
			Header:        h,
			Original: &types.Bundle{
				Txs:               nil,
				MaxBlockNumber:    0,
				MinTimestamp:      0,
				MaxTimestamp:      0,
				RevertingTxHashes: nil,
				Price:             nil,
				ParentHash:        common.Hash{},
				Parent:            nil,
				Counter:           0,
				Hint:              map[string]bool{"hash": true},
				RefundAddress:     common.Address{},
				RefundPercent:     0,
				From:              common.Address{},
				RPCID:             "",
				State:             0,
				PrivacyPeriod:     0,
				PrivacyBuilder:    nil,
				BroadcastBuilder:  nil,
				ArrivalTime:       time.Time{},
			},
			Bundles:       nil,
			bidServer:     nil,
			Slots:         0,
			pool:          nil,
			builderServer: &push.BuilderServer{},
			blockchain:    nil,
			sseServer:     nil,
		}

		time.Sleep(1300 * time.Millisecond)

		bg.Send(ctx.Background(), h, 0)
	})

	Convey("==0,1800", t, func() {
		now := time.Now()
		now_3 := time.Unix(now.Unix()/3*3+3, 0)
		time.Sleep(now_3.Sub(now))

		fmt.Println(time.Now())

		h := &types.Header{Number: big.NewInt(0), Time: uint64(now_3.Unix())}
		bg := &BundleGroup{
			Closed:        false,
			CreatedNumber: 0,
			Header:        h,
			Original: &types.Bundle{
				Txs:               nil,
				MaxBlockNumber:    0,
				MinTimestamp:      0,
				MaxTimestamp:      0,
				RevertingTxHashes: nil,
				Price:             nil,
				ParentHash:        common.Hash{},
				Parent:            nil,
				Counter:           0,
				Hint:              map[string]bool{"hash": true},
				RefundAddress:     common.Address{},
				RefundPercent:     0,
				From:              common.Address{},
				RPCID:             "",
				State:             0,
				PrivacyPeriod:     0,
				PrivacyBuilder:    nil,
				BroadcastBuilder:  nil,
				ArrivalTime:       time.Time{},
			},
			Bundles:       nil,
			bidServer:     nil,
			Slots:         0,
			pool:          nil,
			builderServer: &push.BuilderServer{},
			blockchain:    nil,
			sseServer:     nil,
		}

		time.Sleep(1800 * time.Millisecond)

		bg.Send(ctx.Background(), h, 0)
	})

	Convey("==0,2500", t, func() {
		now := time.Now()
		now_3 := time.Unix(now.Unix()/3*3+3, 0)
		time.Sleep(now_3.Sub(now))

		fmt.Println(time.Now())

		h := &types.Header{Number: big.NewInt(0), Time: uint64(now_3.Unix())}
		bg := &BundleGroup{
			Closed:        false,
			CreatedNumber: 0,
			Header:        h,
			Original: &types.Bundle{
				Txs:               nil,
				MaxBlockNumber:    0,
				MinTimestamp:      0,
				MaxTimestamp:      0,
				RevertingTxHashes: nil,
				Price:             nil,
				ParentHash:        common.Hash{},
				Parent:            nil,
				Counter:           0,
				Hint:              map[string]bool{"hash": true},
				RefundAddress:     common.Address{},
				RefundPercent:     0,
				From:              common.Address{},
				RPCID:             "",
				State:             0,
				PrivacyPeriod:     0,
				PrivacyBuilder:    nil,
				BroadcastBuilder:  nil,
				ArrivalTime:       time.Time{},
			},
			Bundles:       nil,
			bidServer:     nil,
			Slots:         0,
			pool:          nil,
			builderServer: &push.BuilderServer{},
			blockchain:    nil,
			sseServer:     nil,
		}

		time.Sleep(2500 * time.Millisecond)

		bg.Send(ctx.Background(), h, 0)
	})
}

func TestBundleGroup_Send1(t *testing.T) {
	patches := gomonkey.ApplyMethod(reflect.TypeOf(&push.BuilderServer{}), "Send", func(p *push.BuilderServer, header *types.Header, bundle *types.Bundle, createNumber uint64) {
		fmt.Println(time.Now(), "send", header.Number)
		return
	})
	defer patches.Reset()

	Convey("==0,500", t, func() {
		now := time.Now()
		now_3 := time.Unix(now.Unix()/3*3+3, 0)
		time.Sleep(now_3.Sub(now))

		fmt.Println(time.Now())

		h := &types.Header{Number: big.NewInt(1), Time: uint64(now_3.Unix())}
		bg := &BundleGroup{
			Closed:        false,
			CreatedNumber: 0,
			Header:        h,
			Original: &types.Bundle{
				Txs:               nil,
				MaxBlockNumber:    0,
				MinTimestamp:      0,
				MaxTimestamp:      0,
				RevertingTxHashes: nil,
				Price:             nil,
				ParentHash:        common.Hash{},
				Parent:            nil,
				Counter:           0,
				Hint:              map[string]bool{"hash": true},
				RefundAddress:     common.Address{},
				RefundPercent:     0,
				From:              common.Address{},
				RPCID:             "",
				State:             0,
				PrivacyPeriod:     0,
				PrivacyBuilder:    nil,
				BroadcastBuilder:  nil,
				ArrivalTime:       time.Time{},
			},
			Bundles:       nil,
			bidServer:     nil,
			Slots:         0,
			pool:          nil,
			builderServer: &push.BuilderServer{},
			blockchain:    nil,
			sseServer:     nil,
		}

		time.Sleep(500 * time.Millisecond)

		bg.Send(ctx.Background(), h, 0)
	})

	Convey("==0,1300", t, func() {
		now := time.Now()
		now_3 := time.Unix(now.Unix()/3*3+3, 0)
		time.Sleep(now_3.Sub(now))

		fmt.Println(time.Now())

		h := &types.Header{Number: big.NewInt(1), Time: uint64(now_3.Unix())}
		bg := &BundleGroup{
			Closed:        false,
			CreatedNumber: 0,
			Header:        h,
			Original: &types.Bundle{
				Txs:               nil,
				MaxBlockNumber:    0,
				MinTimestamp:      0,
				MaxTimestamp:      0,
				RevertingTxHashes: nil,
				Price:             nil,
				ParentHash:        common.Hash{},
				Parent:            nil,
				Counter:           0,
				Hint:              map[string]bool{"hash": true},
				RefundAddress:     common.Address{},
				RefundPercent:     0,
				From:              common.Address{},
				RPCID:             "",
				State:             0,
				PrivacyPeriod:     0,
				PrivacyBuilder:    nil,
				BroadcastBuilder:  nil,
				ArrivalTime:       time.Time{},
			},
			Bundles:       nil,
			bidServer:     nil,
			Slots:         0,
			pool:          nil,
			builderServer: &push.BuilderServer{},
			blockchain:    nil,
			sseServer:     nil,
		}

		time.Sleep(1300 * time.Millisecond)

		bg.Send(ctx.Background(), h, 0)
	})

	Convey("==0,1800", t, func() {
		now := time.Now()
		now_3 := time.Unix(now.Unix()/3*3+3, 0)
		time.Sleep(now_3.Sub(now))

		fmt.Println(time.Now())

		h := &types.Header{Number: big.NewInt(1), Time: uint64(now_3.Unix())}
		bg := &BundleGroup{
			Closed:        false,
			CreatedNumber: 0,
			Header:        h,
			Original: &types.Bundle{
				Txs:               nil,
				MaxBlockNumber:    0,
				MinTimestamp:      0,
				MaxTimestamp:      0,
				RevertingTxHashes: nil,
				Price:             nil,
				ParentHash:        common.Hash{},
				Parent:            nil,
				Counter:           0,
				Hint:              map[string]bool{"hash": true},
				RefundAddress:     common.Address{},
				RefundPercent:     0,
				From:              common.Address{},
				RPCID:             "",
				State:             0,
				PrivacyPeriod:     0,
				PrivacyBuilder:    nil,
				BroadcastBuilder:  nil,
				ArrivalTime:       time.Time{},
			},
			Bundles:       nil,
			bidServer:     nil,
			Slots:         0,
			pool:          nil,
			builderServer: &push.BuilderServer{},
			blockchain:    nil,
			sseServer:     nil,
		}

		time.Sleep(1800 * time.Millisecond)

		bg.Send(ctx.Background(), h, 0)
	})

	Convey("==0,2500", t, func() {
		now := time.Now()
		now_3 := time.Unix(now.Unix()/3*3+3, 0)
		time.Sleep(now_3.Sub(now))

		fmt.Println(time.Now())

		h := &types.Header{Number: big.NewInt(1), Time: uint64(now_3.Unix())}
		bg := &BundleGroup{
			Closed:        false,
			CreatedNumber: 0,
			Header:        h,
			Original: &types.Bundle{
				Txs:               nil,
				MaxBlockNumber:    0,
				MinTimestamp:      0,
				MaxTimestamp:      0,
				RevertingTxHashes: nil,
				Price:             nil,
				ParentHash:        common.Hash{},
				Parent:            nil,
				Counter:           0,
				Hint:              map[string]bool{"hash": true},
				RefundAddress:     common.Address{},
				RefundPercent:     0,
				From:              common.Address{},
				RPCID:             "",
				State:             0,
				PrivacyPeriod:     0,
				PrivacyBuilder:    nil,
				BroadcastBuilder:  nil,
				ArrivalTime:       time.Time{},
			},
			Bundles:       nil,
			bidServer:     nil,
			Slots:         0,
			pool:          nil,
			builderServer: &push.BuilderServer{},
			blockchain:    nil,
			sseServer:     nil,
		}

		time.Sleep(2500 * time.Millisecond)

		bg.Send(ctx.Background(), h, 0)
	})
}

func TestBundleGroup_Send2(t *testing.T) {
	patches := gomonkey.ApplyMethod(reflect.TypeOf(&push.BuilderServer{}), "Send", func(p *push.BuilderServer, header *types.Header, bundle *types.Bundle, createNumber uint64) {
		fmt.Println(time.Now(), "send", header.Number)
		return
	})
	defer patches.Reset()

	Convey("==0,500", t, func() {
		now := time.Now()
		now_3 := time.Unix(now.Unix()/3*3+3, 0)
		time.Sleep(now_3.Sub(now))

		fmt.Println(time.Now())

		h := &types.Header{Number: big.NewInt(2), Time: uint64(now_3.Unix())}
		bg := &BundleGroup{
			Closed:        false,
			CreatedNumber: 0,
			Header:        h,
			Original: &types.Bundle{
				Txs:               nil,
				MaxBlockNumber:    0,
				MinTimestamp:      0,
				MaxTimestamp:      0,
				RevertingTxHashes: nil,
				Price:             nil,
				ParentHash:        common.Hash{},
				Parent:            nil,
				Counter:           0,
				Hint:              map[string]bool{"hash": true},
				RefundAddress:     common.Address{},
				RefundPercent:     0,
				From:              common.Address{},
				RPCID:             "",
				State:             0,
				PrivacyPeriod:     0,
				PrivacyBuilder:    nil,
				BroadcastBuilder:  nil,
				ArrivalTime:       time.Time{},
			},
			Bundles:       nil,
			bidServer:     nil,
			Slots:         0,
			pool:          nil,
			builderServer: &push.BuilderServer{},
			blockchain:    nil,
			sseServer:     nil,
		}

		time.Sleep(500 * time.Millisecond)

		bg.Send(ctx.Background(), h, 0)
	})

	Convey("==0,1300", t, func() {
		now := time.Now()
		now_3 := time.Unix(now.Unix()/3*3+3, 0)
		time.Sleep(now_3.Sub(now))

		fmt.Println(time.Now())

		h := &types.Header{Number: big.NewInt(2), Time: uint64(now_3.Unix())}
		bg := &BundleGroup{
			Closed:        false,
			CreatedNumber: 0,
			Header:        h,
			Original: &types.Bundle{
				Txs:               nil,
				MaxBlockNumber:    0,
				MinTimestamp:      0,
				MaxTimestamp:      0,
				RevertingTxHashes: nil,
				Price:             nil,
				ParentHash:        common.Hash{},
				Parent:            nil,
				Counter:           0,
				Hint:              map[string]bool{"hash": true},
				RefundAddress:     common.Address{},
				RefundPercent:     0,
				From:              common.Address{},
				RPCID:             "",
				State:             0,
				PrivacyPeriod:     0,
				PrivacyBuilder:    nil,
				BroadcastBuilder:  nil,
				ArrivalTime:       time.Time{},
			},
			Bundles:       nil,
			bidServer:     nil,
			Slots:         0,
			pool:          nil,
			builderServer: &push.BuilderServer{},
			blockchain:    nil,
			sseServer:     nil,
		}

		time.Sleep(1300 * time.Millisecond)

		bg.Send(ctx.Background(), h, 0)
	})

	Convey("==0,1800", t, func() {
		now := time.Now()
		now_3 := time.Unix(now.Unix()/3*3+3, 0)
		time.Sleep(now_3.Sub(now))

		fmt.Println(time.Now())

		h := &types.Header{Number: big.NewInt(2), Time: uint64(now_3.Unix())}
		bg := &BundleGroup{
			Closed:        false,
			CreatedNumber: 0,
			Header:        h,
			Original: &types.Bundle{
				Txs:               nil,
				MaxBlockNumber:    0,
				MinTimestamp:      0,
				MaxTimestamp:      0,
				RevertingTxHashes: nil,
				Price:             nil,
				ParentHash:        common.Hash{},
				Parent:            nil,
				Counter:           0,
				Hint:              map[string]bool{"hash": true},
				RefundAddress:     common.Address{},
				RefundPercent:     0,
				From:              common.Address{},
				RPCID:             "",
				State:             0,
				PrivacyPeriod:     0,
				PrivacyBuilder:    nil,
				BroadcastBuilder:  nil,
				ArrivalTime:       time.Time{},
			},
			Bundles:       nil,
			bidServer:     nil,
			Slots:         0,
			pool:          nil,
			builderServer: &push.BuilderServer{},
			blockchain:    nil,
			sseServer:     nil,
		}

		time.Sleep(1800 * time.Millisecond)

		bg.Send(ctx.Background(), h, 0)
	})

	Convey("==0,2500", t, func() {
		now := time.Now()
		now_3 := time.Unix(now.Unix()/3*3+3, 0)
		time.Sleep(now_3.Sub(now))

		fmt.Println(time.Now())

		h := &types.Header{Number: big.NewInt(2), Time: uint64(now_3.Unix())}
		bg := &BundleGroup{
			Closed:        false,
			CreatedNumber: 0,
			Header:        h,
			Original: &types.Bundle{
				Txs:               nil,
				MaxBlockNumber:    0,
				MinTimestamp:      0,
				MaxTimestamp:      0,
				RevertingTxHashes: nil,
				Price:             nil,
				ParentHash:        common.Hash{},
				Parent:            nil,
				Counter:           0,
				Hint:              map[string]bool{"hash": true},
				RefundAddress:     common.Address{},
				RefundPercent:     0,
				From:              common.Address{},
				RPCID:             "",
				State:             0,
				PrivacyPeriod:     0,
				PrivacyBuilder:    nil,
				BroadcastBuilder:  nil,
				ArrivalTime:       time.Time{},
			},
			Bundles:       nil,
			bidServer:     nil,
			Slots:         0,
			pool:          nil,
			builderServer: &push.BuilderServer{},
			blockchain:    nil,
			sseServer:     nil,
		}

		time.Sleep(2500 * time.Millisecond)

		bg.Send(ctx.Background(), h, 0)
	})
}
