package miner

import (
	"errors"
	"github.com/agiledragon/gomonkey/v2"
	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/common/hexutil"
	"github.com/ethereum/go-ethereum/consensus/clique"
	"github.com/ethereum/go-ethereum/core"
	"github.com/ethereum/go-ethereum/core/rawdb"
	"github.com/ethereum/go-ethereum/core/state"
	"github.com/ethereum/go-ethereum/core/types"
	"github.com/ethereum/go-ethereum/core/vm"
	"github.com/ethereum/go-ethereum/params"
	"github.com/holiman/uint256"
	. "github.com/smartystreets/goconvey/convey"
	"math/big"
	"reflect"
	"testing"
)

var hint = map[string]bool{
	types.HintLogs:             true,
	types.HintGasPrice:         true,
	types.HintHash:             true,
	types.HintFunctionSelector: true,
	types.HintCallData:         true,
	types.HintFrom:             true,
	types.HintTo:               true,
	types.HintGasLimit:         true,
	types.HintValue:            true,
	types.HintNonce:            true,
}

func TestWorker_SimulateBundle(t *testing.T) {
	engine := clique.New(cliqueChainConfig.Clique, rawdb.NewMemoryDatabase())
	defer engine.Close()

	w, _ := newTestWorker(t, cliqueChainConfig, engine, rawdb.NewMemoryDatabase(), 0)

	w.rpcSimulator = types.NewRpcSimulator()

	header := &types.Header{Number: big.NewInt(1), Time: 1234, BaseFee: big.NewInt(1_000_000_000)}
	stateDB := &state.StateDB{}

	env := &environment{
		header: header,
		state:  stateDB,
		signer: types.MakeSigner(w.chainConfig, header.Number, header.Time),
	}

	gasPool := new(core.GasPool).AddGas(21_000)
	signer := types.LatestSigner(params.TestChainConfig)

	Convey("no parent, no bribe, origin raw tx", t, func() {
		patchGetBalance := gomonkey.ApplyMethod(reflect.TypeOf(stateDB), "GetBalance", func(_ *state.StateDB, addr common.Address) *uint256.Int {
			return new(uint256.Int).SetUint64(1_000_000_000)
		})
		defer patchGetBalance.Reset()
		var patchApplyTransaction = gomonkey.ApplyFunc(core.ApplyTransaction, func(config *params.ChainConfig, bc core.ChainContext, author *common.Address, gp *core.GasPool, statedb *state.StateDB, header *types.Header, tx *types.Transaction, usedGas *uint64, cfg vm.Config, receiptProcessors ...core.ReceiptProcessor) (*types.Receipt, error) {
			data1, _ := hexutil.Decode("0x1")
			data2, _ := hexutil.Decode("0x2")
			return &types.Receipt{
				Status:  types.ReceiptStatusSuccessful,
				GasUsed: 21000,
				Logs: []*types.Log{
					{
						Address: common.HexToAddress("0xfffffffffffffffffffffffffffffffffffffffff"),
						Topics:  []common.Hash{common.BigToHash(big.NewInt(1)), common.BigToHash(big.NewInt(2))},
						Data:    data1,
					},
					{
						Address: common.HexToAddress("0xeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"),
						Topics:  []common.Hash{common.BigToHash(big.NewInt(3)), common.BigToHash(big.NewInt(4))},
						Data:    data2,
					},
				},
			}, nil
		})
		defer patchApplyTransaction.Reset()
		callData, _ := hexutil.Decode("0xf340fa01000000000000000000000000b4647b856cb9c3856d559c885bed8b43e0846a47")
		bundle := &types.Bundle{
			Parent:     nil,
			ParentHash: common.Hash{},
			Counter:    0,
			Txs: []*types.Transaction{types.MustSignNewTx(testBankKey, signer, &types.LegacyTx{
				Nonce:    1,
				To:       &testUserAddress,
				Value:    big.NewInt(100_000),
				Gas:      params.TxGas,
				GasPrice: big.NewInt(1_200_000_000),
				Data:     callData,
			})},
			MaxBlockNumber: 100,
			Hint:           hint,
		}
		var tempGasUsed uint64
		var txCount int = 1
		simBundle, bundleData, err := w.simulateBundle(env, bundle, stateDB, gasPool, &txCount, false, false, &tempGasUsed, common.Address{})
		So(err, ShouldBeNil)
		So(simBundle, ShouldNotBeNil)
		So(bundleData, ShouldNotBeNil)
		So(simBundle.RpcBundlePrice, ShouldEqual, new(big.Int).SetUint64(4_200_000_000_000))
	})

	Convey("has parent, max counter = 1", t, func() {

		var patchApplyTransaction = gomonkey.ApplyFunc(core.ApplyTransaction, func(config *params.ChainConfig, bc core.ChainContext, author *common.Address, gp *core.GasPool, statedb *state.StateDB, header *types.Header, tx *types.Transaction, usedGas *uint64, cfg vm.Config, receiptProcessors ...core.ReceiptProcessor) (*types.Receipt, error) {
			data1, _ := hexutil.Decode("0x1")
			data2, _ := hexutil.Decode("0x2")
			return &types.Receipt{
				Status:  types.ReceiptStatusSuccessful,
				GasUsed: 21000,
				Logs: []*types.Log{
					{
						Address: common.HexToAddress("0xfffffffffffffffffffffffffffffffffffffffff"),
						Topics:  []common.Hash{common.BigToHash(big.NewInt(1)), common.BigToHash(big.NewInt(2))},
						Data:    data1,
					},
					{
						Address: common.HexToAddress("0xeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"),
						Topics:  []common.Hash{common.BigToHash(big.NewInt(3)), common.BigToHash(big.NewInt(4))},
						Data:    data2,
					},
				},
			}, nil
		})
		defer patchApplyTransaction.Reset()
		parentBundle := &types.Bundle{
			Parent:     nil,
			ParentHash: common.Hash{},
			Counter:    0,
			Txs: []*types.Transaction{types.MustSignNewTx(testBankKey, signer, &types.LegacyTx{
				Nonce:    1,
				To:       &testUserAddress,
				Value:    big.NewInt(100_000),
				Gas:      params.TxGas,
				GasPrice: big.NewInt(1_200_000_000),
				Data:     nil,
			})},
			MaxBlockNumber: 100,
			Hint:           hint,
		}
		callData, _ := hexutil.Decode("0xf340fa01000000000000000000000000b4647b856cb9c3856d559c885bed8b43e0846a47")
		bundle := &types.Bundle{
			Parent:     parentBundle,
			ParentHash: parentBundle.Hash(),
			Counter:    1,
			Txs: []*types.Transaction{types.MustSignNewTx(testBankKey, signer, &types.LegacyTx{
				Nonce:    2,
				To:       &testUserAddress,
				Value:    big.NewInt(100_000),
				Gas:      params.TxGas,
				GasPrice: big.NewInt(1_200_000_000),
				Data:     callData,
			})},
			MaxBlockNumber: 100,
			Hint:           hint,
			RefundPercent:  90,
			RefundAddress:  common.HexToAddress("0xcccccccccccccccccccccccccccccccc"),
		}

		Convey("no bribe, the result should be error", func() {
			patchGetBalance := gomonkey.ApplyMethod(reflect.TypeOf(stateDB), "GetBalance", func(_ *state.StateDB, addr common.Address) *uint256.Int {
				return new(uint256.Int).SetUint64(1_000_000_000) // 1 gwei
			})
			defer patchGetBalance.Reset()
			var txCount int = 1
			var tempGasUsed uint64
			simBundle, bundleData, err := w.simulateBundle(env, bundle, stateDB, gasPool, &txCount, false, false, &tempGasUsed, common.Address{})
			So(err, ShouldNotBeNil)
			So(simBundle, ShouldBeNil)
			So(bundleData, ShouldBeNil)
		})

		Convey("has bribe, but bribe is not enough to transfer", func() {
			c := 1
			patchGetBalance := gomonkey.ApplyMethod(reflect.TypeOf(stateDB), "GetBalance", func(_ *state.StateDB, addr common.Address) *uint256.Int {
				if c < 4 {
					c++
					return new(uint256.Int).SetUint64(1_000_000_000) // 1 gwei
				}
				return new(uint256.Int).SetUint64(21_000_000_000) // 21 gwei
			})
			defer patchGetBalance.Reset()

			var tempGasUsed uint64
			var txCount = 1
			simBundle, bundleData, err := w.simulateBundle(env, bundle, stateDB, gasPool, &txCount, false, false, &tempGasUsed, common.Address{})
			So(err, ShouldBeNil)
			So(simBundle, ShouldNotBeNil)
			So(bundleData, ShouldNotBeNil)
			// bundle price
			// 4_200_000_000_000
			txFees := new(big.Int).Mul(new(big.Int).Sub(big.NewInt(1_200_000_000), big.NewInt(1_000_000_000)), big.NewInt(21_000*2))
			_, _ = w.rpcSimulator.GetBribeToBuilderAndSender(big.NewInt(20_000_000_000), bundle.RefundPercent)

			So(w.rpcSimulator.GetSingleTxFee(env.header.BaseFee), ShouldEqual, big.NewInt(21_000_000_000_000))
			profitToBuilder := new(big.Int)
			So(simBundle.RpcBundlePrice, ShouldEqual, big.NewInt(0).Add(txFees, profitToBuilder))
		})

		Convey("has bribe, and bribe is enough to transfer", func() {
			c := 1
			patchGetBalance := gomonkey.ApplyMethod(reflect.TypeOf(stateDB), "GetBalance", func(_ *state.StateDB, addr common.Address) *uint256.Int {
				if c < 4 {
					c++
					return new(uint256.Int).SetUint64(1_000_000_000_000_000) // 1 gwei
				}
				return new(uint256.Int).SetUint64(21_000_000_000_000_000) // 21 gwei
			})
			defer patchGetBalance.Reset()

			Convey("no singleTxTip", func() {
				var tempGasUsed uint64
				var txCount = 1
				simBundle, bundleData, err := w.simulateBundle(env, bundle, stateDB, gasPool, &txCount, false, false, &tempGasUsed, common.Address{})
				So(err, ShouldBeNil)
				So(simBundle, ShouldNotBeNil)
				So(bundleData, ShouldNotBeNil)
				// bundle price
				// 4_200_000_000_000
				txFees := new(big.Int).Mul(new(big.Int).Sub(big.NewInt(1_200_000_000), big.NewInt(1_000_000_000)), big.NewInt(21_000*2))
				bribeToBuilder, bribeToUser := w.rpcSimulator.GetBribeToBuilderAndSender(big.NewInt(20_000_000_000_000_000), bundle.RefundPercent)

				So(w.rpcSimulator.GetSingleTxFee(env.header.BaseFee), ShouldEqual, big.NewInt(21_000_000_000_000))
				So(w.rpcSimulator.GetSingleTxTip(env.header.BaseFee), ShouldEqual, new(big.Int))
				profitToBuilder := new(big.Int).Sub(bribeToBuilder, w.rpcSimulator.GetSingleTxFee(env.header.BaseFee))
				b := common.PercentOf(profitToBuilder, types.RpcBuilderProfitPercent)
				So(simBundle.RpcBundlePrice, ShouldEqual, big.NewInt(0).Add(txFees, b))

				_ = new(big.Int).Sub(bribeToUser, w.rpcSimulator.GetSingleTxFee(env.header.BaseFee))
			})

			Convey("has singleTxTip", func() {

				types.RpcGasPrice = big.NewInt(1_100_000_000)
				var tempGasUsed uint64
				var txCount = 1
				simBundle, bundleData, err := w.simulateBundle(env, bundle, stateDB, gasPool, &txCount, false, false, &tempGasUsed, common.Address{})
				So(err, ShouldBeNil)
				So(simBundle, ShouldNotBeNil)
				So(bundleData, ShouldNotBeNil)
				// bundle price
				// 4_200_000_000_000
				txFees := new(big.Int).Mul(new(big.Int).Sub(big.NewInt(1_200_000_000), big.NewInt(1_000_000_000)), big.NewInt(21_000*2))
				bribeToBuilder, bribeToUser := w.rpcSimulator.GetBribeToBuilderAndSender(big.NewInt(20_000_000_000_000_000), bundle.RefundPercent)

				So(w.rpcSimulator.GetSingleTxFee(env.header.BaseFee), ShouldEqual, big.NewInt(23_100_000_000_000))
				So(w.rpcSimulator.GetSingleTxTip(env.header.BaseFee), ShouldEqual, big.NewInt(2_100_000_000_000))

				profitToBuilder := new(big.Int).Sub(bribeToBuilder, w.rpcSimulator.GetSingleTxFee(env.header.BaseFee))
				b := common.PercentOf(profitToBuilder, types.RpcBuilderProfitPercent)
				b.Add(b, big.NewInt(2_100_000_000_000))
				b.Add(b, big.NewInt(2_100_000_000_000))
				So(simBundle.RpcBundlePrice, ShouldEqual, big.NewInt(0).Add(txFees, b))

				_ = new(big.Int).Sub(bribeToUser, w.rpcSimulator.GetSingleTxFee(env.header.BaseFee))
			})
		})

	})

	Convey("has parent, has bribe, maxCount = 1, and bribe is enough to transfer, and has singleTxTip, but a tx execute fail and has no reverting", t, func() {
		c := 1
		patchGetBalance := gomonkey.ApplyMethod(reflect.TypeOf(stateDB), "GetBalance", func(_ *state.StateDB, addr common.Address) *uint256.Int {
			if c < 4 {
				c++
				return new(uint256.Int).SetUint64(1_000_000_000_000_000) // 1 gwei
			}
			return new(uint256.Int).SetUint64(21_000_000_000_000_000) // 21 gwei
		})
		defer patchGetBalance.Reset()
		var patchApplyTransaction = gomonkey.ApplyFunc(core.ApplyTransaction, func(config *params.ChainConfig, bc core.ChainContext, author *common.Address, gp *core.GasPool, statedb *state.StateDB, header *types.Header, tx *types.Transaction, usedGas *uint64, cfg vm.Config, receiptProcessors ...core.ReceiptProcessor) (*types.Receipt, error) {
			data1, _ := hexutil.Decode("0x1")
			data2, _ := hexutil.Decode("0x2")
			return &types.Receipt{
				Status:  types.ReceiptStatusFailed,
				GasUsed: 21000,
				Logs: []*types.Log{
					{
						Address: common.HexToAddress("0xfffffffffffffffffffffffffffffffffffffffff"),
						Topics:  []common.Hash{common.BigToHash(big.NewInt(1)), common.BigToHash(big.NewInt(2))},
						Data:    data1,
					},
					{
						Address: common.HexToAddress("0xeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"),
						Topics:  []common.Hash{common.BigToHash(big.NewInt(3)), common.BigToHash(big.NewInt(4))},
						Data:    data2,
					},
				},
			}, nil
		})
		defer patchApplyTransaction.Reset()
		parentBundle := &types.Bundle{
			Parent:     nil,
			ParentHash: common.Hash{},
			Counter:    0,
			Txs: []*types.Transaction{types.MustSignNewTx(testBankKey, signer, &types.LegacyTx{
				Nonce:    1,
				To:       &testUserAddress,
				Value:    big.NewInt(100_000),
				Gas:      params.TxGas,
				GasPrice: big.NewInt(1_200_000_000),
				Data:     nil,
			})},
			MaxBlockNumber: 100,
			Hint:           hint,
		}

		callData, _ := hexutil.Decode("0xf340fa01000000000000000000000000b4647b856cb9c3856d559c885bed8b43e0846a47")
		bundle := &types.Bundle{
			Parent:     parentBundle,
			ParentHash: parentBundle.Hash(),
			Counter:    1,
			Txs: []*types.Transaction{types.MustSignNewTx(testBankKey, signer, &types.LegacyTx{
				Nonce:    2,
				To:       &testUserAddress,
				Value:    big.NewInt(100_000),
				Gas:      params.TxGas,
				GasPrice: big.NewInt(1_200_000_000),
				Data:     callData,
			})},
			MaxBlockNumber: 100,
			Hint:           hint,
			RefundPercent:  90,
			RefundAddress:  common.HexToAddress("0xcccccccccccccccccccccccccccccccc"),
		}
		var tempGasUsed uint64
		var txCount = 1
		simBundle, bundleData, err := w.simulateBundle(env, bundle, stateDB, gasPool, &txCount, false, false, &tempGasUsed, common.Address{})
		So(err, ShouldEqual, errNonRevertingTxInBundleFailed)
		So(simBundle, ShouldBeNil)
		So(bundleData, ShouldBeNil)
	})

	Convey("has parent, has bribe, maxCount = 1, and bribe is enough to transfer, and has singleTxTip, simulate failed but has reverting", t, func() {
		c := 1
		patchGetBalance := gomonkey.ApplyMethodFunc(stateDB, "GetBalance", func(addr common.Address) *uint256.Int {
			if c < 4 {
				c++
				return new(uint256.Int).SetUint64(1_000_000_000_000_000) // 1 gwei
			}
			return new(uint256.Int).SetUint64(21_000_000_000_000_000) // 21 gwei
		})
		//patchGetBalance := gomonkey.ApplyMethod(reflect.TypeOf(stateDB), "GetBalance", func(_ *state.StateDB, addr common.Address) *uint256.Int {
		//	if c < 7 {
		//		c++
		//		return new(uint256.Int).SetUint64(1_000_000_000_000_000) // 1 gwei
		//	}
		//	return new(uint256.Int).SetUint64(21_000_000_000_000_000) // 21 gwei
		//})
		defer patchGetBalance.Reset()
		receiptStatus := types.ReceiptStatusFailed
		var patchApplyTransaction = gomonkey.ApplyFunc(core.ApplyTransaction, func(config *params.ChainConfig, bc core.ChainContext, author *common.Address, gp *core.GasPool, statedb *state.StateDB, header *types.Header, tx *types.Transaction, usedGas *uint64, cfg vm.Config, receiptProcessors ...core.ReceiptProcessor) (*types.Receipt, error) {
			data1, _ := hexutil.Decode("0x1")
			data2, _ := hexutil.Decode("0x2")
			// 第一个成功，第二个失败
			if receiptStatus == types.ReceiptStatusSuccessful {
				receiptStatus = types.ReceiptStatusFailed
			} else {
				receiptStatus = types.ReceiptStatusSuccessful
			}
			return &types.Receipt{
				TxHash:  common.HexToHash("0x1234"),
				Status:  receiptStatus,
				GasUsed: 21000,
				Logs: []*types.Log{
					{
						Address: common.HexToAddress("0xfffffffffffffffffffffffffffffffffffffffff"),
						Topics:  []common.Hash{common.BigToHash(big.NewInt(1)), common.BigToHash(big.NewInt(2))},
						Data:    data1,
					},
					{
						Address: common.HexToAddress("0xeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"),
						Topics:  []common.Hash{common.BigToHash(big.NewInt(3)), common.BigToHash(big.NewInt(4))},
						Data:    data2,
					},
				},
			}, nil
		})
		defer patchApplyTransaction.Reset()
		parentBundle := &types.Bundle{
			Parent:     nil,
			ParentHash: common.Hash{},
			Counter:    0,
			Txs: []*types.Transaction{types.MustSignNewTx(testBankKey, signer, &types.LegacyTx{
				Nonce:    1,
				To:       &testUserAddress,
				Value:    big.NewInt(100_000),
				Gas:      params.TxGas,
				GasPrice: big.NewInt(1_200_000_000),
				Data:     nil,
			})},
			MaxBlockNumber: 100,
			Hint:           hint,
		}

		callData, _ := hexutil.Decode("0xf340fa01000000000000000000000000b4647b856cb9c3856d559c885bed8b43e0846a47")
		bundle := &types.Bundle{
			Parent:     parentBundle,
			ParentHash: parentBundle.Hash(),
			Counter:    1,
			Txs: []*types.Transaction{types.MustSignNewTx(testBankKey, signer, &types.LegacyTx{
				Nonce:    2,
				To:       &testUserAddress,
				Value:    big.NewInt(100_000),
				Gas:      params.TxGas,
				GasPrice: big.NewInt(1_200_000_000),
				Data:     callData,
			})},
			MaxBlockNumber:    100,
			Hint:              hint,
			RefundPercent:     90,
			RefundAddress:     common.HexToAddress("0xcccccccccccccccccccccccccccccccc"),
			RevertingTxHashes: []common.Hash{common.HexToHash("0x1234")},
		}
		types.RpcGasPrice = big.NewInt(1_100_000_000)
		var tempGasUsed uint64
		var txCount = 1
		simBundle, bundleData, err := w.simulateBundle(env, bundle, stateDB, gasPool, &txCount, false, false, &tempGasUsed, common.Address{})
		So(err, ShouldBeNil)
		So(simBundle, ShouldNotBeNil)
		So(bundleData, ShouldNotBeNil)
		// bundle price
		// 4_200_000_000_000
		txFees := new(big.Int).Mul(new(big.Int).Sub(big.NewInt(1_200_000_000), big.NewInt(1_000_000_000)), big.NewInt(21_000*2))
		bribeToBuilder, bribeToUser := w.rpcSimulator.GetBribeToBuilderAndSender(big.NewInt(20_000_000_000_000_000), bundle.RefundPercent)

		So(w.rpcSimulator.GetSingleTxFee(env.header.BaseFee), ShouldEqual, big.NewInt(23_100_000_000_000))
		So(w.rpcSimulator.GetSingleTxTip(env.header.BaseFee), ShouldEqual, big.NewInt(2_100_000_000_000))

		profitToBuilder := new(big.Int).Sub(bribeToBuilder, w.rpcSimulator.GetSingleTxFee(env.header.BaseFee))
		b := common.PercentOf(profitToBuilder, types.RpcBuilderProfitPercent)
		b.Add(b, big.NewInt(2_100_000_000_000))
		b.Add(b, big.NewInt(2_100_000_000_000))
		So(simBundle.RpcBundlePrice, ShouldEqual, big.NewInt(0).Add(txFees, b))

		_ = new(big.Int).Sub(bribeToUser, w.rpcSimulator.GetSingleTxFee(env.header.BaseFee))
	})

	Convey("has parent, has bribe, maxCount = 1, apply transaction error", t, func() {
		c := 1
		patchGetBalance := gomonkey.ApplyMethod(reflect.TypeOf(stateDB), "GetBalance", func(_ *state.StateDB, addr common.Address) *uint256.Int {
			if c < 4 {
				c++
				return new(uint256.Int).SetUint64(1_000_000_000_000_000) // 1 gwei
			}
			return new(uint256.Int).SetUint64(21_000_000_000_000_000) // 21 gwei
		})
		defer patchGetBalance.Reset()
		var patchApplyTransaction = gomonkey.ApplyFunc(core.ApplyTransaction, func(config *params.ChainConfig, bc core.ChainContext, author *common.Address, gp *core.GasPool, statedb *state.StateDB, header *types.Header, tx *types.Transaction, usedGas *uint64, cfg vm.Config, receiptProcessors ...core.ReceiptProcessor) (*types.Receipt, error) {
			return nil, errors.New("apply error")
		})
		defer patchApplyTransaction.Reset()
		parentBundle := &types.Bundle{
			Parent:     nil,
			ParentHash: common.Hash{},
			Counter:    0,
			Txs: []*types.Transaction{types.MustSignNewTx(testBankKey, signer, &types.LegacyTx{
				Nonce:    1,
				To:       &testUserAddress,
				Value:    big.NewInt(100_000),
				Gas:      params.TxGas,
				GasPrice: big.NewInt(1_200_000_000),
				Data:     nil,
			})},
			MaxBlockNumber: 100,
			Hint:           hint,
		}

		callData, _ := hexutil.Decode("0xf340fa01000000000000000000000000b4647b856cb9c3856d559c885bed8b43e0846a47")
		bundle := &types.Bundle{
			Parent:     parentBundle,
			ParentHash: parentBundle.Hash(),
			Counter:    1,
			Txs: []*types.Transaction{types.MustSignNewTx(testBankKey, signer, &types.LegacyTx{
				Nonce:    2,
				To:       &testUserAddress,
				Value:    big.NewInt(100_000),
				Gas:      params.TxGas,
				GasPrice: big.NewInt(1_200_000_000),
				Data:     callData,
			})},
			MaxBlockNumber: 100,
			Hint:           hint,
			RefundPercent:  90,
			RefundAddress:  common.HexToAddress("0xcccccccccccccccccccccccccccccccc"),
		}
		var txCount = 1
		var tempGasUsed uint64
		simBundle, bundleData, err := w.simulateBundle(env, bundle, stateDB, gasPool, &txCount, false, false, &tempGasUsed, common.Address{})
		So(err, ShouldNotBeNil)
		So(simBundle, ShouldBeNil)
		So(bundleData, ShouldBeNil)
	})
}
