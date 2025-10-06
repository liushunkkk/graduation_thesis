package ethapi

import (
	"context"
	"github.com/ethereum/go-ethereum"
	"github.com/ethereum/go-ethereum/accounts"
	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/common/hexutil"
	"github.com/ethereum/go-ethereum/common/ms"
	"github.com/ethereum/go-ethereum/consensus"
	"github.com/ethereum/go-ethereum/core"
	"github.com/ethereum/go-ethereum/core/bloombits"
	"github.com/ethereum/go-ethereum/core/state"
	"github.com/ethereum/go-ethereum/core/types"
	"github.com/ethereum/go-ethereum/core/vm"
	"github.com/ethereum/go-ethereum/crypto"
	"github.com/ethereum/go-ethereum/ethdb"
	"github.com/ethereum/go-ethereum/event"
	"github.com/ethereum/go-ethereum/invalid_tx"
	"github.com/ethereum/go-ethereum/params"
	"github.com/ethereum/go-ethereum/portal"
	"github.com/ethereum/go-ethereum/portal/zrpc_client/typed/rpc_portal/rpcv2"
	"github.com/ethereum/go-ethereum/rpc"
	. "github.com/smartystreets/goconvey/convey"
	"log"
	"math/big"
	"testing"
	"time"
)

type ScutumTestBackend struct {
	Bundle *types.Bundle
}

func (b ScutumTestBackend) SyncProgress() ethereum.SyncProgress { panic("implement me") }
func (b ScutumTestBackend) SuggestGasTipCap(ctx context.Context) (*big.Int, error) {
	panic("implement me")
}
func (b ScutumTestBackend) FeeHistory(ctx context.Context, blockCount uint64, lastBlock rpc.BlockNumber, rewardPercentiles []float64) (*big.Int, [][]*big.Int, []*big.Int, []float64, error) {
	panic("implement me")
}
func (b ScutumTestBackend) Chain() *core.BlockChain           { panic("implement me") }
func (b ScutumTestBackend) ChainDb() ethdb.Database           { panic("implement me") }
func (b ScutumTestBackend) AccountManager() *accounts.Manager { panic("implement me") }
func (b ScutumTestBackend) ExtRPCEnabled() bool               { panic("implement me") }
func (b ScutumTestBackend) RPCGasCap() uint64                 { panic("implement me") }
func (b ScutumTestBackend) RPCEVMTimeout() time.Duration      { panic("implement me") }
func (b ScutumTestBackend) RPCTxFeeCap() float64 {
	return 0
}
func (b ScutumTestBackend) UnprotectedAllowed() bool { return true }
func (b ScutumTestBackend) SetHead(number uint64)    { panic("implement me") }
func (b ScutumTestBackend) HeaderByNumber(ctx context.Context, number rpc.BlockNumber) (*types.Header, error) {
	panic("implement me")
}
func (b ScutumTestBackend) HeaderByHash(ctx context.Context, hash common.Hash) (*types.Header, error) {
	panic("implement me")
}
func (b ScutumTestBackend) HeaderByNumberOrHash(ctx context.Context, blockNrOrHash rpc.BlockNumberOrHash) (*types.Header, error) {
	panic("implement me")
}
func (b ScutumTestBackend) CurrentHeader() *types.Header {
	return &types.Header{
		Number: big.NewInt(43592052),
		Time:   uint64(1730356127),
	}
}
func (b ScutumTestBackend) CurrentBlock() *types.Header {
	return &types.Header{
		Number: big.NewInt(43592052),
		Time:   uint64(1730356127),
	}
}
func (b ScutumTestBackend) BlockByNumber(ctx context.Context, number rpc.BlockNumber) (*types.Block, error) {
	panic("implement me")
}
func (b ScutumTestBackend) BlockByHash(ctx context.Context, hash common.Hash) (*types.Block, error) {
	panic("implement me")
}
func (b ScutumTestBackend) BlockByNumberOrHash(ctx context.Context, blockNrOrHash rpc.BlockNumberOrHash) (*types.Block, error) {
	panic("implement me")
}
func (b ScutumTestBackend) StateAndHeaderByNumber(ctx context.Context, number rpc.BlockNumber) (*state.StateDB, *types.Header, error) {
	panic("implement me")
}
func (b ScutumTestBackend) StateAndHeaderByNumberOrHash(ctx context.Context, blockNrOrHash rpc.BlockNumberOrHash) (*state.StateDB, *types.Header, error) {
	panic("implement me")
}
func (b ScutumTestBackend) PendingBlockAndReceipts() (*types.Block, types.Receipts) {
	panic("implement me")
}
func (b ScutumTestBackend) GetReceipts(ctx context.Context, hash common.Hash) (types.Receipts, error) {
	panic("implement me")
}
func (b ScutumTestBackend) GetTd(ctx context.Context, hash common.Hash) *big.Int {
	panic("implement me")
}
func (b ScutumTestBackend) GetEVM(ctx context.Context, msg *core.Message, state *state.StateDB, header *types.Header, vmConfig *vm.Config, blockCtx *vm.BlockContext) *vm.EVM {
	panic("implement me")
}
func (b ScutumTestBackend) SubscribeChainEvent(ch chan<- core.ChainEvent) event.Subscription {
	panic("implement me")
}
func (b ScutumTestBackend) SubscribeChainHeadEvent(ch chan<- core.ChainHeadEvent) event.Subscription {
	panic("implement me")
}
func (b ScutumTestBackend) SubscribeChainSideEvent(ch chan<- core.ChainSideEvent) event.Subscription {
	panic("implement me")
}
func (b ScutumTestBackend) GetBlobSidecars(ctx context.Context, hash common.Hash) (types.BlobSidecars, error) {
	panic("implement me")
}
func (b ScutumTestBackend) SendTx(ctx context.Context, signedTx *types.Transaction) error {
	panic("implement me")
}
func (b ScutumTestBackend) SimulateGaslessBundle(bundle *types.Bundle) (*types.SimulateGaslessBundleResp, error) {
	panic("implement me")
}
func (b ScutumTestBackend) BundlePrice() *big.Int { panic("implement me") }
func (b ScutumTestBackend) GetTransaction(ctx context.Context, txHash common.Hash) (bool, *types.Transaction, common.Hash, uint64, uint64, error) {
	panic("implement me")
}
func (b ScutumTestBackend) GetPoolTransactions() (types.Transactions, error) { panic("implement me") }
func (b ScutumTestBackend) GetPoolTransaction(txHash common.Hash) *types.Transaction {
	panic("implement me")
}
func (b ScutumTestBackend) GetPoolNonce(ctx context.Context, addr common.Address) (uint64, error) {
	panic("implement me")
}
func (b ScutumTestBackend) Stats() (pending int, queued int) { panic("implement me") }
func (b ScutumTestBackend) TxPoolContent() (map[common.Address][]*types.Transaction, map[common.Address][]*types.Transaction) {
	panic("implement me")
}
func (b ScutumTestBackend) TxPoolContentFrom(addr common.Address) ([]*types.Transaction, []*types.Transaction) {
	panic("implement me")
}
func (b ScutumTestBackend) SubscribeNewTxsEvent(chan<- core.NewTxsEvent) event.Subscription {
	panic("implement me")
}
func (b ScutumTestBackend) ChainConfig() *params.ChainConfig {
	return params.BSCChainConfig
}
func (b ScutumTestBackend) Engine() consensus.Engine                     { panic("implement me") }
func (b ScutumTestBackend) CurrentValidators() ([]common.Address, error) { panic("implement me") }
func (b ScutumTestBackend) GetBody(ctx context.Context, hash common.Hash, number rpc.BlockNumber) (*types.Body, error) {
	panic("implement me")
}
func (b ScutumTestBackend) GetLogs(ctx context.Context, blockHash common.Hash, number uint64) ([][]*types.Log, error) {
	panic("implement me")
}
func (b ScutumTestBackend) SubscribeRemovedLogsEvent(ch chan<- core.RemovedLogsEvent) event.Subscription {
	panic("implement me")
}
func (b ScutumTestBackend) SubscribeLogsEvent(ch chan<- []*types.Log) event.Subscription {
	panic("implement me")
}
func (b ScutumTestBackend) SubscribePendingLogsEvent(ch chan<- []*types.Log) event.Subscription {
	panic("implement me")
}
func (b ScutumTestBackend) BloomStatus() (uint64, uint64) { panic("implement me") }
func (b ScutumTestBackend) ServiceFilter(ctx context.Context, session *bloombits.MatcherSession) {
	panic("implement me")
}
func (b ScutumTestBackend) SubscribeFinalizedHeaderEvent(ch chan<- core.FinalizedHeaderEvent) event.Subscription {
	panic("implement me")
}
func (b ScutumTestBackend) SubscribeNewVoteEvent(chan<- core.NewVoteEvent) event.Subscription {
	panic("implement me")
}
func (b ScutumTestBackend) MevRunning() bool            { panic("implement me") }
func (b ScutumTestBackend) MevParams() *types.MevParams { panic("implement me") }
func (b ScutumTestBackend) StartMev()                   { panic("implement me") }
func (b ScutumTestBackend) StopMev()                    { panic("implement me") }
func (b ScutumTestBackend) AddBuilder(builder common.Address, builderUrl string) error {
	panic("implement me")
}
func (b ScutumTestBackend) RemoveBuilder(builder common.Address) error { panic("implement me") }
func (b ScutumTestBackend) HasBuilder(builder common.Address) bool     { panic("implement me") }
func (b ScutumTestBackend) SendBid(ctx context.Context, bid *types.BidArgs) (common.Hash, error) {
	panic("implement me")
}
func (b ScutumTestBackend) BestBidGasFee(parentHash common.Hash) *big.Int { panic("implement me") }
func (b ScutumTestBackend) MinerInTurn() bool                             { panic("implement me") }
func (b *ScutumTestBackend) SendBundle(ctx context.Context, bundle *types.Bundle) error {
	b.Bundle = bundle
	return nil
}

func WithSystemUser() {
	invalid_tx.Server = &invalid_tx.InvalidTxServer{}
	invalid_tx.Server.SetServer(&ms.Server{Name: "hhh"})
	portal.UserServer = &portal.RpcUserServer{}
	defaultUser := &rpcv2.GetAllRpcInfoResponse{
		Url:              "default",
		RpcId:            "0000000000000000001",
		ChainId:          "56",
		HintLogs:         true,
		HintHash:         true,
		RefundPercent:    99,
		RefundRecipient:  "tx.origin",
		PrivacyPeriod:    0,
		PrivacyBuilder:   []string{"blockrazor"},
		BroadcastBuilder: []string{"blockrazor", "48club"},
	}
	portal.UserServer.RpcInfos.Store(defaultUser.RpcId, defaultUser)
	portal.UserServer.RpcInfos.Store(defaultUser.Url, defaultUser)
	fullPrivacyUser := &rpcv2.GetAllRpcInfoResponse{
		Url:              "fullprivacy",
		RpcId:            "0000000000000000002",
		ChainId:          "56",
		IsProtected:      true,
		PrivacyPeriod:    0,
		PrivacyBuilder:   []string{"blockrazor"},
		BroadcastBuilder: []string{"blockrazor", "48club"},
	}
	portal.UserServer.RpcInfos.Store(fullPrivacyUser.RpcId, fullPrivacyUser)
	portal.UserServer.RpcInfos.Store(fullPrivacyUser.Url, fullPrivacyUser)
	maxBackRunUser := &rpcv2.GetAllRpcInfoResponse{
		Url:                  "maxbackrun",
		RpcId:                "0000000000000000003",
		ChainId:              "56",
		RefundRecipient:      "tx.origin",
		RefundPercent:        99,
		HintHash:             true,
		HintLogs:             true,
		HintFunctionSelector: true,
		HintCalldata:         true,
		HintTo:               true,
		IsProtected:          true,
		PrivacyPeriod:        0,
		PrivacyBuilder:       []string{"blockrazor"},
		BroadcastBuilder:     []string{"blockrazor", "48club"},
	}
	portal.UserServer.RpcInfos.Store(maxBackRunUser.RpcId, maxBackRunUser)
	portal.UserServer.RpcInfos.Store(maxBackRunUser.Url, maxBackRunUser)
	myUser := &rpcv2.GetAllRpcInfoResponse{
		Url:                  "liushun",
		RpcId:                "0000000000000000004",
		ChainId:              "56",
		RefundRecipient:      "0xD87207B509af39345ABA929D9De22f52Ff175975",
		RefundPercent:        90,
		HintHash:             true,
		HintLogs:             true,
		HintFunctionSelector: true,
		HintCalldata:         true,
		HintTo:               true,
		HintGasPrice:         true,
		HintGasLimit:         true,
		HintNonce:            true,
		HintValue:            true,
		HintFrom:             true,
		IsProtected:          true,
		PrivacyPeriod:        2,
		PrivacyBuilder:       []string{"blockrazor"},
		BroadcastBuilder:     []string{"blockrazor", "48club", "smith"},
	}
	portal.UserServer.RpcInfos.Store(myUser.RpcId, myUser)
	portal.UserServer.RpcInfos.Store(myUser.Url, myUser)
}

func WithoutSystemUser() {
	portal.UserServer = &portal.RpcUserServer{}
	invalid_tx.Server = &invalid_tx.InvalidTxServer{}
	invalid_tx.Server.SetServer(&ms.Server{Name: "hhh"})
}

func GetMyHostContext() context.Context {
	ctx := context.WithValue(context.Background(), "Host", "liushun.bsc.blockrazor.xyz")
	ctx = context.WithValue(ctx, "URL", "/")
	return ctx
}

func GetMyIdContext() context.Context {
	ctx := context.WithValue(context.Background(), "Host", "bsc.blockrazor.xyz")
	ctx = context.WithValue(ctx, "URL", "/0000000000000000004")
	return ctx
}

func GetDefaultContext() context.Context {
	ctx := context.WithValue(context.Background(), "Host", "bsc.blockrazor.xyz")
	ctx = context.WithValue(ctx, "URL", "/")
	return ctx
}

func GetMaxBackRunContext() context.Context {
	ctx := context.WithValue(context.Background(), "Host", "bsc.blockrazor.xyz")
	ctx = context.WithValue(ctx, "URL", "/maxbackrun")
	return ctx
}

func GetFullPrivacyContext() context.Context {
	ctx := context.WithValue(context.Background(), "Host", "bsc.blockrazor.xyz")
	ctx = context.WithValue(ctx, "URL", "/fullprivacy")
	return ctx
}

func GetWithCorrectDefaultParamContext() context.Context {
	ctx := context.WithValue(context.Background(), "Host", "bsc.blockrazor.xyz")
	ctx = context.WithValue(ctx, "RefundPercent", "90")
	ctx = context.WithValue(ctx, "URL", "/")
	return ctx
}

func GetWithCorrectMaxBackRunParamContext() context.Context {
	ctx := context.WithValue(context.Background(), "Host", "bsc.blockrazor.xyz")
	ctx = context.WithValue(ctx, "RefundPercent", "98")
	ctx = context.WithValue(ctx, "URL", "/maxbackrun")
	return ctx
}

func GetWithLowErrorParamContext() context.Context {
	ctx := context.WithValue(context.Background(), "Host", "bsc.blockrazor.xyz")
	ctx = context.WithValue(ctx, "RefundPercent", "-1")
	ctx = context.WithValue(ctx, "URL", "/")
	return ctx
}

func GetWithOverErrorParamContext() context.Context {
	ctx := context.WithValue(context.Background(), "Host", "bsc.blockrazor.xyz")
	ctx = context.WithValue(ctx, "RefundPercent", "120")
	ctx = context.WithValue(ctx, "URL", "/")
	return ctx
}

func GetErrorHostContext() context.Context {
	ctx := context.WithValue(context.Background(), "Host", "888.bsc.blockrazor.xyz")
	ctx = context.WithValue(ctx, "URL", "/")
	return ctx
}

func GetOverHostContext() context.Context {
	ctx := context.WithValue(context.Background(), "Host", "a.b.bsc.blockrazor.xyz")
	ctx = context.WithValue(ctx, "URL", "/")
	return ctx
}

func GetLessHostContext() context.Context {
	ctx := context.WithValue(context.Background(), "Host", "blockrazor.xyz")
	ctx = context.WithValue(ctx, "URL", "/")
	return ctx
}

func GetErrorIdContext() context.Context {
	ctx := context.WithValue(context.Background(), "Host", "bsc.blockrazor.xyz")
	ctx = context.WithValue(ctx, "URL", "/9999999999999999999")
	return ctx
}

func GetCorrectTransaction() *types.Transaction {
	privateKey, err := crypto.HexToECDSA("0ad182d90bff7b643f70a7a8724d3c4f3a3cdca6711eef1301321695b984b36d")
	if err != nil {
		log.Fatal(err)
	}
	nonce := uint64(1)
	value := big.NewInt(1e4)
	gasLimit := uint64(21_000) // in units
	gasPrice := big.NewInt(1e9)
	toAddress := common.HexToAddress("0x43DdA9d1Ac023bd3593Dff5A1A677247Bb98fE11")
	tx := types.NewTx(&types.LegacyTx{
		Nonce:    nonce,
		GasPrice: gasPrice,
		Gas:      gasLimit,
		To:       &toAddress,
		Value:    value,
		Data:     nil,
	})
	signedTx, err := types.SignTx(tx, types.LatestSignerForChainID(big.NewInt(56)), privateKey)
	if err != nil {
		log.Fatal(err)
	}
	return signedTx
}

func GetCreateContractTransaction() *types.Transaction {
	privateKey, err := crypto.HexToECDSA("0ad182d90bff7b643f70a7a8724d3c4f3a3cdca6711eef1301321695b984b36d")
	if err != nil {
		log.Fatal(err)
	}
	nonce := uint64(1)
	value := big.NewInt(1e4)
	gasLimit := uint64(21_000) // in units
	gasPrice := big.NewInt(1e9)
	tx := types.NewTx(&types.LegacyTx{
		Nonce:    nonce,
		GasPrice: gasPrice,
		Gas:      gasLimit,
		Value:    value,
		Data:     nil,
	})
	signedTx, err := types.SignTx(tx, types.LatestSignerForChainID(big.NewInt(56)), privateKey)
	if err != nil {
		log.Fatal(err)
	}
	return signedTx
}

func GetFeeLowTransaction() *types.Transaction {
	privateKey, err := crypto.HexToECDSA("0ad182d90bff7b643f70a7a8724d3c4f3a3cdca6711eef1301321695b984b36d")
	if err != nil {
		log.Fatal(err)
	}
	nonce := uint64(1)
	value := big.NewInt(1e4)
	gasLimit := uint64(21_000)
	gasPrice := big.NewInt(1e8)
	toAddress := common.HexToAddress("0x43DdA9d1Ac023bd3593Dff5A1A677247Bb98fE11")
	tx := types.NewTx(&types.LegacyTx{
		Nonce:    nonce,
		GasPrice: gasPrice,
		Gas:      gasLimit,
		To:       &toAddress,
		Value:    value,
		Data:     nil,
	})
	signedTx, err := types.SignTx(tx, types.LatestSignerForChainID(big.NewInt(56)), privateKey)
	if err != nil {
		log.Fatal(err)
	}
	return signedTx
}

func GetTrueHint() map[string]bool {
	return map[string]bool{
		types.HintTo:               true,
		types.HintValue:            true,
		types.HintNonce:            true,
		types.HintFrom:             true,
		types.HintCallData:         true,
		types.HintFunctionSelector: true,
		types.HintHash:             true,
		types.HintGasPrice:         true,
		types.HintGasLimit:         true,
		types.HintLogs:             true,
	}
}

func GetFalseHint() map[string]bool {
	return map[string]bool{
		types.HintTo:               true,
		types.HintValue:            true,
		types.HintNonce:            true,
		types.HintFrom:             true,
		types.HintCallData:         true,
		types.HintFunctionSelector: true,
		types.HintHash:             true,
		types.HintGasPrice:         true,
		types.HintGasLimit:         true,
		types.HintLogs:             true,
		"sss":                      true,
	}
}

func GetErrorTxBundle() types.SendMevBundleArgs {
	t1 := GetCorrectTransaction()
	b1, err := t1.MarshalBinary()
	if err != nil {
		log.Fatal(err)
	}
	bundle := types.SendMevBundleArgs{
		Hint:              GetTrueHint(),
		Txs:               []hexutil.Bytes{b1, b1[2:]},
		Hash:              common.Hash{},
		RevertingTxHashes: []common.Hash{t1.Hash()},
		RefundPercent:     95,
		RefundAddress:     *t1.To(),
		MaxBlockNumber:    43592060,
	}
	return bundle
}

func GetCorrectBackRunBundle() types.SendMevBundleArgs {
	signedTx := GetCorrectTransaction()
	b1, err := signedTx.MarshalBinary()
	if err != nil {
		log.Fatal(err)
	}
	bundle := types.SendMevBundleArgs{
		Hint:              GetTrueHint(),
		Txs:               []hexutil.Bytes{b1},
		Hash:              common.HexToHash("0x0476b176bd132c1183bde91b7dd73af174243dc5f898d5e21e34d12c5bb73f15"),
		RevertingTxHashes: make([]common.Hash, 0),
		RefundPercent:     95,
		RefundAddress:     *signedTx.To(),
	}
	return bundle
}

func GetTooManyBackRunBundle() types.SendMevBundleArgs {
	signedTx := GetCorrectTransaction()
	b1, err := signedTx.MarshalBinary()
	if err != nil {
		log.Fatal(err)
	}
	bundle := types.SendMevBundleArgs{
		Hint:              GetTrueHint(),
		Txs:               []hexutil.Bytes{b1, b1},
		Hash:              common.HexToHash("0x0476b176bd132c1183bde91b7dd73af174243dc5f898d5e21e34d12c5bb73f15"),
		RevertingTxHashes: make([]common.Hash, 0),
		RefundPercent:     95,
		RefundAddress:     *signedTx.To(),
	}
	return bundle
}

func GetCorrectRawBundle() types.SendMevBundleArgs {
	t1 := GetCorrectTransaction()
	b1, err := t1.MarshalBinary()
	if err != nil {
		log.Fatal(err)
	}
	bundle := types.SendMevBundleArgs{
		Hint:              GetTrueHint(),
		Txs:               []hexutil.Bytes{b1, b1},
		Hash:              common.Hash{},
		RevertingTxHashes: []common.Hash{t1.Hash()},
		RefundPercent:     95,
		RefundAddress:     *t1.To(),
		MaxBlockNumber:    43592060,
	}
	return bundle
}

func GetTooManyRawBundle() types.SendMevBundleArgs {
	signedTx := GetCorrectTransaction()
	b1, err := signedTx.MarshalBinary()
	if err != nil {
		log.Fatal(err)
	}
	var ts []hexutil.Bytes
	for i := 0; i < 60; i++ {
		ts = append(ts, b1)
	}
	bundle := types.SendMevBundleArgs{
		Hint:              GetTrueHint(),
		Txs:               ts,
		Hash:              common.Hash{},
		RevertingTxHashes: make([]common.Hash, 0),
		RefundPercent:     95,
		RefundAddress:     *signedTx.To(),
	}
	return bundle
}

func GetNoTxBundle() types.SendMevBundleArgs {
	toAddress := common.HexToAddress("0x9Abae1b279A4Be25AEaE49a33e807cDd3cCFFa0C")
	bundle := types.SendMevBundleArgs{
		Hint:              GetTrueHint(),
		Txs:               []hexutil.Bytes{},
		Hash:              common.HexToHash("0x0476b176bd132c1183bde91b7dd73af174243dc5f898d5e21e34d12c5bb73f15"),
		RevertingTxHashes: make([]common.Hash, 0),
		RefundPercent:     95,
		RefundAddress:     toAddress,
	}
	return bundle
}

func GetBlockNumberOverBundle() types.SendMevBundleArgs {
	signedTx := GetCorrectTransaction()
	b1, err := signedTx.MarshalBinary()
	if err != nil {
		log.Fatal(err)
	}
	bundle := types.SendMevBundleArgs{
		Hint:              GetTrueHint(),
		Txs:               []hexutil.Bytes{b1},
		Hash:              common.HexToHash("0x0476b176bd132c1183bde91b7dd73af174243dc5f898d5e21e34d12c5bb73f15"),
		RevertingTxHashes: make([]common.Hash, 0),
		RefundPercent:     95,
		RefundAddress:     *signedTx.To(),
		MaxBlockNumber:    43593052,
	}
	return bundle
}

func GetBlockNumberLowBundle() types.SendMevBundleArgs {
	signedTx := GetCorrectTransaction()
	b1, err := signedTx.MarshalBinary()
	if err != nil {
		log.Fatal(err)
	}
	bundle := types.SendMevBundleArgs{
		Hint:              GetTrueHint(),
		Txs:               []hexutil.Bytes{b1},
		Hash:              common.HexToHash("0x0476b176bd132c1183bde91b7dd73af174243dc5f898d5e21e34d12c5bb73f15"),
		RevertingTxHashes: make([]common.Hash, 0),
		RefundPercent:     95,
		RefundAddress:     *signedTx.To(),
		MaxBlockNumber:    43591052,
	}
	return bundle
}

func GetRevertingHashErrorBundle() types.SendMevBundleArgs {
	t1 := GetCorrectTransaction()
	b1, err := t1.MarshalBinary()
	if err != nil {
		log.Fatal(err)
	}
	bundle := types.SendMevBundleArgs{
		Hint:              GetTrueHint(),
		Txs:               []hexutil.Bytes{b1, b1},
		Hash:              common.Hash{},
		RevertingTxHashes: []common.Hash{common.HexToHash("0x23d55baf7af0bc54610f074aedddcfc350bc62a5dc5e9fb25c96fcfee44d8ceb")},
		RefundPercent:     95,
		RefundAddress:     *t1.To(),
		MaxBlockNumber:    43592060,
	}
	return bundle
}

func GetErrorHintBundle() types.SendMevBundleArgs {
	t1 := GetCorrectTransaction()
	b1, err := t1.MarshalBinary()
	if err != nil {
		log.Fatal(err)
	}
	bundle := types.SendMevBundleArgs{
		Hint:              GetFalseHint(),
		Txs:               []hexutil.Bytes{b1, b1},
		Hash:              common.Hash{},
		RevertingTxHashes: []common.Hash{t1.Hash()},
		RefundPercent:     95,
		RefundAddress:     *t1.To(),
		MaxBlockNumber:    43592060,
	}
	return bundle
}

func GetErrorRefundAddressBundle() types.SendMevBundleArgs {
	t1 := GetCorrectTransaction()
	b1, err := t1.MarshalBinary()
	if err != nil {
		log.Fatal(err)
	}
	bundle := types.SendMevBundleArgs{
		Hint:              GetTrueHint(),
		Txs:               []hexutil.Bytes{b1, b1},
		Hash:              common.Hash{},
		RevertingTxHashes: []common.Hash{t1.Hash()},
		RefundPercent:     95,
		MaxBlockNumber:    43592060,
	}
	return bundle
}

func GetErrorRefundPercentBundle() types.SendMevBundleArgs {
	t1 := GetCorrectTransaction()
	b1, err := t1.MarshalBinary()
	if err != nil {
		log.Fatal(err)
	}
	bundle := types.SendMevBundleArgs{
		Hint:              GetTrueHint(),
		Txs:               []hexutil.Bytes{b1, b1},
		Hash:              common.Hash{},
		RevertingTxHashes: []common.Hash{t1.Hash()},
		RefundAddress:     *t1.To(),
		RefundPercent:     140,
		MaxBlockNumber:    43592060,
	}
	return bundle
}

func GetErrorRefundPercentNoHintBundle() types.SendMevBundleArgs {
	t1 := GetCorrectTransaction()
	b1, err := t1.MarshalBinary()
	if err != nil {
		log.Fatal(err)
	}
	bundle := types.SendMevBundleArgs{
		Txs:               []hexutil.Bytes{b1, b1},
		Hash:              common.Hash{},
		RevertingTxHashes: []common.Hash{t1.Hash()},
		RefundPercent:     120,
		MaxBlockNumber:    43592060,
	}
	return bundle
}

func GetTrueRefundPercentNoHintBundle() types.SendMevBundleArgs {
	t1 := GetCorrectTransaction()
	b1, err := t1.MarshalBinary()
	if err != nil {
		log.Fatal(err)
	}
	bundle := types.SendMevBundleArgs{
		Txs:               []hexutil.Bytes{b1, b1},
		Hash:              common.Hash{},
		RevertingTxHashes: []common.Hash{t1.Hash()},
		RefundPercent:     95,
		MaxBlockNumber:    43592060,
	}
	return bundle
}

func SoCommonBuilder(b *ScutumTestBackend) {
	So(b.Bundle.PrivacyPeriod, ShouldEqual, 0)
	So(b.Bundle.PrivacyBuilder, ShouldContain, "blockrazor")
	So(b.Bundle.BroadcastBuilder, ShouldContain, "blockrazor")
	So(b.Bundle.BroadcastBuilder, ShouldContain, "48club")
}

func SoLiuShunBuilder(b *ScutumTestBackend) {
	So(b.Bundle.PrivacyPeriod, ShouldEqual, 2)
	So(b.Bundle.PrivacyBuilder, ShouldContain, "blockrazor")
	So(b.Bundle.BroadcastBuilder, ShouldContain, "blockrazor")
	So(b.Bundle.BroadcastBuilder, ShouldContain, "48club")
	So(b.Bundle.BroadcastBuilder, ShouldContain, "smith")
}

func SoDefaultHint(b *ScutumTestBackend) {
	So(b.Bundle.Hint[types.HintHash], ShouldEqual, true)
	So(b.Bundle.Hint[types.HintLogs], ShouldEqual, true)
	So(b.Bundle.Hint[types.HintTo], ShouldNotEqual, true)
	So(b.Bundle.Hint[types.HintValue], ShouldNotEqual, true)
	So(b.Bundle.Hint[types.HintNonce], ShouldNotEqual, true)
	So(b.Bundle.Hint[types.HintFrom], ShouldNotEqual, true)
	So(b.Bundle.Hint[types.HintFunctionSelector], ShouldNotEqual, true)
	So(b.Bundle.Hint[types.HintCallData], ShouldNotEqual, true)
	So(b.Bundle.Hint[types.HintGasLimit], ShouldNotEqual, true)
	So(b.Bundle.Hint[types.HintGasPrice], ShouldNotEqual, true)
}

func SoFullPrivacyHint(b *ScutumTestBackend) {
	So(b.Bundle.Hint[types.HintHash], ShouldNotEqual, true)
	So(b.Bundle.Hint[types.HintLogs], ShouldNotEqual, true)
	So(b.Bundle.Hint[types.HintTo], ShouldNotEqual, true)
	So(b.Bundle.Hint[types.HintValue], ShouldNotEqual, true)
	So(b.Bundle.Hint[types.HintNonce], ShouldNotEqual, true)
	So(b.Bundle.Hint[types.HintFrom], ShouldNotEqual, true)
	So(b.Bundle.Hint[types.HintFunctionSelector], ShouldNotEqual, true)
	So(b.Bundle.Hint[types.HintCallData], ShouldNotEqual, true)
	So(b.Bundle.Hint[types.HintGasLimit], ShouldNotEqual, true)
	So(b.Bundle.Hint[types.HintGasPrice], ShouldNotEqual, true)
}

func SoMaxBackRunHint(b *ScutumTestBackend) {
	So(b.Bundle.Hint[types.HintHash], ShouldEqual, true)
	So(b.Bundle.Hint[types.HintLogs], ShouldEqual, true)
	So(b.Bundle.Hint[types.HintTo], ShouldEqual, true)
	So(b.Bundle.Hint[types.HintValue], ShouldNotEqual, true)
	So(b.Bundle.Hint[types.HintNonce], ShouldNotEqual, true)
	So(b.Bundle.Hint[types.HintFrom], ShouldNotEqual, true)
	So(b.Bundle.Hint[types.HintFunctionSelector], ShouldEqual, true)
	So(b.Bundle.Hint[types.HintCallData], ShouldEqual, true)
	So(b.Bundle.Hint[types.HintGasLimit], ShouldNotEqual, true)
	So(b.Bundle.Hint[types.HintGasPrice], ShouldNotEqual, true)
}

func SoFullOpenHint(b *ScutumTestBackend) {
	So(b.Bundle.Hint[types.HintHash], ShouldEqual, true)
	So(b.Bundle.Hint[types.HintLogs], ShouldEqual, true)
	So(b.Bundle.Hint[types.HintTo], ShouldEqual, true)
	So(b.Bundle.Hint[types.HintValue], ShouldEqual, true)
	So(b.Bundle.Hint[types.HintNonce], ShouldEqual, true)
	So(b.Bundle.Hint[types.HintFrom], ShouldEqual, true)
	So(b.Bundle.Hint[types.HintFunctionSelector], ShouldEqual, true)
	So(b.Bundle.Hint[types.HintCallData], ShouldEqual, true)
	So(b.Bundle.Hint[types.HintGasLimit], ShouldEqual, true)
	So(b.Bundle.Hint[types.HintGasPrice], ShouldEqual, true)
}

func SoDefaultNoUserBundle(b *ScutumTestBackend) {
	So(b.Bundle.RPCID, ShouldEqual, "default")
	SoCommonBuilder(b)
	SoFullOpenHint(b)
	So(b.Bundle.RefundPercent, ShouldEqual, 95)
}

func SoDefaultBundle(b *ScutumTestBackend) {
	So(b.Bundle.RPCID, ShouldEqual, "0000000000000000001")
	SoCommonBuilder(b)
	SoFullOpenHint(b)
	So(b.Bundle.RefundPercent, ShouldEqual, 95)
}

func SoLiuShunBundle(b *ScutumTestBackend) {
	So(b.Bundle.RPCID, ShouldEqual, "0000000000000000004")
	SoLiuShunBuilder(b)
	SoFullOpenHint(b)
	So(b.Bundle.RefundPercent, ShouldEqual, 95)
	So(b.Bundle.RefundAddress, ShouldEqual, common.HexToAddress("0x43DdA9d1Ac023bd3593Dff5A1A677247Bb98fE11"))
}

func SoDefaultRefundPercentNoHintBundle(b *ScutumTestBackend) {
	So(b.Bundle.RPCID, ShouldEqual, "0000000000000000001")
	SoCommonBuilder(b)
	SoFullPrivacyHint(b)
	So(b.Bundle.RefundPercent, ShouldEqual, 95)
}

func SoDefaultParamNoUser(b *ScutumTestBackend) {
	So(b.Bundle.RPCID, ShouldEqual, "default")
	SoCommonBuilder(b)
	SoDefaultHint(b)
	So(b.Bundle.RefundPercent, ShouldEqual, 90)
}

func SoDefaultParam(b *ScutumTestBackend) {
	So(b.Bundle.RPCID, ShouldEqual, "0000000000000000001")
	SoCommonBuilder(b)
	SoDefaultHint(b)
	So(b.Bundle.RefundPercent, ShouldEqual, 90)
}

func SoDefaultNoUser(b *ScutumTestBackend) {
	So(b.Bundle.RPCID, ShouldEqual, "default")
	SoCommonBuilder(b)
	SoDefaultHint(b)
	So(b.Bundle.RefundPercent, ShouldEqual, 99)
}

func SoDefault(b *ScutumTestBackend) {
	So(b.Bundle.RPCID, ShouldEqual, "0000000000000000001")
	SoCommonBuilder(b)
	SoDefaultHint(b)
	So(b.Bundle.RefundPercent, ShouldEqual, 99)
}

func SoMaxBackRunParamNoUser(b *ScutumTestBackend) {
	So(b.Bundle.RPCID, ShouldEqual, "maxbackrun")
	SoCommonBuilder(b)
	SoMaxBackRunHint(b)
	So(b.Bundle.RefundPercent, ShouldEqual, 98)
}

func SoMaxBackRunParam(b *ScutumTestBackend) {
	So(b.Bundle.RPCID, ShouldEqual, "0000000000000000003")
	SoCommonBuilder(b)
	SoMaxBackRunHint(b)
	So(b.Bundle.RefundPercent, ShouldEqual, 98)
}

func SoFullPrivacyNoUser(b *ScutumTestBackend) {
	So(b.Bundle.RPCID, ShouldEqual, "fullprivacy")
	SoCommonBuilder(b)
	SoFullPrivacyHint(b)
	So(b.Bundle.RefundPercent, ShouldEqual, 99)
}

func SoFullPrivacy(b *ScutumTestBackend) {
	So(b.Bundle.RPCID, ShouldEqual, "0000000000000000002")
	SoCommonBuilder(b)
	SoFullPrivacyHint(b)
	So(b.Bundle.RefundPercent, ShouldEqual, 99)
}

func SoMaxBackRunNoUser(b *ScutumTestBackend) {
	So(b.Bundle.RPCID, ShouldEqual, "maxbackrun")
	SoCommonBuilder(b)
	SoMaxBackRunHint(b)
	So(b.Bundle.RefundPercent, ShouldEqual, 99)
}

func SoMaxBackRun(b *ScutumTestBackend) {
	So(b.Bundle.RPCID, ShouldEqual, "0000000000000000003")
	SoCommonBuilder(b)
	SoMaxBackRunHint(b)
	So(b.Bundle.RefundPercent, ShouldEqual, 99)
}

func SoLiuShun(b *ScutumTestBackend) {
	So(b.Bundle.RPCID, ShouldEqual, "0000000000000000004")
	SoLiuShunBuilder(b)
	SoFullOpenHint(b)
	So(b.Bundle.RefundPercent, ShouldEqual, 90)
	So(b.Bundle.RefundAddress, ShouldEqual, common.HexToAddress("0xD87207B509af39345ABA929D9De22f52Ff175975"))
}

func TestSubmitTransaction(t *testing.T) {
	scutumTestBackend := &ScutumTestBackend{}
	Convey("with portal system user", t, func() {
		WithSystemUser()
		Convey("default context", func() {
			Convey("correct transaction", func() {
				_, err := SubmitTransaction(
					GetDefaultContext(),
					scutumTestBackend,
					GetCorrectTransaction())
				So(err, ShouldBeNil)
				SoDefault(scutumTestBackend)
			})
			Convey("fee low transaction", func() {
				_, err := SubmitTransaction(
					GetDefaultContext(),
					scutumTestBackend,
					GetFeeLowTransaction())
				So(err, ShouldNotBeNil)
			})
			Convey("correct create contract transaction", func() {
				_, err := SubmitTransaction(
					GetDefaultContext(),
					scutumTestBackend,
					GetCreateContractTransaction())
				So(err, ShouldBeNil)
				SoDefault(scutumTestBackend)
			})
		})

		Convey("maxbackrun context", func() {
			Convey("correct transaction", func() {
				_, err := SubmitTransaction(
					GetMaxBackRunContext(),
					scutumTestBackend,
					GetCorrectTransaction())
				So(err, ShouldBeNil)
				SoMaxBackRun(scutumTestBackend)
			})
			Convey("fee low transaction", func() {
				_, err := SubmitTransaction(
					GetMaxBackRunContext(),
					scutumTestBackend,
					GetFeeLowTransaction())
				So(err, ShouldNotBeNil)
			})
			Convey("correct create contract transaction", func() {
				_, err := SubmitTransaction(
					GetMaxBackRunContext(),
					scutumTestBackend,
					GetCreateContractTransaction())
				So(err, ShouldBeNil)
				SoMaxBackRun(scutumTestBackend)
			})
		})

		Convey("fullprivacy context", func() {
			Convey("correct transaction", func() {
				_, err := SubmitTransaction(
					GetFullPrivacyContext(),
					scutumTestBackend,
					GetCorrectTransaction())
				So(err, ShouldBeNil)
				SoFullPrivacy(scutumTestBackend)
			})
			Convey("fee low transaction", func() {
				_, err := SubmitTransaction(
					GetFullPrivacyContext(),
					scutumTestBackend,
					GetFeeLowTransaction())
				So(err, ShouldNotBeNil)
			})
			Convey("correct create contract transaction", func() {
				_, err := SubmitTransaction(
					GetFullPrivacyContext(),
					scutumTestBackend,
					GetCreateContractTransaction())
				So(err, ShouldBeNil)
				SoFullPrivacy(scutumTestBackend)
			})
		})

		Convey("my host context", func() {
			Convey("correct transaction", func() {
				_, err := SubmitTransaction(
					GetMyHostContext(),
					scutumTestBackend,
					GetCorrectTransaction())
				So(err, ShouldBeNil)
				SoLiuShun(scutumTestBackend)
			})
		})

		Convey("my id context", func() {
			Convey("correct transaction", func() {
				_, err := SubmitTransaction(
					GetMyIdContext(),
					scutumTestBackend,
					GetCorrectTransaction())
				So(err, ShouldBeNil)
				SoLiuShun(scutumTestBackend)
			})
		})

		Convey("error host context", func() {
			Convey("correct transaction", func() {
				_, err := SubmitTransaction(
					GetErrorHostContext(),
					scutumTestBackend,
					GetCorrectTransaction())
				So(err, ShouldBeNil)
				SoDefault(scutumTestBackend)
			})
		})

		Convey("error id context", func() {
			Convey("correct transaction", func() {
				_, err := SubmitTransaction(
					GetErrorIdContext(),
					scutumTestBackend,
					GetCorrectTransaction())
				So(err, ShouldBeNil)
				SoDefault(scutumTestBackend)
			})
		})

		Convey("default mode with correct RefundPercent param id context", func() {
			Convey("correct transaction", func() {
				_, err := SubmitTransaction(
					GetWithCorrectDefaultParamContext(),
					scutumTestBackend,
					GetCorrectTransaction())
				So(err, ShouldBeNil)
				SoDefaultParam(scutumTestBackend)
			})
		})

		Convey("default mode with low RefundPercent param id context", func() {
			Convey("correct transaction", func() {
				_, err := SubmitTransaction(
					GetWithLowErrorParamContext(),
					scutumTestBackend,
					GetCorrectTransaction())
				So(err, ShouldNotBeNil)
			})
		})

		Convey("default mode with over RefundPercent param id context", func() {
			Convey("correct transaction", func() {
				_, err := SubmitTransaction(
					GetWithOverErrorParamContext(),
					scutumTestBackend,
					GetCorrectTransaction())
				So(err, ShouldNotBeNil)
			})
		})

		Convey("maxbackrun mode with correct RefundPercent param id context", func() {
			Convey("correct transaction", func() {
				_, err := SubmitTransaction(
					GetWithCorrectMaxBackRunParamContext(),
					scutumTestBackend,
					GetCorrectTransaction())
				So(err, ShouldBeNil)
				SoMaxBackRunParam(scutumTestBackend)
			})
		})

		Convey("five periods host context", func() {
			Convey("correct transaction", func() {
				_, err := SubmitTransaction(
					GetOverHostContext(),
					scutumTestBackend,
					GetCorrectTransaction())
				So(err, ShouldNotBeNil)
			})
		})

		Convey("two periods host context", func() {
			Convey("correct transaction", func() {
				_, err := SubmitTransaction(
					GetLessHostContext(),
					scutumTestBackend,
					GetCorrectTransaction())
				So(err, ShouldNotBeNil)
			})
		})
	})

	Convey("without portal system user", t, func() {
		WithoutSystemUser()
		Convey("default context", func() {
			Convey("correct transaction", func() {
				_, err := SubmitTransaction(
					GetDefaultContext(),
					scutumTestBackend,
					GetCorrectTransaction())
				So(err, ShouldBeNil)
				SoDefaultNoUser(scutumTestBackend)
			})
			Convey("fee low transaction", func() {
				_, err := SubmitTransaction(
					GetDefaultContext(),
					scutumTestBackend,
					GetFeeLowTransaction())
				So(err, ShouldNotBeNil)
			})
			Convey("correct create contract transaction", func() {
				_, err := SubmitTransaction(
					GetDefaultContext(),
					scutumTestBackend,
					GetCreateContractTransaction())
				So(err, ShouldBeNil)
				SoDefaultNoUser(scutumTestBackend)
			})
		})

		Convey("maxbackrun context", func() {
			Convey("correct transaction", func() {
				_, err := SubmitTransaction(
					GetMaxBackRunContext(),
					scutumTestBackend,
					GetCorrectTransaction())
				So(err, ShouldBeNil)
				SoMaxBackRunNoUser(scutumTestBackend)
			})
			Convey("fee low transaction", func() {
				_, err := SubmitTransaction(
					GetMaxBackRunContext(),
					scutumTestBackend,
					GetFeeLowTransaction())
				So(err, ShouldNotBeNil)
			})
			Convey("correct create contract transaction", func() {
				_, err := SubmitTransaction(
					GetMaxBackRunContext(),
					scutumTestBackend,
					GetCreateContractTransaction())
				So(err, ShouldBeNil)
				SoMaxBackRunNoUser(scutumTestBackend)
			})
		})

		Convey("fullprivacy context", func() {
			Convey("correct transaction", func() {
				_, err := SubmitTransaction(
					GetFullPrivacyContext(),
					scutumTestBackend,
					GetCorrectTransaction())
				So(err, ShouldBeNil)
				SoFullPrivacyNoUser(scutumTestBackend)
			})
			Convey("fee low transaction", func() {
				_, err := SubmitTransaction(
					GetFullPrivacyContext(),
					scutumTestBackend,
					GetFeeLowTransaction())
				So(err, ShouldNotBeNil)
			})
			Convey("correct create contract transaction", func() {
				_, err := SubmitTransaction(
					GetFullPrivacyContext(),
					scutumTestBackend,
					GetCreateContractTransaction())
				So(err, ShouldBeNil)
				SoFullPrivacyNoUser(scutumTestBackend)
			})
		})

		Convey("my host context", func() {
			Convey("correct transaction", func() {
				_, err := SubmitTransaction(
					GetMyHostContext(),
					scutumTestBackend,
					GetCorrectTransaction())
				So(err, ShouldBeNil)
				SoDefaultNoUser(scutumTestBackend)
			})
		})

		Convey("my id context", func() {
			Convey("correct transaction", func() {
				_, err := SubmitTransaction(
					GetMyIdContext(),
					scutumTestBackend,
					GetCorrectTransaction())
				So(err, ShouldBeNil)
				SoDefaultNoUser(scutumTestBackend)
			})
		})

		Convey("error host context", func() {
			Convey("correct transaction", func() {
				_, err := SubmitTransaction(
					GetErrorHostContext(),
					scutumTestBackend,
					GetCorrectTransaction())
				So(err, ShouldBeNil)
			})
			SoDefaultNoUser(scutumTestBackend)
		})

		Convey("error id context", func() {
			Convey("correct transaction", func() {
				_, err := SubmitTransaction(
					GetErrorIdContext(),
					scutumTestBackend,
					GetCorrectTransaction())
				So(err, ShouldBeNil)
				SoDefaultNoUser(scutumTestBackend)
			})
		})

		Convey("default mode with correct RefundPercent param id context", func() {
			Convey("correct transaction", func() {
				_, err := SubmitTransaction(
					GetWithCorrectDefaultParamContext(),
					scutumTestBackend,
					GetCorrectTransaction())
				So(err, ShouldBeNil)
				SoDefaultParamNoUser(scutumTestBackend)
			})
		})

		Convey("default mode with low RefundPercent param id context", func() {
			Convey("correct transaction", func() {
				_, err := SubmitTransaction(
					GetWithLowErrorParamContext(),
					scutumTestBackend,
					GetCorrectTransaction())
				So(err, ShouldNotBeNil)
			})
		})

		Convey("default mode with over RefundPercent param id context", func() {
			Convey("correct transaction", func() {
				_, err := SubmitTransaction(
					GetWithOverErrorParamContext(),
					scutumTestBackend,
					GetCorrectTransaction())
				So(err, ShouldNotBeNil)
			})
		})

		Convey("macbackrun mode with correct RefundPercent param id context", func() {
			Convey("correct transaction", func() {
				_, err := SubmitTransaction(
					GetWithCorrectMaxBackRunParamContext(),
					scutumTestBackend,
					GetCorrectTransaction())
				So(err, ShouldBeNil)
				SoMaxBackRunParamNoUser(scutumTestBackend)
			})
		})

		Convey("five periods host context", func() {
			Convey("correct transaction", func() {
				_, err := SubmitTransaction(
					GetOverHostContext(),
					scutumTestBackend,
					GetCorrectTransaction())
				So(err, ShouldNotBeNil)
			})
		})

		Convey("two periods host context", func() {
			Convey("correct transaction", func() {
				_, err := SubmitTransaction(
					GetLessHostContext(),
					scutumTestBackend,
					GetCorrectTransaction())
				So(err, ShouldNotBeNil)
			})
		})
	})
}

func TestSendMevBundle(t *testing.T) {
	api := &PrivateTxBundleAPI{
		b: &ScutumTestBackend{},
	}

	Convey("validate bundle", t, func() {
		WithSystemUser()
		Convey("correct raw bundle", func() {
			_, err := api.SendMevBundle(GetDefaultContext(), GetCorrectRawBundle())
			So(err, ShouldBeNil)
			SoDefaultBundle(api.b.(*ScutumTestBackend))
		})
		Convey("correct back run bundle", func() {
			_, err := api.SendMevBundle(GetDefaultContext(), GetCorrectBackRunBundle())
			So(err, ShouldBeNil)
			SoDefaultBundle(api.b.(*ScutumTestBackend))
		})
		Convey("no tx bundle", func() {
			_, err := api.SendMevBundle(GetDefaultContext(), GetNoTxBundle())
			So(err, ShouldNotBeNil)
		})
		Convey("too many tx in back run bundle", func() {
			_, err := api.SendMevBundle(GetDefaultContext(), GetTooManyBackRunBundle())
			So(err, ShouldNotBeNil)
		})
		Convey("too many tx in raw bundle", func() {
			_, err := api.SendMevBundle(GetDefaultContext(), GetTooManyRawBundle())
			So(err, ShouldNotBeNil)
		})
		Convey("block number over bundle", func() {
			_, err := api.SendMevBundle(GetDefaultContext(), GetBlockNumberOverBundle())
			So(err, ShouldNotBeNil)
		})
		Convey("block number low bundle", func() {
			_, err := api.SendMevBundle(GetDefaultContext(), GetBlockNumberLowBundle())
			So(err, ShouldNotBeNil)
		})
		Convey("error tx bundle", func() {
			_, err := api.SendMevBundle(GetDefaultContext(), GetErrorTxBundle())
			So(err, ShouldNotBeNil)
		})
		Convey("reverting tx error bundle", func() {
			_, err := api.SendMevBundle(GetDefaultContext(), GetRevertingHashErrorBundle())
			So(err, ShouldNotBeNil)
		})
		Convey("hint error bundle", func() {
			_, err := api.SendMevBundle(GetDefaultContext(), GetErrorHintBundle())
			So(err, ShouldNotBeNil)
		})
		Convey("RefundAddress error in share bundle", func() {
			_, err := api.SendMevBundle(GetDefaultContext(), GetErrorRefundAddressBundle())
			So(err, ShouldNotBeNil)
		})
		Convey("RefundPercent error in share bundle", func() {
			_, err := api.SendMevBundle(GetDefaultContext(), GetErrorRefundPercentBundle())
			So(err, ShouldNotBeNil)
		})
		Convey("RefundPercent error in no share bundle", func() {
			_, err := api.SendMevBundle(GetDefaultContext(), GetErrorRefundPercentNoHintBundle())
			So(err, ShouldNotBeNil)
		})
		Convey("RefundPercent true in share bundle", func() {
			_, err := api.SendMevBundle(GetDefaultContext(), GetTrueRefundPercentNoHintBundle())
			So(err, ShouldBeNil)
			SoDefaultRefundPercentNoHintBundle(api.b.(*ScutumTestBackend))
		})
	})

	Convey("With system user", t, func() {
		WithSystemUser()
		Convey("default context", func() {
			_, err := api.SendMevBundle(GetDefaultContext(), GetCorrectRawBundle())
			So(err, ShouldBeNil)
			SoDefaultBundle(api.b.(*ScutumTestBackend))
		})
		Convey("GetFullPrivacyContext", func() {
			_, err := api.SendMevBundle(GetFullPrivacyContext(), GetCorrectRawBundle())
			So(err, ShouldBeNil)
			SoDefaultBundle(api.b.(*ScutumTestBackend))
		})
		Convey("GetMaxBackRunContext", func() {
			_, err := api.SendMevBundle(GetMaxBackRunContext(), GetCorrectRawBundle())
			So(err, ShouldBeNil)
			SoDefaultBundle(api.b.(*ScutumTestBackend))
		})
		Convey("GetMyHostContext", func() {
			_, err := api.SendMevBundle(GetMyHostContext(), GetCorrectRawBundle())
			So(err, ShouldBeNil)
			SoLiuShunBundle(api.b.(*ScutumTestBackend))
		})
		Convey("GetMyIdContext", func() {
			_, err := api.SendMevBundle(GetMyIdContext(), GetCorrectRawBundle())
			So(err, ShouldBeNil)
			SoLiuShunBundle(api.b.(*ScutumTestBackend))
		})
		Convey("GetErrorIdContext", func() {
			_, err := api.SendMevBundle(GetErrorIdContext(), GetCorrectRawBundle())
			So(err, ShouldBeNil)
			SoDefaultBundle(api.b.(*ScutumTestBackend))
		})
		Convey("GetErrorHostContext", func() {
			_, err := api.SendMevBundle(GetErrorHostContext(), GetCorrectRawBundle())
			So(err, ShouldBeNil)
			SoDefaultBundle(api.b.(*ScutumTestBackend))
		})
		Convey("GetOverHostContext", func() {
			_, err := api.SendMevBundle(GetOverHostContext(), GetCorrectRawBundle())
			So(err, ShouldNotBeNil)
		})
		Convey("GetLessHostContext", func() {
			_, err := api.SendMevBundle(GetLessHostContext(), GetCorrectRawBundle())
			So(err, ShouldNotBeNil)
		})
	})

	Convey("Without system user", t, func() {
		WithoutSystemUser()
		Convey("default context", func() {
			_, err := api.SendMevBundle(GetDefaultContext(), GetCorrectRawBundle())
			So(err, ShouldBeNil)
			SoDefaultNoUserBundle(api.b.(*ScutumTestBackend))
		})
		Convey("GetFullPrivacyContext", func() {
			_, err := api.SendMevBundle(GetFullPrivacyContext(), GetCorrectRawBundle())
			So(err, ShouldBeNil)
			SoDefaultNoUserBundle(api.b.(*ScutumTestBackend))
		})
		Convey("GetMaxBackRunContext", func() {
			_, err := api.SendMevBundle(GetMaxBackRunContext(), GetCorrectRawBundle())
			So(err, ShouldBeNil)
			SoDefaultNoUserBundle(api.b.(*ScutumTestBackend))
		})
		Convey("GetMyHostContext", func() {
			_, err := api.SendMevBundle(GetMyHostContext(), GetCorrectRawBundle())
			So(err, ShouldBeNil)
			SoDefaultNoUserBundle(api.b.(*ScutumTestBackend))
		})
		Convey("GetMyIdContext", func() {
			_, err := api.SendMevBundle(GetMyIdContext(), GetCorrectRawBundle())
			So(err, ShouldBeNil)
			SoDefaultNoUserBundle(api.b.(*ScutumTestBackend))
		})
		Convey("GetErrorIdContext", func() {
			_, err := api.SendMevBundle(GetErrorIdContext(), GetCorrectRawBundle())
			So(err, ShouldBeNil)
			SoDefaultNoUserBundle(api.b.(*ScutumTestBackend))
		})
		Convey("GetErrorHostContext", func() {
			_, err := api.SendMevBundle(GetErrorHostContext(), GetCorrectRawBundle())
			So(err, ShouldBeNil)
			SoDefaultNoUserBundle(api.b.(*ScutumTestBackend))
		})
		Convey("GetOverHostContext", func() {
			_, err := api.SendMevBundle(GetOverHostContext(), GetCorrectRawBundle())
			So(err, ShouldNotBeNil)
		})
		Convey("GetLessHostContext", func() {
			_, err := api.SendMevBundle(GetLessHostContext(), GetCorrectRawBundle())
			So(err, ShouldNotBeNil)
		})
	})

}
