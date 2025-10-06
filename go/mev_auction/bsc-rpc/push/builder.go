package push

import (
	"context"
	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/common/ms"
	"github.com/ethereum/go-ethereum/core/types"
	"github.com/ethereum/go-ethereum/ethclient"
	"github.com/ethereum/go-ethereum/log"
	. "github.com/ethereum/go-ethereum/log/zap"
	"github.com/ethereum/go-ethereum/portal"
	"github.com/ethereum/go-ethereum/push/blockrazor"
	"github.com/ethereum/go-ethereum/push/bloxroute"
	"github.com/ethereum/go-ethereum/push/club48"
	"github.com/ethereum/go-ethereum/push/define"
	"github.com/ethereum/go-ethereum/push/nodereal"
	"github.com/ethereum/go-ethereum/push/smith"
	"go.uber.org/zap"
	"strings"
	"time"
)

var PublicNodes []string

var Builders = map[string]Builder{
	"blockrazor": blockrazor.NewBlockRazor(),
	"48club":     club48.NewClub48(),
	"smith":      smith.NewSmith(),
	"bloxroute":  bloxroute.NewBloxRoute(),
	"nodereal":   nodereal.NewNodeReal(),
}

type Builder interface {
	SendBundle(param define.Param, hash common.Hash)
	GetPublicAddress() common.Address
	SendRawPrivateTransaction(tx string, bundleHash common.Hash)
}

type BuilderServer struct {
	servers map[string]*ms.Server
}

func NewBuilderServer() *BuilderServer {
	bs := &BuilderServer{servers: make(map[string]*ms.Server)}
	for n, b := range Builders {
		builer := b
		name := n
		s, _ := ms.NewSvr(name, func(ctx context.Context, msg interface{}, num int) (interface{}, error) {
			builderParam := msg.(*define.BuilderParam)

			if (name == "blockrazor" && builderParam.Counter == 0) || (name == "48club" && builderParam.Counter == 0) {
				go builer.SendRawPrivateTransaction(builderParam.Param.Txs[0], builderParam.BundleHash)
			}
			if (name != "blockrazor") || (name == "blockrazor" && builderParam.Counter != 0) {
				go builer.SendBundle(*builderParam.Param, builderParam.BundleHash)
			}
			return nil, nil
		}, nil)
		s.ActionGoroutineNum = 5
		bs.servers[name] = s
	}

	return bs
}

func (bs *BuilderServer) Start() {
	for _, s := range bs.servers {
		s.Go()
	}
}

func (bs *BuilderServer) Stop() {
	for _, s := range bs.servers {
		s.Stop()
	}
}

// Send header为当前已出的最新块
func (bs *BuilderServer) Send(header *types.Header, bundle *types.Bundle, createNumber uint64) {
	builderParam, portalBundleData := bundle.GenBuilderReq(header)

	//validatorNextIs48 := validator.Server.NextBlockIs48Club(header.Number.Int64())
	//validatorAfterNextIs48 := validator.Server.NextBlockIs48Club(header.Number.Int64() + 1) // the block after next is club48.

	var builderList []string
	if header.Number.Uint64()-createNumber < uint64(bundle.PrivacyPeriod) {
		// PrivacyBuilder
		builderList = bundle.PrivacyBuilder
	} else {
		// BroadcastBuilder
		builderList = bundle.BroadcastBuilder
	}

	if header.Number.Uint64()-createNumber >= uint64(bundle.PrivacyPeriod) && bundle.Erc20Tx {
		for _, addr := range PublicNodes {
			go sendToPublicNode(bundle, addr)
		}
	}

	for _, builder := range builderList {
		builder = strings.ToLower(builder)
		if _, ok := Builders[strings.ToLower(builder)]; !ok {
			log.Error("builder not exist")
			continue
		}

		b := bs.servers[builder]
		if b != nil {
			req := *builderParam
			b.PushMsgToServer(context.Background(), &define.BuilderParam{
				Param:      &req,
				BundleHash: bundle.Hash(),
				Counter:    bundle.Counter,
			})
		}
	}

	// send portal data to portal
	if portalBundleData != nil {
		//fmt.Println("called portal.BundleSaver.Send")
		portal.BundleSaver.Send(portalBundleData)
	}
}

func sendToPublicNode(bundle *types.Bundle, addr string) {
	dial, err := ethclient.Dial(addr)
	if err != nil {
		return
	}
	ctx, _ := context.WithTimeout(context.Background(), 1*time.Second)
	err = dial.SendTransaction(ctx, bundle.Txs[0])
	if err != nil {
		Zap.Error("failed to send raw tx: ", zap.Any("err", err.Error()), zap.Any("addr", addr))
	} else {
		Zap.Info("send raw tx to public node, tx hash is ", zap.Any("bundleHash", bundle.Hash()), zap.Any("addr", addr))
	}
}
