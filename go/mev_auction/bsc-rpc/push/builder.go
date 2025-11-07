package push

import (
	"context"
	"github.com/ethereum/go-ethereum-test/base"
	"github.com/ethereum/go-ethereum-test/push/blockrazor"
	"github.com/ethereum/go-ethereum-test/push/bloxroute"
	"github.com/ethereum/go-ethereum-test/push/club48"
	"github.com/ethereum/go-ethereum-test/push/define"
	"github.com/ethereum/go-ethereum-test/push/nodereal"
	"github.com/ethereum/go-ethereum-test/push/smith"
	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/common/ms"
	"github.com/ethereum/go-ethereum/core/types"
	"github.com/ethereum/go-ethereum/log"
	"strings"
)

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
			go builer.SendBundle(*builderParam.Param, builderParam.BundleHash)
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
func (bs *BuilderServer) Send(header *types.Header, bundle *base.Bundle, createNumber uint64) {
	builderParam, _ := bundle.GenBuilderReq(header)

	var builderList []string
	if header.Number.Uint64()-createNumber < uint64(bundle.PrivacyPeriod) {
		builderList = bundle.PrivacyBuilder
	} else {
		builderList = bundle.BroadcastBuilder
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
}
