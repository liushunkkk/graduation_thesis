package portal

import (
	"context"
	"github.com/ethereum/go-ethereum/common/lru"
	"github.com/ethereum/go-ethereum/common/ms"
	. "github.com/ethereum/go-ethereum/log/zap"
	"github.com/ethereum/go-ethereum/portal/zrpc_client"
	"github.com/ethereum/go-ethereum/portal/zrpc_client/typed/rpc_portal/txv2"
	"github.com/zeromicro/go-zero/zrpc"
	"go.uber.org/zap"
	"google.golang.org/grpc/metadata"
	"time"
)

var BundleSaver *Saver

type Saver struct {
	server          *ms.Server
	cs              *zrpc_client.Clientset
	target          zrpc.Client
	apiHealth       []*zrpc_client.Clientset
	apiHealthTarget []zrpc.Client
	bundleCache     *lru.Cache[string, struct{}]
}

func NewSaver() *Saver {
	s := &Saver{}

	target, err := zrpc.NewClientWithTarget(Address, zrpc.WithNonBlock())
	if err != nil {
		panic(err)
	}
	s.target = target
	s.cs = zrpc_client.NewClientWithOptions(zrpc_client.WithRpcPortalClient(target))

	for _, addr := range ApiHealth {
		client, err := zrpc.NewClientWithTarget(addr, zrpc.WithNonBlock())
		if err != nil {
			panic(err)
		}
		s.apiHealthTarget = append(s.apiHealthTarget, client)
		s.apiHealth = append(s.apiHealth, zrpc_client.NewClientWithOptions(zrpc_client.WithRpcPortalClient(client)))
	}

	s.server, _ = ms.NewSvr("portal-saver", s.MsgSender, nil)
	s.server.ActionGoroutineNum = 20

	s.bundleCache = lru.NewCache[string, struct{}](10000)

	BundleSaver = s
	return s
}

func (s *Saver) Start() {
	s.server.Go()
}

func (s *Saver) Stop() {
	s.server.Stop()
	s.target.Conn().Close()

	for _, target := range s.apiHealthTarget {
		target.Conn().Close()
	}
}

func (s *Saver) MsgSender(ctx context.Context, msg interface{}, num int) (resp interface{}, err error) {
	go func() {
		sr := msg.(*txv2.BundleSaveRequest)
		if s.bundleCache.Contains(sr.BundleHash) {
			return
		}
		// 创建带有自定义头部信息的客户端上下文
		md := metadata.Pairs(
			"Authorization", "Basic YWRtaW46YWRtaW4=",
		)
		// 将元数据添加到上下文中
		portalCtx := metadata.NewOutgoingContext(context.Background(), md)
		_, err = s.cs.RpcPortal().Txv2().BundleSave(portalCtx, sr)
		if err != nil {
			Zap.Error("save bundle to portal failed: ", zap.Any("data", sr))
			return
		}
		Zap.Info("save bundle to portal successfully: ", zap.Any("bundleHash", sr.BundleHash), zap.Any("parentHash", sr.ParentHash), zap.Any("data", sr))

		for i, addr := range s.apiHealth {
			ctxTimeout, _ := context.WithTimeout(context.Background(), 1*time.Second)
			_, err = addr.RpcPortal().Txv2().BundleSave(ctxTimeout, sr)
			if err != nil {
				Zap.Error("save bundle to apiHealth failed: ", zap.Any("addr", ApiHealth[i]), zap.Any("data", sr))
				continue
			}
			Zap.Info("save bundle to apiHealth successfully: ", zap.Any("addr", ApiHealth[i]), zap.Any("bundleHash", sr.BundleHash), zap.Any("parentHash", sr.ParentHash))
		}

		s.bundleCache.Add(sr.BundleHash, struct{}{})
	}()
	return nil, nil
}

func (s *Saver) Send(msg interface{}) {
	s.server.PushMsgToServer(context.Background(), msg)
}
