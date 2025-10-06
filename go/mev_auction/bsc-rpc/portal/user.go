package portal

import (
	"context"
	"github.com/ethereum/go-ethereum/common/ms"
	. "github.com/ethereum/go-ethereum/log/zap"
	"github.com/ethereum/go-ethereum/portal/zrpc_client"
	"github.com/ethereum/go-ethereum/portal/zrpc_client/typed/rpc_portal/rpcv2"
	"github.com/ethereum/go-ethereum/portal/zrpc_client/typed/rpc_portal/userv2"
	"github.com/zeromicro/go-zero/zrpc"
	"go.uber.org/zap"
	"google.golang.org/grpc/metadata"
	"sync"
	"time"
)

var UserServer *RpcUserServer

type RpcUserServer struct {
	server    *ms.Server
	target    zrpc.Client
	cs        *zrpc_client.Clientset
	RpcInfos  sync.Map
	UserInfos sync.Map
}

func NewRpcUserServer() *RpcUserServer {
	UserServer = &RpcUserServer{}
	target, err := zrpc.NewClientWithTarget(Address, zrpc.WithNonBlock())
	if err != nil {
		panic(err)
	}
	UserServer.target = target
	UserServer.cs = zrpc_client.NewClientWithOptions(zrpc_client.WithRpcPortalClient(target))

	UserServer.server, _ = ms.NewSvr("portal-user", nil, []ms.TimedTask{{Task: UserServer.Task, Time: 1 * time.Minute}})
	return UserServer
}

func (s *RpcUserServer) Start() {
	s.server.Go()
}

func (s *RpcUserServer) Stop() {
	s.server.Stop()
	s.target.Conn().Close()
}

func (s *RpcUserServer) Task(_ int) {
	// 创建带有自定义头部信息的客户端上下文
	md := metadata.Pairs(
		"Authorization", "Basic YWRtaW46YWRtaW4=",
	)
	// 将元数据添加到上下文中
	portalCtx := metadata.NewOutgoingContext(context.Background(), md)
	list, err := s.cs.RpcPortal().Rpcv2().GetAllRpcInfoList(portalCtx, &rpcv2.GetAllRpcInfoListRequest{ChainId: "56"})
	if err != nil {
		Zap.Error("request RpcInfoList from portal failed", zap.Error(err))
		return
	}
	Zap.Info("request RpcInfoList from portal successfully: ", zap.Any("dataLen", len(list.RpcInfoList)))
	for _, v := range list.RpcInfoList {
		if v.Url != "" {
			s.RpcInfos.Store(v.Url, v)
		}
		s.RpcInfos.Store(v.RpcId, v)
	}

	users, err := s.cs.RpcPortal().UserV2().GetAllUsers(portalCtx, &userv2.GetAllUsersRequest{PlanId: 0})
	if err != nil {
		Zap.Error("request GetAllUsers from portal failed", zap.Error(err))
		return
	}
	Zap.Info("request GetAllUsers from portal successfully: ", zap.Any("dataLen", len(users.UserList)))

	for _, v := range users.UserList {
		s.UserInfos.Store(v.AuthToken, v)
	}
}

func (s *RpcUserServer) GetAllRpcInfoList(key interface{}) *rpcv2.GetAllRpcInfoResponse {
	r, ok := s.RpcInfos.Load(key)
	if !ok {
		return nil
	}
	return r.(*rpcv2.GetAllRpcInfoResponse)
}

func (s *RpcUserServer) GetUserInfo(key interface{}) *userv2.GetUserInfoResponse {
	r, ok := s.UserInfos.Load(key)
	if !ok {
		return nil
	}
	return r.(*userv2.GetUserInfoResponse)
}
