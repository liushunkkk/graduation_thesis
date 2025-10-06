package zrpc_client

import (
	"github.com/zeromicro/go-zero/zrpc"

	"github.com/ethereum/go-ethereum/portal/zrpc_client/typed/rpc_portal"
)

type Opt func(client *Clientset)

func WithRpcPortalClient(cli zrpc.Client) Opt {
	return func(client *Clientset) {
		client.rpcPortal = rpc_portal.New(cli)
	}
}
