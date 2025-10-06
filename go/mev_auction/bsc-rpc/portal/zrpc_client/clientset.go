package zrpc_client

import (
	"github.com/ethereum/go-ethereum/portal/zrpc_client/typed/rpc_portal"
)

type Interface interface {
	RpcPortal() rpc_portal.Interface
}

type Clientset struct {
	rpcPortal *rpc_portal.Client
}

func (x *Clientset) RpcPortal() rpc_portal.Interface {
	return x.rpcPortal
}

func NewClientWithOptions(ops ...Opt) *Clientset {
	cs := &Clientset{}

	for _, op := range ops {
		op(cs)
	}

	return cs
}
