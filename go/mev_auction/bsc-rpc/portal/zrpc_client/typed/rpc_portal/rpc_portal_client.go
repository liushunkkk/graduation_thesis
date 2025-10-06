package rpc_portal

import (
	"github.com/zeromicro/go-zero/zrpc"

	"github.com/ethereum/go-ethereum/portal/zrpc_client/typed/rpc_portal/rpcs"

	"github.com/ethereum/go-ethereum/portal/zrpc_client/typed/rpc_portal/tx"

	"github.com/ethereum/go-ethereum/portal/zrpc_client/typed/rpc_portal/user"

	"github.com/ethereum/go-ethereum/portal/zrpc_client/typed/rpc_portal/rpcv2"

	"github.com/ethereum/go-ethereum/portal/zrpc_client/typed/rpc_portal/txv2"

	"github.com/ethereum/go-ethereum/portal/zrpc_client/typed/rpc_portal/userv2"
)

type Interface interface {
	Rpcs() rpcs.Rpcs

	Tx() tx.Tx

	User() user.User

	Rpcv2() rpcv2.Rpcv2

	Txv2() txv2.Txv2

	UserV2() userv2.UserV2
}

type Client struct {
	client zrpc.Client
}

func New(c zrpc.Client) *Client {
	return &Client{client: c}
}

func (x *Client) Rpcs() rpcs.Rpcs {
	return rpcs.NewRpcs(x.client)
}

func (x *Client) Tx() tx.Tx {
	return tx.NewTx(x.client)
}

func (x *Client) User() user.User {
	return user.NewUser(x.client)
}

func (x *Client) Rpcv2() rpcv2.Rpcv2 {
	return rpcv2.NewRpcv2(x.client)
}

func (x *Client) Txv2() txv2.Txv2 {
	return txv2.NewTxv2(x.client)
}

func (x *Client) UserV2() userv2.UserV2 {
	return userv2.NewUserV2(x.client)
}
