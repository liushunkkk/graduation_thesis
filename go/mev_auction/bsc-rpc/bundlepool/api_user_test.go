package bundlepool

import (
	"context"
	"fmt"
	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/ethclient"
	"github.com/ethereum/go-ethereum/portal/zrpc_client/model/rpc_portal/pb/rpcpb"
	"log"
	"testing"
)

// 设置用户的配置信息
func TestApi_PushRpcInfo(t *testing.T) {
	testClient, err := ethclient.Dial("https://bsc.rpc.blockrazor.io")

	if err != nil {
		log.Fatal(err)
	}
	args := &rpcpb.GetRpcInfoResponse{
		RpcId:                "1827949375179984896",
		ChainId:              "56",
		Url:                  "beanz",
		HintHash:             false,
		HintFrom:             true,
		HintTo:               false,
		HintValue:            false,
		HintNonce:            false,
		HintCalldata:         true,
		HintFunctionSelector: false,
		HintGasLimit:         false,
		HintGasPrice:         false,
		HintLogs:             false,
		RefundRecipient:      "tx.origin",
		RefundPercent:        90,
		IsProtected:          true,
	}
	err = testClient.PushRpcInfo(context.Background(), args)

	if err != nil {
		log.Fatal("push data error ", err)
	}
}

// metamask 获取交易收据接口
func TestApi_GetTransactionReceipt(t *testing.T) {
	testClient, err := ethclient.Dial("https://bsc.rpc.blockrazor.io/1827949375179984999")
	if err != nil {
		log.Fatal(err)
	}

	h := common.HexToHash("0x0177c46e1c94d36aa74b67d52230086db4841b87ae31d9113d4dc543493e3bee")

	transaction, err := testClient.TransactionReceipt(context.Background(), h)
	if err != nil {
		log.Fatal("GetTransactionReceipt error ", err)
	}

	fmt.Println(transaction)
}

// protal 获取交易信息接口
func TestApi_GetRpcTransaction(t *testing.T) {
	testClient, err := ethclient.Dial("https://bsc.rpc.blockrazor.io/1827949375179984896")
	if err != nil {
		log.Fatal(err)
	}

	h := common.HexToHash("0x0177c46e1c94d36aa74b67d52230086db4841b87ae31d9113d4dc543493e3bee")

	transaction, err := testClient.GetRpcTransaction(context.Background(), h)
	if err != nil {
		log.Fatal("GetRpcTransaction error ", err)
	}

	fmt.Println(transaction)
}

// portal 跟踪交易
func TestApi_TraceTransaction(t *testing.T) {
	testClient, err := ethclient.Dial("https://bsc.rpc.blockrazor.io/")
	if err != nil {
		log.Fatal(err)
	}

	h := common.HexToHash("0x48960e7466c37b404ae0fadfe0d5654887b6710ff9bb7185fffd4b16ade130b9")

	transaction, err := testClient.TraceTransaction(context.Background(), h, common.Address{})
	if err != nil {
		log.Fatal("GetRpcTransaction error ", err)
	}

	fmt.Println(transaction)
}
