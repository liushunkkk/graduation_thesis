package core

import (
	"bsc-rpc-client/client"
	"bsc-rpc-client/model"
	"fmt"
	"github.com/ethereum/go-ethereum/core/types"
	"time"
)

type User struct {
	id      int
	rpc     *client.JSONRPCClient
	txCount int
}

func NewUser(id int, rpcURL string) *User {
	return &User{
		id:  id,
		rpc: client.NewJSONRPCClient(rpcURL),
	}
}

func (u *User) Start() {
	ticker := time.NewTicker(20 * time.Millisecond)
	defer ticker.Stop()

	for range ticker.C {
		u.sendTransaction()
	}
}

func (u *User) sendTransaction() {
	next := GlobalTxLoader.Next()
	// 模拟交易数据
	tx := types.NewTx(nil)
	_ = tx.UnmarshalJSON([]byte(next.OriginJsonString))

	input, _ := tx.MarshalBinary()

	args := &model.SendRawTransactionArgs{
		Input:          input,
		MaxBlockNumber: 1,
	}

	resp, err := u.rpc.SendRawTransaction(args)
	if err != nil {
		fmt.Printf("User %d 发送交易失败: %v\n", u.id, err)
		return
	}

	u.txCount++
	if resp != nil {
		fmt.Printf("User %d 发送交易成功 #%d, 返回TxHash: %s\n", u.id, u.txCount, resp.TxHash.Hex())
	}
}

func InitUser() {
	rpcURL := "http://localhost:8080"
	userCount := 1 // 可以模拟多个用户
	for i := 1; i <= userCount; i++ {
		user := NewUser(i, rpcURL)
		go user.Start()
	}
}
