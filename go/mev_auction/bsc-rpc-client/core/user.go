package core

import (
	"bsc-rpc-client/client"
	"bsc-rpc-client/model"
	"bsc-rpc-client/zap_logger"
	"fmt"
	"github.com/ethereum/go-ethereum/common/hexutil"
	"github.com/ethereum/go-ethereum/core/types"
	"go.uber.org/zap"
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
	next := UserTxLoader.Next()
	if next == nil {
		return
	}
	// 模拟交易数据
	tx := new(types.Transaction)
	err := tx.UnmarshalJSON(hexutil.Bytes(next.OriginJsonString))
	if err != nil {
		return
	}

	input, _ := tx.MarshalBinary()

	args := &model.SendRawTransactionArgs{
		Input:          input,
		MaxBlockNumber: GlobalBlockManager.GetCurrentBlock() + 1,
	}

	resp, err := u.rpc.SendRawTransaction(args)
	if err != nil {
		zap_logger.Zap.Info(fmt.Sprintf("[User-%d] 发送交易失败: %v", u.id, err))
		return
	}

	u.txCount++
	if resp != nil {
		zap_logger.Zap.Info(fmt.Sprintf("[User-%d] 发送交易成功 #%d", u.id, u.txCount), zap.Any("sendTime", time.Now().UnixNano()), zap.Any("txHash", tx.Hash().Hex()), zap.Any("respTxHash", resp.TxHash.Hex()))
	}
}

func InitUser(num int) {
	rpcURL := "http://localhost:8080"
	for i := 1; i <= num; i++ {
		user := NewUser(i, rpcURL)
		go user.Start()
	}
}
