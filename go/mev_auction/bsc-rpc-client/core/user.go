package core

import (
	"bsc-rpc-client/client"
	"bsc-rpc-client/model"
	"bsc-rpc-client/zap_logger"
	"context"
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
	ticker  int
}

func NewUser(id int, rpcURL string, ticker int) *User {
	return &User{
		id:     id,
		rpc:    client.NewJSONRPCClient(rpcURL),
		ticker: ticker,
	}
}

func (u *User) Start(ctx context.Context) {
	ticker := time.NewTicker(time.Duration(u.ticker-5) * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			time.Sleep(10 * time.Millisecond)
			u.sendTransaction()
		}
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
		UserId:         u.id,
		MaxBlockNumber: GlobalBlockManager.GetCurrentBlock() + 1,
	}

	zap_logger.Zap.Info(fmt.Sprintf("[User-%d] 开始发送交易 #%d", u.id, u.txCount), zap.Any("sendTime", time.Now().UnixMicro()), zap.Any("txHash", tx.Hash().Hex()))
	resp, err := u.rpc.SendRawTransaction(args)
	if err != nil {
		zap_logger.Zap.Info(fmt.Sprintf("[User-%d] 发送交易失败: %v", u.id, err))
		return
	}

	u.txCount++
	if resp != nil {
		zap_logger.Zap.Info(fmt.Sprintf("[User-%d] 发送交易成功 #%d", u.id, u.txCount), zap.Any("sendTime", time.Now().UnixMicro()), zap.Any("txHash", tx.Hash().Hex()), zap.Any("respTxHash", resp.TxHash.Hex()))
	}
}

func InitUser(ctx context.Context, num int, highStream bool) {
	rpcURL := "http://localhost:8080"
	for i := 1; i <= num; i++ {
		var user *User
		if highStream && i == 1 {
			user = NewUser(i, rpcURL, i*20)
		} else {
			user = NewUser(i, rpcURL, i*50)
		}
		go user.Start(ctx)
	}
}
