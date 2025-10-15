package main

import (
	"bsc-rpc-client/core"
	"context"
	"os"
	"time"
)

func main() {
	err1 := os.Remove("./log/rpc.log")
	if err1 != nil {
		panic(err1)
	}
	ctx := context.Background()
	cancelCtx, cancel := context.WithCancel(ctx)
	core.GlobalBlockManager.StartAutoIncrement()
	// 先让区块同步好，然后再开始发送交易

	time.Sleep(10 * time.Second)
	err := core.UserTxLoader.LoadFromDB(10000)
	if err != nil {
		panic(err)
	}
	err = core.SearcherTxLoader.LoadFromDB(30000)
	if err != nil {
		panic(err)
	}
	isBlock := true // 是否开启模拟搜索者阻塞
	core.InitSearcher(cancelCtx, 1, isBlock)
	core.InitUser(cancelCtx, 1)

	time.Sleep(1 * time.Minute)

	cancel() // 停止所有用户和搜索者

	time.Sleep(10 * time.Second) // 再同步几个区块
}
