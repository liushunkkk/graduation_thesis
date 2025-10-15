package main

import (
	"bsc-rpc-client/core"
	"time"
)

func main() {
	core.GlobalBlockManager.StartAutoIncrement()
	// 先让区块同步好，然后再开始发送交易

	time.Sleep(10 * time.Second)
	err := core.UserTxLoader.LoadFromDB(10000)
	if err != nil {
		panic(err)
	}
	err = core.SearcherTxLoader.LoadFromDB(20000)
	if err != nil {
		panic(err)
	}
	core.InitSearcher(1)
	core.InitUser(1)

	time.Sleep(1 * time.Minute)
}
