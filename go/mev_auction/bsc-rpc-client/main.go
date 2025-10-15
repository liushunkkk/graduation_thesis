package main

import "bsc-rpc-client/core"

func main() {
	err := core.UserTxLoader.LoadFromDB(10000)
	if err != nil {
		panic(err)
	}
	err = core.SearcherTxLoader.LoadFromDB(20000)
	if err != nil {
		panic(err)
	}
	core.InitSearcher()
	core.InitUser()
	core.GlobalBlockManager.StartAutoIncrement()

	select {} // 阻塞主线程，让用户一直发送交易
}
