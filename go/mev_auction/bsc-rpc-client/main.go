package main

import "bsc-rpc-client/core"

func main() {
	core.InitSearcher()
	core.InitUser()
	core.GlobalBlockManager.StartAutoIncrement()
	err := core.GlobalTxLoader.LoadFromDB(50000)
	if err != nil {
		panic(err)
	}

	select {} // 阻塞主线程，让用户一直发送交易
}
