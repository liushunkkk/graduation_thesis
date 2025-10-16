package main

import (
	"bsc-rpc-client/core"
	"context"
	"os"
	"time"
)

func main() {
	if fileExists("./log/rpc.log") {
		err1 := os.Remove("./log/rpc.log")
		if err1 != nil {
			panic(err1)
		}
	}
	ctx := context.Background()
	cancelCtx, cancel := context.WithCancel(ctx)
	core.GlobalBlockManager.StartAutoIncrement()
	// 先让区块同步好，然后再开始发送交易

	time.Sleep(10 * time.Second)
	err := core.UserTxLoader.LoadFromDB(20000)
	if err != nil {
		panic(err)
	}
	err = core.SearcherTxLoader.LoadFromDB(60000)
	if err != nil {
		panic(err)
	}

	openBlock := false // 是否开启模拟搜索者阻塞
	if openBlock {
		go func() {
			time.Sleep(1 * time.Minute)
			core.GlobalIsBlock = true
		}()
	}
	highStream := true // 是否开启模拟某用户高并发
	core.InitSearcher(cancelCtx, 2, openBlock)
	core.InitUser(cancelCtx, 3, highStream)

	if highStream {
		time.Sleep(1 * time.Minute)
	} else {
		time.Sleep(3 * time.Minute)
	}

	cancel() // 停止所有用户和搜索者

	time.Sleep(10 * time.Second) // 再同步几个区块
}

func fileExists(path string) bool {
	_, err := os.Stat(path)
	if err == nil {
		return true // 文件存在
	}
	if os.IsNotExist(err) {
		return false // 文件不存在
	}
	// 其他错误，例如权限问题，也认为文件存在与否不确定
	return false
}
