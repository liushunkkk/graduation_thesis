package main

import (
	"context"
	"errors"
	"fmt"
	"github.com/ethereum/go-ethereum-test/bundlepool"
	"github.com/ethereum/go-ethereum-test/jsonrpcserver"
	"github.com/ethereum/go-ethereum-test/mev_api"
	"github.com/ethereum/go-ethereum-test/push"
	"github.com/ethereum/go-ethereum-test/zap_logger"
	"go.uber.org/zap"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"
)

const (
	SendMevBundleEndpointName      = "eth_sendMevBundle"
	SendRawTransactionEndpointName = "eth_sendRawTransaction"
	ResetHeaderEndpointName        = "eth_resetHeader"
)

func main() {
	if fileExists("./log/rpc.log") {
		err1 := os.Remove("./log/rpc.log")
		if err1 != nil {
			panic(err1)
		}
	}
	ctx, ctxCancel := context.WithCancel(context.Background())
	pushServer := &push.SSEServer{IPLimitCount: 100} // 相当于没限制
	pool := bundlepool.New(pushServer, nil)
	api := mev_api.NewAPI(pool)
	jsonRPCServer, err := jsonrpcserver.NewHandler(jsonrpcserver.Methods{
		SendMevBundleEndpointName:      api.SendMevBundle,
		SendRawTransactionEndpointName: api.SendRawTransaction,
		ResetHeaderEndpointName:        api.ResetHeader,
	})
	if err != nil {
		zap_logger.Zap.Fatal("Failed to create jsonrpc server", zap.Error(err))
	}

	http.Handle("/", jsonRPCServer)
	http.Handle("/stream", pushServer)
	server := &http.Server{
		Addr:              fmt.Sprintf(":%s", "8080"),
		ReadHeaderTimeout: 5 * time.Second,
	}

	connectionsClosed := make(chan struct{})
	go func() {
		notifier := make(chan os.Signal, 1)
		signal.Notify(notifier, os.Interrupt, syscall.SIGTERM)
		<-notifier
		zap_logger.Zap.Info("Shutting down...")
		ctxCancel()
		// 优雅关闭服务
		if err := server.Shutdown(context.Background()); err != nil {
			zap_logger.Zap.Error("Failed to shutdown server", zap.Error(err))
		}
		close(connectionsClosed)
	}()

	pushServer.Start()

	go func() {
		pool.PrintState()
	}()

	// 启动
	err = server.ListenAndServe()
	if err != nil && !errors.Is(err, http.ErrServerClosed) {
		zap_logger.Zap.Fatal("ListenAndServe: ", zap.Error(err))
	}

	<-ctx.Done()
	<-connectionsClosed
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
