package bundlepool

import (
	"context"
	"crypto/ecdsa"
	"fmt"
	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/core/types"
	"github.com/ethereum/go-ethereum/crypto"
	"github.com/ethereum/go-ethereum/ethclient"
	"log"
	"math/big"
	"strings"
	"testing"
	"time"
)

// var sa = "https://bsc.blockrazor.xyz"
var sa = "http://34.226.211.254:8545"
var testClientRate *ethclient.Client
var bnbClient *ethclient.Client

type TxTime struct {
	TxHash                           string
	SendTxTime                       time.Time
	SendTxSuccessTime                time.Time
	GetTransactionReceiptTime        time.Time
	GetTransactionReceiptSuccessTime time.Time
	BlockTime                        time.Time

	// 时间差（以毫秒为单位）
	SendTxToSendTxSuccessMs            int64
	GetReceiptToGetReceiptSuccessMs    int64
	SendTxSuccessToBlockTimeMs         int64
	BlockTimeToGetReceiptSuccessTimeMs int64
}

var TxTimeList []*TxTime

// 计算时间差（以毫秒为单位）
func (t *TxTime) CalculateDurations() {
	t.SendTxToSendTxSuccessMs = t.SendTxSuccessTime.Sub(t.SendTxTime).Milliseconds()
	t.GetReceiptToGetReceiptSuccessMs = t.GetTransactionReceiptSuccessTime.Sub(t.GetTransactionReceiptTime).Milliseconds()
	t.SendTxSuccessToBlockTimeMs = t.BlockTime.Sub(t.SendTxSuccessTime).Milliseconds()
	t.BlockTimeToGetReceiptSuccessTimeMs = t.GetTransactionReceiptSuccessTime.Sub(t.BlockTime).Milliseconds()
}

func init() {
	client, err := ethclient.Dial(sa)
	if err != nil {
		log.Fatal(err)
	}
	testClientRate = client

	bnbClient, _ = ethclient.Dial("https://bsc-dataseed.bnbchain.org")
}

func sendOneTx() {
	privateKey, err := crypto.HexToECDSA("0ad182d90bff7b643f70a7a8724d3c4f3a3cdca6711eef1301321695b984b36d")
	if err != nil {
		log.Fatal(err)
	}
	publicKey := privateKey.Public()
	publicKeyECDSA, ok := publicKey.(*ecdsa.PublicKey)
	if !ok {
		log.Fatal("cannot assert type: publicKey is not of type *ecdsa.PublicKey")
	}

	fromAddress := crypto.PubkeyToAddress(*publicKeyECDSA)
	nonce, err := testClientRate.PendingNonceAt(context.Background(), fromAddress)
	if err != nil {
		log.Fatal(err)
	}
	value := big.NewInt(1e4)
	gasLimit := uint64(21_000) // in units
	gasPrice := big.NewInt(1e9)
	toAddress := common.HexToAddress("0x43DdA9d1Ac023bd3593Dff5A1A677247Bb98fE11")

	tx := types.NewTx(&types.LegacyTx{
		Nonce:    nonce,
		GasPrice: gasPrice,
		Gas:      gasLimit,
		To:       &toAddress,
		Value:    value,
		Data:     nil,
	})

	signedTx, err := types.SignTx(tx, types.LatestSignerForChainID(big.NewInt(56)), privateKey)
	if err != nil {
		log.Fatal(err)
	}

	txTime := &TxTime{}

	txTime.TxHash = signedTx.Hash().Hex()
	fmt.Println("Tx Hash", signedTx.Hash().Hex())
	txTime.SendTxTime = time.Now()
	fmt.Println("Client.SendTransaction调用之前时间：", txTime.SendTxTime)
	err = testClientRate.SendTransaction(context.Background(), signedTx)
	txTime.SendTxSuccessTime = time.Now()
	fmt.Println("Client.SendTransaction调用之后时间：", txTime.SendTxSuccessTime)
	if err != nil {
		log.Fatal(err)
	}

	go func() {
		for {
			//transactionReceipt, err := testClientRate.TransactionReceipt(context.Background(), signedTx.Hash())
			_, _, err := bnbClient.TransactionByHash(context.Background(), signedTx.Hash())
			if err != nil {
				continue
			}
			fmt.Println("BNB官方RPC节点调用 Client.TransactionByHash 成功时间：", time.Now())
			break
		}
	}()

	for {
		txTime.GetTransactionReceiptTime = time.Now()
		//transactionReceipt, err := testClientRate.TransactionReceipt(context.Background(), signedTx.Hash())
		_, _, err := testClientRate.TransactionByHash(context.Background(), signedTx.Hash())
		if err != nil {
			continue
		}
		txTime.GetTransactionReceiptSuccessTime = time.Now()
		fmt.Println("scutum 调用 Client.TransactionByHash 成功时间：", txTime.GetTransactionReceiptSuccessTime)
		for {
			transactionReceipt, err := testClientRate.TransactionReceipt(context.Background(), signedTx.Hash())
			if err != nil {
				continue
			}
			blockByNumber, err := testClientRate.BlockByNumber(context.Background(), transactionReceipt.BlockNumber)
			if err != nil {
				continue
			}
			txTime.BlockTime = time.Unix(int64(blockByNumber.Time()), 0)
			fmt.Println("该交易出块时间：", txTime.BlockTime)
			break
		}
		break
	}

	time.Sleep(time.Second)
	TxTimeList = append(TxTimeList, txTime)
}

// 表格化输出 TxTimeList
func PrintTxTimeTable(txTimeList []*TxTime) {
	// 打印表头
	fmt.Printf("%-70s %-30s %-30s %-30s %-30s %-30s %-30s %-30s %-30s %-30s\n",
		"TxHash", "SendTxTime", "SendTxSuccessTime", "GetReceiptTime", "GetReceiptSuccessTime",
		"BlockTime", "eth_sendTransaction (ms)", "eth_getTransactionByHash (ms)", "交易上块 (ms)", "上块与查到收据延迟 (ms)")

	// 打印分隔线
	fmt.Println(strings.Repeat("-", 300))

	var avgSendTxMs, avgGetTxReceiptMs, avgTxSuccessMs, avgApiAndBlockMs int64

	// 打印每条记录
	for _, tx := range txTimeList {
		fmt.Printf("%-70s %-30s %-30s %-30s %-30s %-30s %-30d %-30d %-30d %-30d\n",
			tx.TxHash,
			tx.SendTxTime.String()[:25],
			tx.SendTxSuccessTime.String()[:25],
			tx.GetTransactionReceiptTime.String()[:25],
			tx.GetTransactionReceiptSuccessTime.String()[:25],
			tx.BlockTime.String()[:25],
			tx.SendTxToSendTxSuccessMs,
			tx.GetReceiptToGetReceiptSuccessMs,
			tx.SendTxSuccessToBlockTimeMs,
			tx.BlockTimeToGetReceiptSuccessTimeMs)
		avgSendTxMs += tx.SendTxToSendTxSuccessMs
		avgGetTxReceiptMs += tx.GetReceiptToGetReceiptSuccessMs
		avgTxSuccessMs += tx.SendTxSuccessToBlockTimeMs
		avgApiAndBlockMs += tx.BlockTimeToGetReceiptSuccessTimeMs
	}

	avgSendTxMs /= int64(len(txTimeList))
	avgGetTxReceiptMs /= int64(len(txTimeList))
	avgTxSuccessMs /= int64(len(txTimeList))
	avgApiAndBlockMs /= int64(len(txTimeList))

	fmt.Println("eth_sendTransaction 接口平均时间/ms：", avgSendTxMs)
	fmt.Println("eth_getTransactionByHash 接口平均时间/ms：", avgGetTxReceiptMs)
	fmt.Println("交易上块 平均时间/ms：", avgTxSuccessMs)
	fmt.Println("出块与查到收据延迟 平均时间/ms：", avgApiAndBlockMs)
}

func TestApiRate(t *testing.T) {
	for i := 0; i < 20; i++ {
		fmt.Printf("第 %v 条交易：", i+1)
		sendOneTx()
	}
	//testClientRate.TransactionByHash()
	//fmt.Println(TxTimeList)
	for _, txTime := range TxTimeList {
		txTime.CalculateDurations()
	}

	PrintTxTimeTable(TxTimeList)
}
