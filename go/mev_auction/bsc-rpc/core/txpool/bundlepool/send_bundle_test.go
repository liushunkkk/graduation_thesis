package bundlepool

import (
	"bufio"
	"bytes"
	"context"
	"crypto/ecdsa"
	"fmt"
	"github.com/duke-git/lancet/v2/strutil"
	"github.com/ethereum/go-ethereum/accounts/abi"
	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/common/hexutil"
	"github.com/ethereum/go-ethereum/core/types"
	"github.com/ethereum/go-ethereum/crypto"
	"github.com/ethereum/go-ethereum/ethclient"
	"github.com/ethereum/go-ethereum/params"
	"github.com/ethereum/go-ethereum/push/define"
	"github.com/goccy/go-json"
	"golang.org/x/crypto/sha3"
	"log"
	"math/big"
	"net/http"
	"strings"
	"sync"
	"testing"
	"time"
)

// 用于构造 bundle 时的数据
var nonce1 = 137 // nonce for searcher01
var nonce2 = 13  // nonce for searcher02

func TestSendMultiNonceHighRawTx(t *testing.T) {
	testClient, err := ethclient.Dial("http://34.226.211.254:8545")
	if err != nil {
		log.Fatal(err)
	}

	// 原始用户
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
	// 自动获取当前的 nouce，也就是该用户当前交易数 + 1
	nonce, err := testClient.PendingNonceAt(context.Background(), fromAddress)
	if err != nil {
		log.Fatal(err)
	}
	value := big.NewInt(1 * 1e4)
	gasLimit := uint64(30_000) // in units
	gasPrice := big.NewInt(1e9)
	// searcher01
	toAddress := common.HexToAddress("0x9Abae1b279A4Be25AEaE49a33e807cDd3cCFFa0C")

	for i := 0; i < 60; i++ {
		newNonce := nonce + uint64(i+10)
		tx := types.NewTx(&types.LegacyTx{
			Nonce:    newNonce,
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

		err = testClient.SendTransaction(context.Background(), signedTx)
		fmt.Println("nonce:", newNonce, "tx hash:", signedTx.Hash().Hex())
		if err != nil {
			log.Fatal(err)
		}
	}
}

func TestSendNonceHighBeforeNonceCorrectRawTx(t *testing.T) {
	testClient, err := ethclient.Dial("http://34.226.211.254:8545")
	if err != nil {
		log.Fatal(err)
	}
	privateKey, err := crypto.HexToECDSA("0ad182d90bff7b643f70a7a8724d3c4f3a3cdca6711eef1301321695b984b36d") // s1
	if err != nil {
		log.Fatal(err)
	}
	publicKey := privateKey.Public()
	publicKeyECDSA, ok := publicKey.(*ecdsa.PublicKey)
	if !ok {
		log.Fatal("cannot assert type: publicKey is not of type *ecdsa.PublicKey")
	}
	fromAddress := crypto.PubkeyToAddress(*publicKeyECDSA)
	// 自动获取当前的 nouce，也就是该用户当前交易数 + 1
	nonce, err := testClient.PendingNonceAt(context.Background(), fromAddress)
	if err != nil {
		log.Fatal(err)
	}
	value := big.NewInt(1 * 1e4)
	gasLimit := uint64(30_000) // in units
	gasPrice := big.NewInt(1e9)
	// S01 0x9Abae1b279A4Be25AEaE49a33e807cDd3cCFFa0C
	toAddress := common.HexToAddress("0x456CDD3291E678C90fB1cbD9aa2588a5ee3D8280") // searcher02
	nonce += 1
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
	err = testClient.SendTransaction(context.Background(), signedTx)
	fmt.Println("nonce:", nonce, "tx hash:", signedTx.Hash().Hex())
	if err != nil {
		log.Fatal(err)
	}

	time.Sleep(30 * time.Second)

	nonce -= 1
	tx1 := types.NewTx(&types.LegacyTx{
		Nonce:    nonce,
		GasPrice: gasPrice,
		Gas:      gasLimit,
		To:       &toAddress,
		Value:    value,
		Data:     nil,
	})
	signedTx1, err := types.SignTx(tx1, types.LatestSignerForChainID(big.NewInt(56)), privateKey)
	if err != nil {
		log.Fatal(err)
	}
	err = testClient.SendTransaction(context.Background(), signedTx1)
	fmt.Println("nonce:", nonce, "tx hash:", signedTx1.Hash().Hex())
	if err != nil {
		log.Fatal(err)
	}
}

func TestSendNonceTooLowAndFeeLowRawTx(t *testing.T) {
	testClient, err := ethclient.Dial("http://34.226.211.254:8545")
	if err != nil {
		log.Fatal(err)
	}

	// 原始用户
	privateKey, err := crypto.HexToECDSA("504303443a7d10c1544a45acfa19cb1c9497995013bfed3c9ac8689bf4a2421e")
	if err != nil {
		log.Fatal(err)
	}
	publicKey := privateKey.Public()
	publicKeyECDSA, ok := publicKey.(*ecdsa.PublicKey)
	if !ok {
		log.Fatal("cannot assert type: publicKey is not of type *ecdsa.PublicKey")
	}

	fromAddress := crypto.PubkeyToAddress(*publicKeyECDSA)
	// 自动获取当前的 nouce，也就是该用户当前交易数 + 1
	nonce, err := testClient.PendingNonceAt(context.Background(), fromAddress)
	if err != nil {
		log.Fatal(err)
	}
	nonce -= 10
	value := big.NewInt(1 * 1e4)
	gasLimit := uint64(30_000) // in units
	gasPrice := big.NewInt(1e9)
	// searcher01
	toAddress := common.HexToAddress("0x9Abae1b279A4Be25AEaE49a33e807cDd3cCFFa0C")

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

	err = testClient.SendTransaction(context.Background(), signedTx)
	fmt.Println("nonce:", nonce, "tx hash:", signedTx.Hash().Hex())
	if err != nil {
		log.Fatal(err)
	}
}

func TestGetMevBundleJson(t *testing.T) {
	tx := types.NewTx(&types.LegacyTx{})
	privateKey, err := crypto.HexToECDSA("0ad182d90bff7b643f70a7a8724d3c4f3a3cdca6711eef1301321695b984b36d")
	if err != nil {
		log.Fatal(err)
	}
	signedTx, err := types.SignTx(tx, types.LatestSignerForChainID(big.NewInt(56)), privateKey)
	if err != nil {
		log.Fatal(err)
	}
	b := new(bytes.Buffer)
	err = signedTx.EncodeRLP(b)
	if err != nil {
		log.Fatal(err)
	}

	hint := map[string]bool{
		types.HintTo:               true,
		types.HintValue:            true,
		types.HintNonce:            true,
		types.HintFrom:             true,
		types.HintCallData:         true,
		types.HintFunctionSelector: true,
		types.HintHash:             true,
		types.HintGasPrice:         true,
		types.HintGasLimit:         true,
		types.HintLogs:             true,
	}

	bundle := &types.SendMevBundleArgs{
		Hint:              hint,
		Txs:               []hexutil.Bytes{b.Bytes()},
		Hash:              common.Hash{},
		RevertingTxHashes: []common.Hash{common.Hash{}},
		RefundPercent:     90,
		RefundAddress:     common.HexToAddress("0x9Abae1b279A4Be25AEaE49a33e807cDd3cCFFa0C"),
	}
	marshal, err := json.Marshal(bundle)
	if err != nil {
		return
	}
	fmt.Println(strutil.BytesToString(marshal))
}

// 发送一个用户原始交易
func TestSendRawTx(t *testing.T) {
	//testClient, err := ethclient.Dial("https://bsc.blockrazor.xyz/1827949375179984896?RefundPercent=85")
	//testClient, err := ethclient.Dial("https://bsc.blockrazor.xyz/1827949375179984896")
	//testClient, err := ethclient.Dial("https://bsc.blockrazor.xyz/fullprivacy")
	//testClient, err := ethclient.Dial("https://bsc.blockrazor.xyz")
	//testClient, err := ethclient.Dial("http://34.226.211.254:8545/1845760505365401600")
	testClient, err := ethclient.Dial("http://34.226.211.254:8545?RefundPercent=99")
	if err != nil {
		log.Fatal(err)
	}

	// 原始用户
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
	// 自动获取当前的 nouce，也就是该用户当前交易数 + 1
	nonce, err := testClient.PendingNonceAt(context.Background(), fromAddress)
	if err != nil {
		log.Fatal(err)
	}
	value := big.NewInt(1e4)
	gasLimit := uint64(21_000) // in units
	gasPrice := big.NewInt(1e9)
	// searcher01
	toAddress := common.HexToAddress("0x43DdA9d1Ac023bd3593Dff5A1A677247Bb98fE11")

	// New transaction 新的 api
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

	err = testClient.SendTransaction(context.Background(), signedTx)
	fmt.Println("tx hash ", signedTx.Hash().Hex())
	if err != nil {
		log.Fatal(err)
	}

}

func TestSendRawBundle(t *testing.T) {
	//testClient, err := ethclient.Dial("https://bsc.blockrazor.xyz")
	testClient, err := ethclient.Dial("http://34.226.211.254:8545")
	if err != nil {
		log.Fatal(err)
	}

	// 原始用户
	privateKey, err := crypto.HexToECDSA("13761d94baacad374f46d11c1a6b2c1a1b4cb800c2b647b4a0da468a2929a92f")
	if err != nil {
		log.Fatal(err)
	}
	publicKey := privateKey.Public()
	publicKeyECDSA, ok := publicKey.(*ecdsa.PublicKey)
	if !ok {
		log.Fatal("cannot assert type: publicKey is not of type *ecdsa.PublicKey")
	}

	fromAddress := crypto.PubkeyToAddress(*publicKeyECDSA)
	// 自动获取当前的 nouce，也就是该用户当前交易数 + 1
	nonce, err := testClient.PendingNonceAt(context.Background(), fromAddress)
	if err != nil {
		log.Fatal(err)
	}
	value := big.NewInt(0)
	gasLimit := uint64(21_000) // in units
	gasPrice := big.NewInt(1e9)
	// searcher01
	toAddress := common.HexToAddress("0x9Abae1b279A4Be25AEaE49a33e807cDd3cCFFa0C")
	// New transaction 新的 api
	tx1 := types.NewTx(&types.LegacyTx{
		Nonce:    nonce,
		GasPrice: gasPrice,
		Gas:      gasLimit,
		To:       &toAddress,
		Value:    value,
		Data:     nil,
	})
	signedTx1, err := types.SignTx(tx1, types.LatestSignerForChainID(big.NewInt(56)), privateKey)
	if err != nil {
		log.Fatal(err)
	}
	b1, err := signedTx1.MarshalBinary()
	if err != nil {
		log.Fatal(err)
	}

	// New transaction 新的 api
	tx2 := types.NewTx(&types.LegacyTx{
		Nonce:    nonce + 1,
		GasPrice: gasPrice,
		Gas:      gasLimit,
		To:       &toAddress,
		Value:    value,
		Data:     nil,
	})
	signedTx2, err := types.SignTx(tx2, types.LatestSignerForChainID(big.NewInt(56)), privateKey)
	if err != nil {
		log.Fatal(err)
	}
	b2, err := signedTx2.MarshalBinary()
	if err != nil {
		log.Fatal(err)
	}

	hint := map[string]bool{
		types.HintTo:               true,
		types.HintValue:            true,
		types.HintNonce:            true,
		types.HintFrom:             true,
		types.HintCallData:         true,
		types.HintFunctionSelector: true,
		types.HintHash:             true,
		types.HintGasPrice:         true,
		types.HintGasLimit:         true,
		types.HintLogs:             true,
	}

	bundle := &types.SendMevBundleArgs{
		Hint:              hint,
		Txs:               []hexutil.Bytes{b1, b2},
		Hash:              common.Hash{},
		RevertingTxHashes: make([]common.Hash, 0),
		RefundPercent:     90,
		RefundAddress:     fromAddress,
	}
	bundleHash, err := testClient.SendMevBundle(context.Background(), *bundle)
	if err != nil {
		return
	}
	fmt.Println("bundleHash: ", bundleHash)
}

// 模拟 searcher 产出 mev，先订阅服务，接受 rpc 推流，然后收到推流后，追加交易，发送一次 bundle
func TestSendBundleOnce(t *testing.T) {

	// 指定用户发送，这里就是 api_user_test.go 里设置的用户
	//testClient, err := ethclient.Dial("https://bsc.blockrazor.xyz/1827949375179984896")
	testClient, err := ethclient.Dial("https://bsc.blockrazor.xyz")
	if err != nil {
		log.Fatal(err)
	}

	// 发起GET请求到SSE服务端
	resp, err := http.Get("https://bsc.blockrazor.xyz/stream")
	if err != nil {
		log.Fatalf("failed to connect to SSE server: %v", err)
	}
	defer resp.Body.Close()

	// 检查响应头是否为 "text/event-stream"
	if ct := resp.Header.Get("Content-Type"); ct != "text/event-stream" {
		log.Fatalf("unexpected Content-Type: %s", ct)
	}

	scanner := bufio.NewScanner(resp.Body)
	for scanner.Scan() {
		line := scanner.Text()
		if strings.HasPrefix(line, "data:") {
			// 处理数据行
			data := strings.TrimPrefix(line, "data: ")
			fmt.Println(time.Now())
			fmt.Printf("Received data: %s\n", data)
			sseBundleData := &define.SseBundleData{}
			json.Unmarshal([]byte(data), &sseBundleData)
			// first bundle
			bundle := getBundle1(sseBundleData)
			bundle.Hash = common.HexToHash(sseBundleData.Hash)
			bundle.MaxBlockNumber = sseBundleData.MaxBlockNumber
			fmt.Println(time.Now(), "before sendMevBundle")
			_, err = testClient.SendMevBundle(context.Background(), *bundle)
			if err != nil {
				log.Fatal("bundle send err ", err)
			}
			fmt.Println(time.Now())
			fmt.Printf("bundle sent: %v\n", bundle)
			break
		} else if strings.HasPrefix(line, "ping:") {
			// 处理ping消息
			fmt.Println("Received ping")
		}
	}

	if err := scanner.Err(); err != nil {
		log.Fatalf("error reading from response: %v", err)
	}
}

// 模拟 searcher 产出 mev，先订阅服务，接受 rpc 推流，然后收到推流后，追加交易，发送两次 bundle
func TestSendBundleTwice(t *testing.T) {

	// 指定用户发送，这里就是 api_user_test.go 里设置的用户
	//testClient, err := ethclient.Dial("https://bsc.blockrazor.xyz/1827949375179984896")
	testClient, err := ethclient.Dial("https://bsc.blockrazor.xyz")
	if err != nil {
		log.Fatal(err)
	}

	// 发起GET请求到SSE服务端
	resp, err := http.Get("https://bsc.blockrazor.xyz/stream")
	if err != nil {
		log.Fatalf("failed to connect to SSE server: %v", err)
	}
	defer resp.Body.Close()

	// 检查响应头是否为 "text/event-stream"
	if ct := resp.Header.Get("Content-Type"); ct != "text/event-stream" {
		log.Fatalf("unexpected Content-Type: %s", ct)
	}

	// 创建一个扫描器，用于逐行读取响应内容
	scanner := bufio.NewScanner(resp.Body)
	count := 0
	for scanner.Scan() {
		line := scanner.Text()
		if strings.HasPrefix(line, "data:") {
			// 处理数据行
			data := strings.TrimPrefix(line, "data: ")
			fmt.Println(time.Now())
			fmt.Printf("Received data: %s\n", data)
			sseBundleData := &define.SseBundleData{}
			json.Unmarshal([]byte(data), &sseBundleData)
			if count == 0 {
				count++
				// first bundle
				bundle := getBundle1(sseBundleData)
				bundle.Hash = common.HexToHash(sseBundleData.Hash)
				bundle.MaxBlockNumber = sseBundleData.MaxBlockNumber
				fmt.Println(time.Now(), "before sendMevBundle")
				_, err = testClient.SendMevBundle(context.Background(), *bundle)
				if err != nil {
					log.Fatal("bundle send err ", err)
				}
				fmt.Println(time.Now())
				fmt.Printf("bundle sent: %v\n", bundle)
			} else {
				//second bundle
				bundle := getBundle2(sseBundleData)
				bundle.Hash = common.HexToHash(sseBundleData.Hash)
				bundle.MaxBlockNumber = sseBundleData.MaxBlockNumber
				fmt.Println(time.Now(), "before sendMebBundle")
				_, err = testClient.SendMevBundle(context.Background(), *bundle)
				if err != nil {
					log.Fatal("bundle send err ", err)
				}
				fmt.Println(time.Now())
				fmt.Printf("bundle sent: %v\n", bundle)
				break
			}
		} else if strings.HasPrefix(line, "ping:") {
			// 处理ping消息
			fmt.Println("Received ping")
		}
	}

	if err := scanner.Err(); err != nil {
		log.Fatalf("error reading from response: %v", err)
	}
}

// 直接发送合约交易，调用 bid
func TestSendBidContractTx(t *testing.T) {

	testClient, err := ethclient.Dial("https://bsc.blockrazor.xyz")
	if err != nil {
		log.Fatal(err)
	}

	// 原始用户
	privateKey, err := crypto.HexToECDSA("504303443a7d10c1544a45acfa19cb1c9497995013bfed3c9ac8689bf4a2421e")
	if err != nil {
		log.Fatal(err)
	}
	publicKey := privateKey.Public()
	publicKeyECDSA, ok := publicKey.(*ecdsa.PublicKey)
	if !ok {
		log.Fatal("cannot assert type: publicKey is not of type *ecdsa.PublicKey")
	}

	fromAddress := crypto.PubkeyToAddress(*publicKeyECDSA)
	value := big.NewInt(4 * 1e14)
	gasLimit := uint64(150_000)
	gasPrice := big.NewInt(1e9)
	// 合约地址
	toAddress := common.HexToAddress("0x0C529A429A9fB8aC769E55513226970563335904")

	data, err := TestBribeRpcMetaData.GenTxBidInput(common.HexToAddress("0x9Abae1b279A4Be25AEaE49a33e807cDd3cCFFa0C"), big.NewInt(10_090_000))
	if err != nil {
		log.Fatal(err)
	}

	nonce, err := testClient.PendingNonceAt(context.Background(), fromAddress)
	if err != nil {
		return
	}

	tx := types.NewTx(&types.LegacyTx{
		Nonce:    nonce,
		GasPrice: gasPrice,
		Gas:      gasLimit,
		To:       &toAddress,
		Value:    value,
		Data:     data,
	})

	fmt.Println(time.Now(), "before SignTx")
	signedTx, err := types.SignTx(tx, types.LatestSignerForChainID(big.NewInt(56)), privateKey)
	fmt.Println(time.Now(), "before SignTx")
	err = testClient.SendTransaction(context.Background(), signedTx)
	if err != nil {
		log.Fatal(err)
	}
	log.Println("tx hash ", signedTx.Hash().Hex())
}

// 调用合约，设置 SetSystemAddress
func TestSendSetSystemAddressContractTx(t *testing.T) {

	testClient, err := ethclient.Dial("https://bsc.blockrazor.xyz")
	if err != nil {
		log.Fatal(err)
	}

	// 原始用户
	privateKey, err := crypto.HexToECDSA("504303443a7d10c1544a45acfa19cb1c9497995013bfed3c9ac8689bf4a2421e")
	if err != nil {
		log.Fatal(err)
	}
	publicKey := privateKey.Public()
	publicKeyECDSA, ok := publicKey.(*ecdsa.PublicKey)
	if !ok {
		log.Fatal("cannot assert type: publicKey is not of type *ecdsa.PublicKey")
	}

	fromAddress := crypto.PubkeyToAddress(*publicKeyECDSA)
	value := big.NewInt(0)
	gasLimit := uint64(100_000)
	gasPrice := big.NewInt(1e9)
	// 合约地址
	toAddress := common.HexToAddress("0x0C529A429A9fB8aC769E55513226970563335904")

	data, err := TestBribeRpcMetaData.GenTxSetSystemAddressInput(common.HexToAddress("0x1Bc0631E1d71FF2571a559374aD98F9F3860A271"))
	if err != nil {
		log.Fatal(err)
	}

	nonce, err := testClient.PendingNonceAt(context.Background(), fromAddress)
	if err != nil {
		return
	}

	tx := types.NewTx(&types.LegacyTx{
		Nonce:    nonce,
		GasPrice: gasPrice,
		Gas:      gasLimit,
		To:       &toAddress,
		Value:    value,
		Data:     data,
	})

	fmt.Println(time.Now(), "before SignTx")
	signedTx, err := types.SignTx(tx, types.LatestSignerForChainID(big.NewInt(56)), privateKey)
	fmt.Println(time.Now(), "before SignTx")
	err = testClient.SendTransaction(context.Background(), signedTx)
	if err != nil {
		log.Fatal(err)
	}
	log.Println("tx hash ", signedTx.Hash().Hex())
}

// 调用合约，设置 frozen 冻结合约
func TestSendSetFrozenContractTx(t *testing.T) {

	testClient, err := ethclient.Dial("https://bsc.blockrazor.xyz")
	if err != nil {
		log.Fatal(err)
	}

	// 原始用户
	privateKey, err := crypto.HexToECDSA("504303443a7d10c1544a45acfa19cb1c9497995013bfed3c9ac8689bf4a2421e")
	if err != nil {
		log.Fatal(err)
	}
	publicKey := privateKey.Public()
	publicKeyECDSA, ok := publicKey.(*ecdsa.PublicKey)
	if !ok {
		log.Fatal("cannot assert type: publicKey is not of type *ecdsa.PublicKey")
	}

	fromAddress := crypto.PubkeyToAddress(*publicKeyECDSA)
	value := big.NewInt(0)
	gasLimit := uint64(100_000)
	gasPrice := big.NewInt(1e9)
	// 合约地址
	toAddress := common.HexToAddress("0x0C529A429A9fB8aC769E55513226970563335904")

	data, err := TestBribeRpcMetaData.GenTxSetFrozenInput(true)
	if err != nil {
		log.Fatal(err)
	}

	nonce, err := testClient.PendingNonceAt(context.Background(), fromAddress)
	if err != nil {
		return
	}

	tx := types.NewTx(&types.LegacyTx{
		Nonce:    nonce,
		GasPrice: gasPrice,
		Gas:      gasLimit,
		To:       &toAddress,
		Value:    value,
		Data:     data,
	})

	fmt.Println(time.Now(), "before SignTx")
	signedTx, err := types.SignTx(tx, types.LatestSignerForChainID(big.NewInt(56)), privateKey)
	fmt.Println(time.Now(), "before SignTx")
	err = testClient.SendTransaction(context.Background(), signedTx)
	if err != nil {
		log.Fatal(err)
	}
	log.Println("tx hash ", signedTx.Hash().Hex())
}

// 调用合约，设置 frozen 取消冻结合约
func TestSendSetUnFrozenContractTx(t *testing.T) {

	testClient, err := ethclient.Dial("https://bsc.blockrazor.xyz")
	if err != nil {
		log.Fatal(err)
	}

	// 原始用户
	privateKey, err := crypto.HexToECDSA("504303443a7d10c1544a45acfa19cb1c9497995013bfed3c9ac8689bf4a2421e")
	if err != nil {
		log.Fatal(err)
	}
	publicKey := privateKey.Public()
	publicKeyECDSA, ok := publicKey.(*ecdsa.PublicKey)
	if !ok {
		log.Fatal("cannot assert type: publicKey is not of type *ecdsa.PublicKey")
	}

	fromAddress := crypto.PubkeyToAddress(*publicKeyECDSA)
	value := big.NewInt(0)
	gasLimit := uint64(100_000)
	gasPrice := big.NewInt(1e9)
	// 合约地址
	toAddress := common.HexToAddress("0x0C529A429A9fB8aC769E55513226970563335904")

	data, err := TestBribeRpcMetaData.GenTxSetFrozenInput(false)
	if err != nil {
		log.Fatal(err)
	}

	nonce, err := testClient.PendingNonceAt(context.Background(), fromAddress)
	if err != nil {
		return
	}

	tx := types.NewTx(&types.LegacyTx{
		Nonce:    nonce,
		GasPrice: gasPrice,
		Gas:      gasLimit,
		To:       &toAddress,
		Value:    value,
		Data:     data,
	})

	fmt.Println(time.Now(), "before SignTx")
	signedTx, err := types.SignTx(tx, types.LatestSignerForChainID(big.NewInt(56)), privateKey)
	fmt.Println(time.Now(), "before SignTx")
	err = testClient.SendTransaction(context.Background(), signedTx)
	if err != nil {
		log.Fatal(err)
	}
	log.Println("tx hash ", signedTx.Hash().Hex())
}

// 性能测试，使用单个账户发送
func TestSendRawTxBench(t *testing.T) {
	//testClient, err := ethclient.Dial("https://bsc.blockrazor.xyz/bsc/18279493751799848961111111111111?RefundPercent=85")
	//testClient, err := ethclient.Dial("https://bsc.blockrazor.xyz/maxbackrun")
	testClient, err := ethclient.Dial("https://bsc.blockrazor.xyz")
	if err != nil {
		log.Fatal(err)
	}

	// 原始用户
	privateKey, err := crypto.HexToECDSA("504303443a7d10c1544a45acfa19cb1c9497995013bfed3c9ac8689bf4a2421e")
	if err != nil {
		log.Fatal(err)
	}
	// 手动从私钥得到公钥，然后计算 hash 得到地址
	publicKey := privateKey.Public()
	publicKeyECDSA, ok := publicKey.(*ecdsa.PublicKey)
	if !ok {
		log.Fatal("cannot assert type: publicKey is not of type *ecdsa.PublicKey")
	}

	fromAddress := crypto.PubkeyToAddress(*publicKeyECDSA)
	// 自动获取当前的 nouce，也就是该用户当前交易数 + 1
	nonce, err := testClient.PendingNonceAt(context.Background(), fromAddress)
	if err != nil {
		log.Fatal(err)
	}
	// 构造 transaction 交易对象
	value := big.NewInt(1 * 1e14)
	gasLimit := uint64(30_000) // in units
	// 获取当前系统推荐的 gasPrice
	gasPrice := big.NewInt(1e9)
	// searcher01
	toAddress := common.HexToAddress("0x9Abae1b279A4Be25AEaE49a33e807cDd3cCFFa0C")

	group := sync.WaitGroup{}
	for i := 0; i < 500; i++ {
		group.Add(1)
		go func() {
			nonce--
			// New transaction 新的 api
			tx := types.NewTx(&types.LegacyTx{
				Nonce:    nonce,
				GasPrice: gasPrice,
				Gas:      gasLimit,
				To:       &toAddress,
				Value:    value,
				Data:     nil,
			})

			// 使用发件人的私钥对 tx 进行签名
			//chainId, err := testClient.NetworkID(context.Background())
			//if err != nil {
			//	log.Fatal(err)
			//}
			signedTx, err := types.SignTx(tx, types.LatestSignerForChainID(big.NewInt(56)), privateKey)
			if err != nil {
				log.Fatal(err)
			}
			err = testClient.SendTransaction(context.Background(), signedTx)
			//fmt.Println("tx hash ", signedTx.Hash().Hex())
			if err != nil {
				log.Println(err)
			} else {
				fmt.Println("success tx hash ", signedTx.Hash().Hex())
			}
			group.Done()
		}()
	}
	group.Wait()

}

// 测试订阅功能
func TestSubscribe(t *testing.T) {
	// 发起GET请求到SSE服务端
	resp, err := http.Get("https://bsc.blockrazor.xyz/stream")
	if err != nil {
		log.Fatalf("failed to connect to SSE server: %v", err)
	}
	defer resp.Body.Close()

	// 检查响应头是否为 "text/event-stream"
	if ct := resp.Header.Get("Content-Type"); ct != "text/event-stream" {
		log.Fatalf("unexpected Content-Type: %s", ct)
	}

	// 创建一个扫描器，用于逐行读取响应内容
	scanner := bufio.NewScanner(resp.Body)
	for scanner.Scan() {
		line := scanner.Text()
		if strings.HasPrefix(line, "data:") {
			// 处理数据行
			data := strings.TrimPrefix(line, "data: ")
			fmt.Println(time.Now())
			fmt.Printf("Received data: %s\n", data)
		} else if strings.HasPrefix(line, "ping:") {
			// 处理ping消息
			fmt.Println("Received ping")
		}
	}

	if err := scanner.Err(); err != nil {
		log.Fatalf("error reading from response: %v", err)
	}

}

// 性能测试，随机生成很多账户发送
func TestSendRandomRawTxBench(t *testing.T) {
	//testClient, err := ethclient.Dial("https://bsc.blockrazor.xyz/bsc/18279493751799848961111111111111?RefundPercent=85")
	//testClient, err := ethclient.Dial("https://bsc.blockrazor.xyz/maxbackrun")
	testClient, err := ethclient.Dial("https://bsc.blockrazor.xyz")
	if err != nil {
		log.Fatal(err)
	}

	// 得到私钥
	start := time.Now()
	group := sync.WaitGroup{}
	for i := 0; i < 5000; i++ {
		group.Add(1)
		go func() {
			// 生成随机私钥
			privateKey, err := crypto.GenerateKey()
			if err != nil {
				log.Fatal(err)
			}
			value := big.NewInt(0)
			gasLimit := uint64(30_000) // in units
			gasPrice := big.NewInt(params.GWei)
			//gasPrice := big.NewInt(0)
			toAddress := common.HexToAddress("0x9Abae1b279A4Be25AEaE49a33e807cDd3cCFFa0C")
			tx := types.NewTx(&types.LegacyTx{
				Nonce:    0,
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
			err = testClient.SendTransaction(context.Background(), signedTx)
			//fmt.Println("tx hash ", signedTx.Hash().Hex())
			if err != nil {
				log.Println(err)
			} else {
				fmt.Println("success tx hash ", signedTx.Hash().Hex())
			}
			group.Done()
		}()
	}
	group.Wait()
	end := time.Now()
	elapsed := end.Sub(start)
	fmt.Printf("elapsed: %s\n", elapsed)

}

// searcher01构造的第一层 bundle
func getBundle1(sseBundleData *define.SseBundleData) *types.SendMevBundleArgs {
	// searcher01
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
	value := big.NewInt(1 * 1e4) // 第一次贿赂额 searcher01 的贿赂
	gasLimit := uint64(100_000)
	//gasPrice := big.NewInt(1e9)
	toAddress := common.HexToAddress(sseBundleData.ProxyBidContract)

	transferFnSignature := []byte("proxyBid(address,uint256)")
	hash := sha3.NewLegacyKeccak256()
	hash.Write(transferFnSignature)
	methodID := hash.Sum(nil)[:4] // 只取前四个字节即可
	// 然后是两个参数
	// 两个数据都需要构建成 16 进制的 32 字节大小，不够的用 0 填充
	// 代币接收地址
	refundAddress := common.HexToAddress(sseBundleData.RefundAddress)
	paddedAddress := common.LeftPadBytes(refundAddress.Bytes(), 32)
	// 转账金额
	cfg := big.NewInt(int64(sseBundleData.RefundCfg))
	paddedAmount := common.LeftPadBytes(cfg.Bytes(), 32)
	// 构建 data 的字节切片对象
	var data []byte
	data = append(data, methodID...)
	data = append(data, paddedAddress...)
	data = append(data, paddedAmount...)

	//data, err := TestBribeRpcMetaData.GenTxBidInput(common.HexToAddress(sseBundleData.RefundAddress), big.NewInt(int64(sseBundleData.RefundCfg)))
	//if err != nil {
	//	log.Fatal(err)
	//}

	tx := types.NewTx(&types.DynamicFeeTx{
		Nonce:     uint64(nonce1),
		GasFeeCap: big.NewInt(1e9),
		GasTipCap: big.NewInt(1e9),
		Gas:       gasLimit,
		To:        &toAddress,
		Value:     value,
		Data:      data,
	})

	fmt.Println(time.Now(), "before SignTx")
	signedTx, err := types.SignTx(tx, types.LatestSignerForChainID(big.NewInt(56)), privateKey)
	fmt.Println(time.Now(), "before SignTx")
	if err != nil {
		log.Fatal(err)
	}
	b, err := signedTx.MarshalBinary()
	if err != nil {
		return nil
	}
	//b := new(bytes.Buffer)
	//err = signedTx.EncodeRLP(b)
	//if err != nil {
	//	return nil
	//}

	hint := map[string]bool{
		types.HintTo:               true,
		types.HintValue:            true,
		types.HintNonce:            true,
		types.HintFrom:             true,
		types.HintCallData:         true,
		types.HintFunctionSelector: true,
		types.HintHash:             true,
		types.HintGasPrice:         true,
		types.HintGasLimit:         true,
		types.HintLogs:             true,
	}

	bundle := &types.SendMevBundleArgs{
		Hint:              hint,
		Txs:               []hexutil.Bytes{b},
		Hash:              common.Hash{},
		RevertingTxHashes: make([]common.Hash, 0),
		RefundPercent:     90,
		RefundAddress:     fromAddress,
	}

	return bundle
}

// searcher02 构造的第二层 bundle
func getBundle2(sseBundleData *define.SseBundleData) *types.SendMevBundleArgs {
	// searcher02
	privateKey, err := crypto.HexToECDSA("f284e94732ea2696ced097982639912acac32901d89747d2c04ab3a1f2bca8d9")
	if err != nil {
		log.Fatal(err)
	}
	publicKey := privateKey.Public()
	publicKeyECDSA, ok := publicKey.(*ecdsa.PublicKey)
	if !ok {
		log.Fatal("cannot assert type: publicKey is not of type *ecdsa.PublicKey")
	}

	fromAddress := crypto.PubkeyToAddress(*publicKeyECDSA)
	value := big.NewInt(1 * 1e14) // 第二次贿赂额 searcher02 的贿赂
	gasLimit := uint64(100_000)
	gasPrice := big.NewInt(1e9)
	toAddress := common.HexToAddress(sseBundleData.ProxyBidContract)

	data, err := TestBribeRpcMetaData.GenTxBidInput(common.HexToAddress(sseBundleData.RefundAddress), big.NewInt(int64(sseBundleData.RefundCfg)))
	if err != nil {
		log.Fatal(err)
	}

	tx := types.NewTx(&types.LegacyTx{
		Nonce:    uint64(nonce2),
		GasPrice: gasPrice,
		Gas:      gasLimit,
		To:       &toAddress,
		Value:    value,
		Data:     data,
	})

	fmt.Println(time.Now(), "before SignTx")
	signedTx, err := types.SignTx(tx, types.LatestSignerForChainID(big.NewInt(56)), privateKey)
	fmt.Println(time.Now(), "before SignTx")
	if err != nil {
		log.Fatal(err)
	}
	b := new(bytes.Buffer)
	err = signedTx.EncodeRLP(b)
	if err != nil {
		return nil
	}

	hint := map[string]bool{
		types.HintTo:               true,
		types.HintValue:            true,
		types.HintNonce:            true,
		types.HintFrom:             true,
		types.HintCallData:         true,
		types.HintFunctionSelector: true,
		types.HintHash:             true,
		types.HintGasPrice:         true,
		types.HintGasLimit:         true,
		types.HintLogs:             true,
	}

	bundle := &types.SendMevBundleArgs{
		Hint:              hint,
		Txs:               []hexutil.Bytes{b.Bytes()},
		Hash:              common.Hash{},
		RevertingTxHashes: make([]common.Hash, 0),
		RefundPercent:     90,
		RefundAddress:     fromAddress,
	}

	return bundle
}

// 这下面是 abi 的函数用于生成发起合约交易需要的 data 信息。
// 需要调用哪个合约函数，就调用哪个方法获取 data 即可
type TestRpcMetaData struct {
	mu  sync.Mutex
	Bin string
	ABI string
	ab  *abi.ABI
}

var TestBribeRpcMetaData = &TestRpcMetaData{
	ABI: "[{\"inputs\":[],\"stateMutability\":\"nonpayable\",\"type\":\"constructor\"},{\"inputs\":[{\"internalType\":\"address\",\"name\":\"refundAddress\",\"type\":\"address\"},{\"internalType\":\"uint256\",\"name\":\"refundCfg\",\"type\":\"uint256\"}],\"name\":\"proxyBid\",\"outputs\":[],\"stateMutability\":\"payable\",\"type\":\"function\"},{\"inputs\":[],\"name\":\"frozen\",\"outputs\":[{\"internalType\":\"bool\",\"name\":\"\",\"type\":\"bool\"}],\"stateMutability\":\"view\",\"type\":\"function\"},{\"inputs\":[],\"name\":\"owner\",\"outputs\":[{\"internalType\":\"address\",\"name\":\"\",\"type\":\"address\"}],\"stateMutability\":\"view\",\"type\":\"function\"},{\"inputs\":[{\"internalType\":\"bool\",\"name\":\"_frozen\",\"type\":\"bool\"}],\"name\":\"setFrozen\",\"outputs\":[],\"stateMutability\":\"nonpayable\",\"type\":\"function\"},{\"inputs\":[{\"internalType\":\"address\",\"name\":\"_systemAddress\",\"type\":\"address\"}],\"name\":\"setSystemAddress\",\"outputs\":[],\"stateMutability\":\"nonpayable\",\"type\":\"function\"},{\"inputs\":[],\"name\":\"systemAddress\",\"outputs\":[{\"internalType\":\"address\",\"name\":\"\",\"type\":\"address\"}],\"stateMutability\":\"view\",\"type\":\"function\"},{\"stateMutability\":\"payable\",\"type\":\"receive\"}]",
	Bin: "0x6080604052348015600e575f80fd5b50335f806101000a81548173ffffffffffffffffffffffffffffffffffffffff021916908373ffffffffffffffffffffffffffffffffffffffff1602179055503360015f6101000a81548173ffffffffffffffffffffffffffffffffffffffff021916908373ffffffffffffffffffffffffffffffffffffffff1602179055505f8060146101000a81548160ff021916908315150217905550610b71806100b45f395ff3fe608060405260043610610058575f3560e01c8063054f7d9c14610063578063062101971461008d57806359d667a5146100b55780637e932d32146100d15780638da5cb5b146100f9578063d3e848f1146101235761005f565b3661005f57005b5f80fd5b34801561006e575f80fd5b5061007761014d565b604051610084919061069a565b60405180910390f35b348015610098575f80fd5b506100b360048036038101906100ae9190610711565b61015f565b005b6100cf60048036038101906100ca919061076f565b610230565b005b3480156100dc575f80fd5b506100f760048036038101906100f291906107d7565b61058d565b005b348015610104575f80fd5b5061010d610638565b60405161011a9190610811565b60405180910390f35b34801561012e575f80fd5b5061013761065d565b6040516101449190610811565b60405180910390f35b5f60149054906101000a900460ff1681565b60015f9054906101000a900473ffffffffffffffffffffffffffffffffffffffff1673ffffffffffffffffffffffffffffffffffffffff163373ffffffffffffffffffffffffffffffffffffffff16146101ee576040517f08c379a00000000000000000000000000000000000000000000000000000000081526004016101e590610884565b60405180910390fd5b805f806101000a81548173ffffffffffffffffffffffffffffffffffffffff021916908373ffffffffffffffffffffffffffffffffffffffff16021790555050565b5f60149054906101000a900460ff161561027f576040517f08c379a0000000000000000000000000000000000000000000000000000000008152600401610276906108ec565b60405180910390fd5b5f620f42408261028f9190610964565b90505f60646103e8846102a29190610964565b6102ac9190610994565b90505f6103e8846102bd9190610994565b90506064831115610303576040517f08c379a00000000000000000000000000000000000000000000000000000000081526004016102fa90610a0e565b60405180910390fd5b6063821115610347576040517f08c379a000000000000000000000000000000000000000000000000000000000815260040161033e90610a0e565b60405180910390fd5b606481111561038b576040517f08c379a000000000000000000000000000000000000000000000000000000000815260040161038290610a0e565b60405180910390fd5b5f34116103cd576040517f08c379a00000000000000000000000000000000000000000000000000000000081526004016103c490610a76565b60405180910390fd5b5f3490505f606485836103e09190610a94565b6103ea9190610964565b90505f81836103f99190610ad5565b90505f6064868361040a9190610a94565b6104149190610964565b90505f81836104239190610ad5565b90505f821115610472578973ffffffffffffffffffffffffffffffffffffffff166108fc8390811502906040515f60405180830381858888f19350505050158015610470573d5f803e3d5ffd5b505b5f606487836104819190610a94565b61048b9190610964565b90505f818361049a9190610ad5565b90505f8211156104fd5773fffffffffffffffffffffffffffffffffffffffe73ffffffffffffffffffffffffffffffffffffffff166108fc8390811502906040515f60405180830381858888f193505050501580156104fb573d5f803e3d5ffd5b505b5f818761050a9190610b08565b111561057f575f8054906101000a900473ffffffffffffffffffffffffffffffffffffffff1673ffffffffffffffffffffffffffffffffffffffff166108fc82886105559190610b08565b90811502906040515f60405180830381858888f1935050505015801561057d573d5f803e3d5ffd5b505b505050505050505050505050565b60015f9054906101000a900473ffffffffffffffffffffffffffffffffffffffff1673ffffffffffffffffffffffffffffffffffffffff163373ffffffffffffffffffffffffffffffffffffffff161461061c576040517f08c379a000000000000000000000000000000000000000000000000000000000815260040161061390610884565b60405180910390fd5b805f60146101000a81548160ff02191690831515021790555050565b60015f9054906101000a900473ffffffffffffffffffffffffffffffffffffffff1681565b5f8054906101000a900473ffffffffffffffffffffffffffffffffffffffff1681565b5f8115159050919050565b61069481610680565b82525050565b5f6020820190506106ad5f83018461068b565b92915050565b5f80fd5b5f73ffffffffffffffffffffffffffffffffffffffff82169050919050565b5f6106e0826106b7565b9050919050565b6106f0816106d6565b81146106fa575f80fd5b50565b5f8135905061070b816106e7565b92915050565b5f60208284031215610726576107256106b3565b5b5f610733848285016106fd565b91505092915050565b5f819050919050565b61074e8161073c565b8114610758575f80fd5b50565b5f8135905061076981610745565b92915050565b5f8060408385031215610785576107846106b3565b5b5f610792858286016106fd565b92505060206107a38582860161075b565b9150509250929050565b6107b681610680565b81146107c0575f80fd5b50565b5f813590506107d1816107ad565b92915050565b5f602082840312156107ec576107eb6106b3565b5b5f6107f9848285016107c3565b91505092915050565b61080b816106d6565b82525050565b5f6020820190506108245f830184610802565b92915050565b5f82825260208201905092915050565b7f4e6f7420636f6e7472616374206f776e657200000000000000000000000000005f82015250565b5f61086e60128361082a565b91506108798261083a565b602082019050919050565b5f6020820190508181035f83015261089b81610862565b9050919050565b7f4269642066756e6374696f6e2069732063757272656e746c792066726f7a656e5f82015250565b5f6108d660208361082a565b91506108e1826108a2565b602082019050919050565b5f6020820190508181035f830152610903816108ca565b9050919050565b7f4e487b71000000000000000000000000000000000000000000000000000000005f52601260045260245ffd5b7f4e487b71000000000000000000000000000000000000000000000000000000005f52601160045260245ffd5b5f61096e8261073c565b91506109798361073c565b9250826109895761098861090a565b5b828204905092915050565b5f61099e8261073c565b91506109a98361073c565b9250826109b9576109b861090a565b5b828206905092915050565b7f726566756e6443666720697320696c6c6567616c0000000000000000000000005f82015250565b5f6109f860148361082a565b9150610a03826109c4565b602082019050919050565b5f6020820190508181035f830152610a25816109ec565b9050919050565b7f56616c7565206d7573742062652067726561746572207468616e2030000000005f82015250565b5f610a60601c8361082a565b9150610a6b82610a2c565b602082019050919050565b5f6020820190508181035f830152610a8d81610a54565b9050919050565b5f610a9e8261073c565b9150610aa98361073c565b9250828202610ab78161073c565b91508282048414831517610ace57610acd610937565b5b5092915050565b5f610adf8261073c565b9150610aea8361073c565b9250828203905081811115610b0257610b01610937565b5b92915050565b5f610b128261073c565b9150610b1d8361073c565b9250828201905080821115610b3557610b34610937565b5b9291505056fea264697066735822122094ac77a09b4cc70ab422a56e6db8858759d77c81bbadd9cc7bc8ea0490c0804864736f6c634300081a0033",
}

func (m *TestRpcMetaData) GetAbi() (*abi.ABI, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.ab != nil {
		return m.ab, nil
	}
	if parsed, err := abi.JSON(strings.NewReader(m.ABI)); err != nil {
		return nil, err
	} else {
		m.ab = &parsed
	}
	return m.ab, nil
}

func (m *TestRpcMetaData) GenTxBidInput(refundAddress common.Address, refundConfig *big.Int) ([]byte, error) {
	refundAbi, err := TestBribeRpcMetaData.GetAbi()
	if err != nil {
		return nil, err
	}
	input, err := (*refundAbi).Pack("proxyBid", refundAddress, refundConfig)
	if err != nil {
		return nil, err
	}
	return input, nil
}

func (m *TestRpcMetaData) GenTxSetSystemAddressInput(_systemAddress common.Address) ([]byte, error) {
	refundAbi, err := TestBribeRpcMetaData.GetAbi()
	if err != nil {
		return nil, err
	}
	input, err := (*refundAbi).Pack("setSystemAddress", _systemAddress)
	if err != nil {
		return nil, err
	}
	return input, nil
}

func (m *TestRpcMetaData) GenTxSetFrozenInput(_frozen bool) ([]byte, error) {
	refundAbi, err := TestBribeRpcMetaData.GetAbi()
	if err != nil {
		return nil, err
	}
	input, err := (*refundAbi).Pack("setFrozen", _frozen)
	if err != nil {
		return nil, err
	}
	return input, nil
}

func (m *TestRpcMetaData) GenTxSystemAddressInput() ([]byte, error) {
	refundAbi, err := TestBribeRpcMetaData.GetAbi()
	if err != nil {
		return nil, err
	}
	input, err := (*refundAbi).Pack("systemAddress")
	if err != nil {
		return nil, err
	}
	return input, nil
}

func (m *TestRpcMetaData) GenTxFrozenInput() ([]byte, error) {
	refundAbi, err := TestBribeRpcMetaData.GetAbi()
	if err != nil {
		return nil, err
	}
	input, err := (*refundAbi).Pack("frozen")
	if err != nil {
		return nil, err
	}
	return input, nil
}

/*
curl -s --data '
{
  "id": 1,
  "jsonrpc": "2.0",
  "method": "eth_sendMevBundle",
  "params": [{
    "hash": "0x0000000000000000000000000000000000000000000000000000000000000000",
    "txs": [
      "0xf84a8080808080808193a0437a5584216e68d1ff5bd7803161865e058f9bf4637fd1391213eac03ae64444a00df12bffe475d5dd8cc1544b72ee280471f1dcb5173827ba41eb25cfc3e54284"
    ],
    "revertingTxHashes": [
		"0x0000000000000000000000000000000000000000000000000000000000000000"
	],
    "maxBlockNumber": 0,
    "hint": {
      "calldata": true,
      "from": true,
      "function_selector": true,
      "gas_limit": true,
      "gas_price": true,
      "hash": true,
      "logs": true,
      "nonce": true,
      "to": true,
      "value": true
    },
    "refundRecipient": "0x9abae1b279a4be25aeae49a33e807cdd3ccffa0c",
    "refundPercent": 90
  }]
}
' -H "Content-Type: application/json" -X POST https://bsc.blockrazor.xyz
*/
