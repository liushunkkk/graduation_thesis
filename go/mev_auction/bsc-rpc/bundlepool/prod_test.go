package bundlepool

import (
	"bufio"
	"bytes"
	"context"
	"crypto/ecdsa"
	"encoding/json"
	"fmt"
	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/common/hexutil"
	"github.com/ethereum/go-ethereum/core/types"
	"github.com/ethereum/go-ethereum/crypto"
	"github.com/ethereum/go-ethereum/ethclient"
	"github.com/ethereum/go-ethereum/push/define"
	"golang.org/x/crypto/sha3"
	"log"
	"math/big"
	"net/http"
	"strings"
	"testing"
	"time"
)

//var serverAddress = "https://ls.bsc.blockrazor.xyz"

//var serverAddress = "https://ls.bsc-back.blockrazor.xyz"

var serverAddress = "http://34.226.211.254:8545/1843906873313464320"
var testClientProd *ethclient.Client

func init() {
	client, err := ethclient.Dial(serverAddress)
	if err != nil {
		log.Fatal(err)
	}
	testClientProd = client
}

// 使用search01发送一个用户原始交易
func TestSendRawTxProd(t *testing.T) {
	// search01
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
	nonce, err := testClientProd.PendingNonceAt(context.Background(), fromAddress)
	if err != nil {
		log.Fatal(err)
	}
	value := big.NewInt(1e4)
	gasLimit := uint64(21_016) // in units
	gasPrice := big.NewInt(1e9)
	toAddress := common.HexToAddress("0x43DdA9d1Ac023bd3593Dff5A1A677247Bb98fE11")

	tx := types.NewTx(&types.LegacyTx{
		Nonce:    nonce,
		GasPrice: gasPrice,
		Gas:      gasLimit,
		To:       &toAddress,
		Value:    value,
		Data:     []byte("1"),
	})

	signedTx, err := types.SignTx(tx, types.LatestSignerForChainID(big.NewInt(56)), privateKey)
	if err != nil {
		log.Fatal(err)
	}

	err = testClientProd.SendTransaction(context.Background(), signedTx)
	fmt.Println("tx hash ", signedTx.Hash().Hex())
	if err != nil {
		log.Fatal(err)
	}
}

func TestSendRawBundleProd(t *testing.T) {
	// search01
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
	nonce, err := testClientProd.PendingNonceAt(context.Background(), fromAddress)
	if err != nil {
		log.Fatal(err)
	}

	var txs []hexutil.Bytes

	for i := 0; i < 15; i++ {
		value := big.NewInt(0)
		gasLimit := uint64(21_000) // in units
		gasPrice := big.NewInt(1)
		toAddress := common.HexToAddress("0x43DdA9d1Ac023bd3593Dff5A1A677247Bb98fE11")
		tx := types.NewTx(&types.LegacyTx{
			Nonce:    nonce + uint64(i),
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
		fmt.Println("tx:", i, signedTx.Hash())
		b1, err := signedTx.MarshalBinary()
		if err != nil {
			log.Fatal(err)
		}
		txs = append(txs, b1)
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
		Txs:               txs,
		Hash:              common.Hash{},
		RevertingTxHashes: make([]common.Hash, 0),
		RefundPercent:     90,
		RefundAddress:     fromAddress,
	}
	bundleHash, err := testClientProd.SendMevBundle(context.Background(), *bundle)
	if err != nil {
		return
	}
	fmt.Println("bundleHash: ", bundleHash)
}

// 使用原始用户发送一个交易
func SendRawTxProdByUser() {
	// search01
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
	nonce, err := testClientProd.PendingNonceAt(context.Background(), fromAddress)
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

	err = testClientProd.SendTransaction(context.Background(), signedTx)
	fmt.Println("tx hash ", signedTx.Hash().Hex())
	if err != nil {
		log.Fatal(err)
	}
}

// 模拟 searcher 产出 mev，先订阅服务，接受 rpc 推流，然后收到推流后，追加交易，发送一次 bundle
func TestSendBundleOnceProd(t *testing.T) {
	resp, err := http.Get(serverAddress + "/stream")
	if err != nil {
		log.Fatalf("failed to connect to SSE server: %v", err)
	}
	defer resp.Body.Close()

	// 检查响应头是否为 "text/event-stream"
	if ct := resp.Header.Get("Content-Type"); ct != "text/event-stream" {
		log.Fatalf("unexpected Content-Type: %s", ct)
	}

	nonceForSearch01, err := testClientProd.PendingNonceAt(context.Background(), common.HexToAddress("0x9Abae1b279A4Be25AEaE49a33e807cDd3cCFFa0C"))
	if err != nil {
		log.Fatal(err)
	}

	go func() {
		time.Sleep(time.Second)
		SendRawTxProdByUser()
	}()

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
			bundle := getBundle1Prod(sseBundleData, nonceForSearch01)
			bundle.Hash = common.HexToHash(sseBundleData.Hash)
			bundle.MaxBlockNumber = sseBundleData.MaxBlockNumber
			fmt.Println(time.Now(), "before sendMevBundle")
			_, err = testClientProd.SendMevBundle(context.Background(), *bundle)
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
func TestSendBundleTwiceProd(t *testing.T) {
	resp, err := http.Get(serverAddress + "/stream")
	if err != nil {
		log.Fatalf("failed to connect to SSE server: %v", err)
	}
	defer resp.Body.Close()

	// 检查响应头是否为 "text/event-stream"
	if ct := resp.Header.Get("Content-Type"); ct != "text/event-stream" {
		log.Fatalf("unexpected Content-Type: %s", ct)
	}

	nonceForSearch01, err := testClientProd.PendingNonceAt(context.Background(), common.HexToAddress("0x9Abae1b279A4Be25AEaE49a33e807cDd3cCFFa0C"))
	if err != nil {
		log.Fatal(err)
	}
	nonceForSearch02, err := testClientProd.PendingNonceAt(context.Background(), common.HexToAddress("0x456CDD3291E678C90fB1cbD9aa2588a5ee3D8280"))
	if err != nil {
		log.Fatal(err)
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
				bundle := getBundle1Prod(sseBundleData, nonceForSearch01)
				bundle.Hash = common.HexToHash(sseBundleData.Hash)
				bundle.MaxBlockNumber = sseBundleData.MaxBlockNumber
				fmt.Println(time.Now(), "before sendMevBundle")
				_, err = testClientProd.SendMevBundle(context.Background(), *bundle)
				if err != nil {
					log.Fatal("bundle send err ", err)
				}
				fmt.Println(time.Now())
				fmt.Printf("bundle sent: %v\n", bundle)
			} else {
				//second bundle
				bundle := getBundle2Prod(sseBundleData, nonceForSearch02)
				bundle.Hash = common.HexToHash(sseBundleData.Hash)
				bundle.MaxBlockNumber = sseBundleData.MaxBlockNumber
				fmt.Println(time.Now(), "before sendMebBundle")
				_, err = testClientProd.SendMevBundle(context.Background(), *bundle)
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

// searcher01构造的第一层 bundle
func getBundle1Prod(sseBundleData *define.SseBundleData, nonce uint64) *types.SendMevBundleArgs {
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
	refundAddress := common.HexToAddress(sseBundleData.RefundAddress)
	paddedAddress := common.LeftPadBytes(refundAddress.Bytes(), 32)
	cfg := big.NewInt(int64(sseBundleData.RefundCfg))
	paddedAmount := common.LeftPadBytes(cfg.Bytes(), 32)
	var data []byte
	data = append(data, methodID...)
	data = append(data, paddedAddress...)
	data = append(data, paddedAmount...)

	tx := types.NewTx(&types.DynamicFeeTx{
		Nonce:     nonce,
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

func getBundle2Prod(sseBundleData *define.SseBundleData, nonce uint64) *types.SendMevBundleArgs {
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
	value := big.NewInt(1e4) // 第二次贿赂额 searcher02 的贿赂
	gasLimit := uint64(100_000)
	gasPrice := big.NewInt(1e9)
	toAddress := common.HexToAddress(sseBundleData.ProxyBidContract)

	data, err := TestBribeRpcMetaData.GenTxBidInput(common.HexToAddress(sseBundleData.RefundAddress), big.NewInt(int64(sseBundleData.RefundCfg)))
	if err != nil {
		log.Fatal(err)
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
