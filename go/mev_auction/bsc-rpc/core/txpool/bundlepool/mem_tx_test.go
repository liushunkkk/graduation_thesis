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
	"sync"
	"testing"
	"time"
)

var memAddr = "http://34.226.211.254:8545"
var memOurClient *ethclient.Client
var memBnbClient *ethclient.Client

func init() {
	var err error
	memOurClient, err = ethclient.Dial(memAddr)
	if err != nil {
		log.Fatal(err)
	}
	memBnbClient, err = ethclient.Dial("https://bsc-dataseed.bnbchain.org")
	if err != nil {
		log.Fatal(err)
	}
}

func memSendOneTx() {
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
	nonce, err := memOurClient.PendingNonceAt(context.Background(), fromAddress)
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
	fmt.Println("Tx Hash", signedTx.Hash().Hex())

	var wg sync.WaitGroup

	wg.Add(1)
	go func() {
		fmt.Println("send to bnb before:", time.Now())
		err = memBnbClient.SendTransaction(context.Background(), signedTx)
		if err != nil {
			log.Fatal(err)
		}
		fmt.Println("Send to bnb after", time.Now())
		wg.Done()
	}()

	num := 100
	time.Sleep(time.Duration(num) * time.Millisecond)

	wg.Add(1)
	go func() {
		fmt.Println("send to our before:", time.Now())
		err = memOurClient.SendTransaction(context.Background(), signedTx)
		if err != nil {
			log.Fatal(err)
		}
		fmt.Println("Send to our after", time.Now())
		wg.Done()
	}()

	wg.Wait()
}

func TestMemTx(t *testing.T) {
	memSendOneTx()
}

func TestBenchmarkTx(t *testing.T) {
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
	nonce, err := memOurClient.PendingNonceAt(context.Background(), fromAddress)
	if err != nil {
		log.Fatal(err)
	}
	gasLimit := uint64(21_000) // in units
	gasPrice := big.NewInt(1e9)
	toAddress := common.HexToAddress("0x43DdA9d1Ac023bd3593Dff5A1A677247Bb98fE11")

	var wg sync.WaitGroup
	for i := 1; i < 500; i++ {
		wg.Add(1)
		go func() {
			c, _ := ethclient.Dial(memAddr)
			k := 0
			for {
				k++
				value := big.NewInt(int64(i*100000 + k))
				fmt.Println("value: ", value)
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
					log.Println(err)
				}
				fmt.Println("Tx Hash", signedTx.Hash().Hex())

				fmt.Println("send to our before:", time.Now())
				err = c.SendTransaction(context.Background(), signedTx)
				if err != nil {
					log.Println(err)
				}
				fmt.Println("Send to our after", time.Now())
			}
		}()
	}

	wg.Wait()
}
