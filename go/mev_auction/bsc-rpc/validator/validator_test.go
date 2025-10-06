package validator

import (
	"context"
	"fmt"
	"github.com/agiledragon/gomonkey/v2"
	"github.com/ethereum/go-ethereum/common/ms"
	"github.com/ethereum/go-ethereum/core"
	"github.com/ethereum/go-ethereum/core/types"
	"github.com/ethereum/go-ethereum/ethclient"
	. "github.com/smartystreets/goconvey/convey"
	"log"
	"math/big"
	"reflect"
	"strings"
	"testing"
	"time"
)

func TestNewValidatorServer(t *testing.T) {
	Convey("test", t, func() {
		client, err := ethclient.Dial("https://bsc.blockrazor.xyz")
		if err != nil {
			return
		}

		server := NewValidatorServer(&core.BlockChain{})

		method := gomonkey.ApplyMethod(reflect.TypeOf(&core.BlockChain{}), "CurrentBlock", func(chain *core.BlockChain) *types.Header {

			header, err := client.HeaderByNumber(context.Background(), nil)
			if err != nil {
				return nil
			}
			return header
		})
		defer method.Reset()

		m1 := gomonkey.ApplyMethod(reflect.TypeOf(&core.BlockChain{}), "GetHeaderByNumber", func(chain *core.BlockChain, number uint64) *types.Header {
			header, err := client.HeaderByNumber(context.Background(), big.NewInt(int64(number)))
			if err != nil {
				return nil
			}
			return header
		})
		defer m1.Reset()

		ms.Init(server)
		server.Go()

		header, err := client.HeaderByNumber(context.Background(), nil)
		if err != nil {
			log.Fatalf("Failed to retrieve latest block header: %v", err)
		}

		for i := header.Number.Int64(); i < header.Number.Int64()+200; i++ {
			b := server.NextBlockIs48Club(i)
			fmt.Println(i+1, b)
			time.Sleep(3 * time.Second)
			var h *types.Header
			for {
				h = (&core.BlockChain{}).GetHeaderByNumber(uint64(i + 1))
				if h != nil {
					break
				}
			}

			_, ok := club48Validator[strings.ToLower(h.Coinbase.Hex())]
			So(ok, ShouldEqual, b)
		}

		server.Stop()
	})
}

func TestNewValidatorServer1(t *testing.T) {
	Convey("test1", t, func() {
		client, err := ethclient.Dial("https://bsc.blockrazor.xyz")
		if err != nil {
			return
		}

		server := NewValidatorServer(&core.BlockChain{})

		method := gomonkey.ApplyMethod(reflect.TypeOf(&core.BlockChain{}), "CurrentBlock", func(chain *core.BlockChain) *types.Header {

			header, err := client.HeaderByNumber(context.Background(), nil)
			if err != nil {
				return nil
			}
			return header
		})
		defer method.Reset()

		m1 := gomonkey.ApplyMethod(reflect.TypeOf(&core.BlockChain{}), "GetHeaderByNumber", func(chain *core.BlockChain, number uint64) *types.Header {
			header, err := client.HeaderByNumber(context.Background(), big.NewInt(int64(number)))
			if err != nil {
				return nil
			}
			return header
		})
		defer m1.Reset()

		ms.Init(server)
		server.Go()

		header, err := client.HeaderByNumber(context.Background(), nil)
		if err != nil {
			log.Fatalf("Failed to retrieve latest block header: %v", err)
		}

		time.Sleep(6 * time.Second)

		for i := header.Number.Int64(); i < header.Number.Int64()+1000; i++ {
			b := server.NextBlock(i + 1)
			var h *types.Header
			for {
				h = (&core.BlockChain{}).GetHeaderByNumber(uint64(i + 2))
				if h != nil {
					break
				}
			}
			fmt.Println(i)
			So(strings.ToLower(h.Coinbase.Hex()), ShouldEqual, strings.ToLower(b))

			time.Sleep(3 * time.Second)
		}

		server.Stop()
	})
}
