package data_collection

import (
	"github.com/ethereum/go-ethereum/ethclient"
)

var EthClient *ethclient.Client

func init() {
	EthClient = NewEthClient()
}

func NewEthClient() *ethclient.Client {
	infuraUrl := "https://bsc-mainnet.infura.io/v3/d481f2467f41493c9d699a59f6e03213"
	//infuraUrl := "https://bsc-mainnet.infura.io/v3/72b5ad0bd13b46f291459d1dc4952771"
	client, err := ethclient.Dial(infuraUrl)
	if err != nil {
		panic(err)
	}
	return client
}
