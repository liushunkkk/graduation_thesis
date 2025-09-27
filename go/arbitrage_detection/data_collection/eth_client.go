package data_collection

import (
	"github.com/ethereum/go-ethereum/ethclient"
)

var EthClient *ethclient.Client

func init() {
	EthClient = NewEthClient()
}

func NewEthClient() *ethclient.Client {
	// infura
	//clientUrl := "https://bsc-mainnet.infura.io/v3/d481f2467f41493c9d699a59f6e03213"
	//clientUrl := "https://bsc-mainnet.infura.io/v3/72b5ad0bd13b46f291459d1dc4952771"

	// rpcfast 用不了
	//clientUrl := "https://bsc-mainnet.rpcfast.com?api_key=FQFzN9eYMQ2F3f6ICCYdhtLmw2HCk7nRsmMUKsivZSWLYvwg53grvEIzorIt8Xps"

	// chain stack
	clientUrl := "https://bsc-mainnet.core.chainstack.com/04b893f36a6e7162205b532a6efbee07" // github
	//clientUrl := "https://bsc-mainnet.core.chainstack.com/0424da0312cca554dccbaed288ee0c2d" // qq email
	client, err := ethclient.Dial(clientUrl)
	if err != nil {
		panic(err)
	}
	return client
}
