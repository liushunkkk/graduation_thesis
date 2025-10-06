package data_collection

import (
	"context"
	"encoding/json"
	"fmt"
	"math/big"
	"net/http"
	"sort"
	"strings"
	"time"

	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/core/types"
	"github.com/ethereum/go-ethereum/ethclient"
)

var BSC_BUILDER = map[string]string{
	"0x48a5Ed9abC1a8FBe86ceC4900483f43a7f2dBB48": "48club",
	"0x487e5Dfe70119C1b320B8219B190a6fa95a5BB48": "48club",
	"0x48FeE1BB3823D72fdF80671ebaD5646Ae397BB48": "48club",
	"0x48B4bBEbF0655557A461e91B8905b85864B8BB48": "48club",
	"0x4827b423D03a349b7519Dda537e9A28d31ecBB48": "48club",
	"0x48B2665E5E9a343409199D70F7495c8aB660BB48": "48club",
	"0x3FC0c936c00908c07723ffbf2d536D6E0f62C3A4": "blockbus",
	"0x17e9F0D7E45A500f0148B29C6C98EfD19d95F138": "blockbus",
	"0x1319Be8b8Ec4AA81f501924BdCF365fBcAa8d753": "blockbus",
	"0x5532CdB3c0c4278f9848fc4560b495b70bA67455": "blockrazor",
	"0xBA4233f6e478DB76698b0A5000972Af0196b7bE1": "blockrazor",
	"0x539E24781f616F0d912B60813aB75B7b80b75C53": "blockrazor",
	"0x49D91b1Ab0CC6A1591c2e5863E602d7159d36149": "blockrazor",
	"0x50061047B9c7150f0Dc105f79588D1B07D2be250": "blockrazor",
	"0x0557E8CB169F90F6eF421a54e29d7dd0629Ca597": "blockrazor",
	"0x488e37fcB2024A5B2F4342c7dE636f0825dE6448": "blockrazor",
	"0x6Dddf681C908705472D09B1D7036B2241B50e5c7": "blocksmith",
	"0x76736159984AE865a9b9Cc0Df61484A49dA68191": "blocksmith",
	"0x5054b21D8baea3d602dca8761B235ee10bc0231E": "blocksmith",
	"0xD4376FdC9b49d90e6526dAa929f2766a33BFFD52": "bloxroute",
	"0x2873fc7aD9122933BECB384f5856f0E87918388d": "bloxroute",
	"0x432101856a330aafdeB049dD5fA03a756B3f1c66": "bloxroute",
	"0x2B217a4158933AAdE6D6494e3791D454B4D13AE7": "bloxroute",
	"0x0da52E9673529b6E06F444FbBED2904A37f66415": "bloxroute",
	"0xE1ec1AeCE7953ecB4539749B9AA2eEF63354860a": "bloxroute",
	"0x89434FC3a09e583F2cb4e47A8B8fe58De8BE6a15": "bloxroute",
	"0x10353562E662E333C0c2007400284e0e21cF74fF": "bloxroute",
	"0xa6d6086222812eFD5292fF284b0F7ff2a2B86Af4": "darwin",
	"0x3265A3243ee84e667a73073504cA4CdeD1413D82": "darwin",
	"0xdf11CD23992Fd48Cf2d245aC144010673275f285": "darwin",
	"0x9a3234b450518fadA098388B88e00deCAd96ad38": "inblock",
	"0xb49f86586a840AB9920D2f340a85586E50FD30a2": "inblock",
	"0x0F6D8b72F3687de6f2824903a83B3ba13c0e88A0": "inblock",
	"0x36CB523286D57680efBbfb417C63653115bCEBB5": "jetbldr",
	"0x3aD6121407f6EDb65C8B2a518515D45863C206A8": "jetbldr",
	"0x345324dC15F1CDcF9022E3B7F349e911fb823b4C": "jetbldr",
	"0xfd38358475078F81a45077f6e59dff8286e0dCA1": "jetbldr",
	"0x7F5fbFd8e2eB3160dF4c96757DEEf29E26F969a3": "jetbldr",
	"0xA0Cde9891C6966fCe740817cc5576De2C669AB43": "jetbldr",
	"0x79102dB16781ddDfF63F301C9Be557Fd1Dd48fA0": "nodereal",
	"0x5B526b45e833704d84b5C2EB0F41323dA9466c48": "nodereal",
	"0xd0d56b330a0dea077208b96910ce452fd77e1b6f": "nodereal",
	"0xa547F87B2BADE689a404544859314CBC01f2605e": "nodereal",
	"0x4f24ce4cd03a6503de97cf139af2c26347930b99": "nodereal",
	"0xFD3F1Ad459D585C50Cf4630649817C6E0cec7335": "nodereal",
	"0x812720cb4639550D7BDb1d8F2be463F4a9663762": "xzbuilder",
}

// 假设这些是已定义的常量和结构体
const (
	BSCChainID      = 56
	EtherscanAPIKey = "YOUR_ETHERSCAN_API_KEY" // 替换为你的API密钥
)

// TxWithGasPrice 用于按GasPrice排序交易
type TxWithGasPrice struct {
	tx       *types.Transaction
	gasPrice *big.Int
	txIndex  int
}

// getBuilder 判断区块是否由已知构建者创建，并返回构建者名称
func getBuilder(block *types.Block) string {
	if block == nil || len(block.Transactions()) == 0 {
		return ""
	}

	txs := block.Transactions()
	// 取最后4笔交易进行检查
	startIdx := len(txs) - 4
	if startIdx < 0 {
		startIdx = 0
	}

	signer := types.LatestSignerForChainID(big.NewInt(BSCChainID))

	for i := startIdx; i < len(txs); i++ {
		tx := txs[i]
		// 获取发送者地址
		sender, err := types.Sender(signer, tx)
		if err != nil {
			continue
		}

		// 检查发送者是否是已知构建者
		if name, exists := BSC_BUILDER[sender.Hex()]; exists {
			return name
		}

		// 检查接收者是否是已知构建者
		if tx.To() != nil {
			if name, exists := BSC_BUILDER[tx.To().Hex()]; exists {
				return name
			}
		}
	}

	return ""
}

// extractPrivateTx 从区块中提取隐私流交易
func extractPrivateTx(block *types.Block) []*types.Transaction {
	if block == nil || len(block.Transactions()) == 0 {
		return []*types.Transaction{}
	}

	txs := block.Transactions()
	totalTxCount := len(txs)

	// 如果交易太少，直接返回所有交易
	if totalTxCount <= 5 {
		return txs
	}

	// 收集交易及其GasPrice
	var txsWithGas []TxWithGasPrice
	for i, tx := range txs {
		gasPrice := tx.GasPrice()
		if gasPrice == nil {
			continue
		}
		txsWithGas = append(txsWithGas, TxWithGasPrice{
			tx:       tx,
			gasPrice: gasPrice,
			txIndex:  i,
		})
	}

	// 按GasPrice降序排序
	sort.Slice(txsWithGas, func(i, j int) bool {
		return txsWithGas[i].gasPrice.Cmp(txsWithGas[j].gasPrice) > 0
	})

	// 计算隐私流交易边界：总交易数的20% + 延伸2笔
	privacyFlowCount := int(float64(totalTxCount) * 0.2)
	if privacyFlowCount < 1 {
		privacyFlowCount = 1
	}
	boundaryIdx := privacyFlowCount + 2
	if boundaryIdx > totalTxCount {
		boundaryIdx = totalTxCount
	}

	// 返回区块头部到边界的交易
	return txs[:boundaryIdx]
}

// isSwap 判断交易是否包含至少2种代币的Transfer事件
func isSwap(tx *types.Transaction) bool {
	if EthClient == nil {
		return false
	}

	receipt, err := EthClient.TransactionReceipt(context.Background(), tx.Hash())
	if err != nil {
		return false
	}

	// Transfer事件签名哈希
	transferSigHash := common.HexToHash("0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef")

	// 收集涉及的代币地址
	tokenSet := make(map[common.Address]bool)

	for _, log := range receipt.Logs {
		// 检查是否是Transfer事件
		if len(log.Topics) > 0 && log.Topics[0] == transferSigHash {
			tokenSet[log.Address] = true
		}
	}

	// 至少涉及2种代币
	return len(tokenSet) >= 2
}

// isProfitable 判断交易是否存在套利账户
func isProfitable(tx *types.Transaction) bool {
	if EthClient == nil {
		return false
	}

	receipt, err := EthClient.TransactionReceipt(context.Background(), tx.Hash())
	if err != nil {
		return false
	}

	// Transfer事件签名哈希
	transferSigHash := common.HexToHash("0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef")

	// 账户-代币-余额变化映射
	accountTokenChanges := make(map[common.Address]map[common.Address]*big.Int)

	for _, log := range receipt.Logs {
		// 只处理Transfer事件
		if len(log.Topics) < 3 || log.Topics[0] != transferSigHash {
			continue
		}

		// 解析事件数据
		tokenAddr := log.Address
		from := common.HexToAddress(log.Topics[1].Hex())
		to := common.HexToAddress(log.Topics[2].Hex())

		// 解析金额
		amount := new(big.Int)
		amount.SetString(strings.TrimPrefix(string(log.Data), "0x"), 16)

		// 初始化映射
		if _, exists := accountTokenChanges[from]; !exists {
			accountTokenChanges[from] = make(map[common.Address]*big.Int)
		}
		if _, exists := accountTokenChanges[to]; !exists {
			accountTokenChanges[to] = make(map[common.Address]*big.Int)
		}
		if accountTokenChanges[from][tokenAddr] == nil {
			accountTokenChanges[from][tokenAddr] = big.NewInt(0)
		}
		if accountTokenChanges[to][tokenAddr] == nil {
			accountTokenChanges[to][tokenAddr] = big.NewInt(0)
		}

		// 更新余额变化：from减少，to增加
		accountTokenChanges[from][tokenAddr].Sub(accountTokenChanges[from][tokenAddr], amount)
		accountTokenChanges[to][tokenAddr].Add(accountTokenChanges[to][tokenAddr], amount)
	}

	// 检查是否存在套利账户：某账户某种代币只有增加没有减少
	for account, tokenChanges := range accountTokenChanges {
		for token, change := range tokenChanges {
			// 该代币有余额增加
			if change.Cmp(big.NewInt(0)) > 0 {
				// 检查该账户是否有该代币的减少
				hasDecrease := false
				for _, t := range receipt.Logs {
					if t.Address == token && len(t.Topics) >= 2 &&
						common.HexToAddress(t.Topics[1].Hex()) == account {
						hasDecrease = true
						break
					}
				}
				if !hasDecrease {
					return true
				}
			}
		}
	}

	return false
}

// isOpenSource 检查交易涉及的合约是否开源
func isOpenSource(tx *types.Transaction) bool {
	// 简单交易没有接收合约，视为未开源
	if tx.To() == nil {
		return false
	}

	contractAddr := tx.To().Hex()
	apiURL := fmt.Sprintf("https://api.bscscan.com/api?module=contract&action=getsourcecode&address=%s&apikey=%s",
		contractAddr, EtherscanAPIKey)

	resp, err := http.Get(apiURL)
	if err != nil {
		return false
	}
	defer resp.Body.Close()

	var result map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return false
	}

	// 检查API响应状态
	if result["status"] != "1" {
		return false
	}

	// 解析结果
	resultData, ok := result["result"].([]interface{})
	if !ok || len(resultData) == 0 {
		return false
	}

	contractData, ok := resultData[0].(map[string]interface{})
	if !ok {
		return false
	}

	// 检查源码是否存在
	sourceCode, ok := contractData["SourceCode"].(string)
	return ok && sourceCode != "" && sourceCode != " "
}

// FindArbi 查找指定区块范围内的原子套利交易
func FindArbi(start, end int64) {
	ctx := context.Background()

	// 确保客户端已初始化
	if EthClient == nil {
		var err error
		EthClient, err = ethclient.Dial("https://bsc-dataseed.binance.org/")
		if err != nil {
			fmt.Printf("无法连接到BSC客户端: %v\n", err)
			return
		}
		defer EthClient.Close()
	}

	handleOne := func(n int64) error {
		block, err := EthClient.BlockByNumber(ctx, big.NewInt(n))
		if err != nil {
			return fmt.Errorf("获取区块 %d 失败: %w", n, err)
		}

		// 获取区块构建者
		builder := getBuilder(block)
		if builder == "" {
			return nil // 不是已知构建者创建的区块，跳过
		}

		// 提取隐私流交易
		privateTxs := extractPrivateTx(block)
		if len(privateTxs) == 0 {
			return nil
		}

		// 检查每笔隐私流交易是否为原子套利交易
		signer := types.LatestSignerForChainID(big.NewInt(BSCChainID))
		for _, tx := range privateTxs {
			// 检查是否满足原子套利交易的三个条件
			if !isSwap(tx) {
				continue
			}
			if !isProfitable(tx) {
				continue
			}
			if isOpenSource(tx) {
				continue
			}

			// 获取发送者地址
			from, err := types.Sender(signer, tx)
			if err != nil {
				continue
			}

			// 保存结果到数据库
			toAddr := "0x"
			if tx.To() != nil {
				toAddr = tx.To().Hex()
			}

			arbitrageTx := &ArbitraryTransaction{
				Builder:   builder,
				From:      from.Hex(),
				To:        toAddr,
				BlockNum:  n,
				TxHash:    tx.Hash().Hex(),
				TimeStamp: time.Unix(int64(block.Time()), 0),
				MevType:   "atomic",
			}

			// 保存到数据库
			if err := DB.Table(Table_ArbitraryTransaction).Create(arbitrageTx).Error; err != nil {
				fmt.Printf("保存套利交易 %s 失败: %v\n", tx.Hash().Hex(), err)
			} else {
				fmt.Printf("发现原子套利交易: 区块 %d, TxHash: %s\n", n, tx.Hash().Hex())
			}
		}

		return nil
	}

	// 遍历处理指定范围内的所有区块
	for i := start; i < end; i++ {
		err := handleOne(i)
		if err != nil {
			fmt.Printf("处理区块 %d 错误: %v\n", i, err)
			time.Sleep(10 * time.Second) // 出错时重试
			i--                          // 重试当前区块
		}
	}
}
