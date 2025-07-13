package data_collection

import (
	"context"
	"fmt"
	"github.com/ethereum/go-ethereum/common"
	"gorm.io/gorm"
	"math/big"
	"math/rand"
	"slices"
	"strconv"
)

// RunDataCollection 收集数据，根据从研究院拿到的套利数据集进行数据的收集
// 从链上拿到数据，存到数据库中
func RunDataCollection() {
	handleOne := func(id int) {
		ctx := context.Background()
		var transaction ArbitraryTransaction

		DB.First(&transaction, id)

		txHash := common.HexToHash(transaction.TxHash)
		tx, _, err := EthClient.TransactionByHash(ctx, txHash)

		if err != nil {
			fmt.Printf("[%d] [%s] TransactionByHash error: %s\n", id, transaction.TxHash, err)
		}

		receipt, err := EthClient.TransactionReceipt(ctx, txHash)
		if err != nil {
			fmt.Printf("[%d] [%s] TransactionReceipt error: %s\n", id, transaction.TxHash, err)
		}

		ethereumTransaction, err := ConvertToEthereumTransaction(tx)
		if err != nil {
			fmt.Printf("[%d] [%s] ConvertToEthereumTransaction error: %s\n", id, transaction.TxHash, err)
		}
		ethereumReceipt, err := ConvertToEthereumReceipt(receipt)
		if err != nil {
			fmt.Printf("[%d] [%s] ConvertToEthereumReceipt error: %s\n", id, transaction.TxHash, err)
		}

		err = DB.Transaction(func(tx *gorm.DB) error {
			tx1 := tx.Table(ethereumTransaction.TableName()).Create(ethereumTransaction)
			if tx1.Error != nil || tx1.RowsAffected == 0 {
				return tx.Error
			}
			tx2 := tx.Table(ethereumReceipt.TableName()).Create(ethereumReceipt)
			if tx2.Error != nil || tx2.RowsAffected == 0 {
				return tx.Error
			}
			fmt.Printf("[%d] [%s] arbitrary transaction inserted!\n", id, transaction.TxHash)
			return nil
		})

		if err != nil {
			fmt.Printf("[%d] [%s] insert arbitrary transaction into db error: %s\n", id, transaction.TxHash, err)
		}
	}

	id := 50001

	for id <= 50000 {
		handleOne(id)
		id++
	}
}

// RunDataCompletion 数据补全，主要是补全EthereumTransaction表的block_number字段
func RunDataCompletion() {
	completeOne := func(id uint) {
		var receipt EthereumReceipt
		DB.Table(Table_EthereumReceipts).Find(&receipt, id)
		if receipt.ID == id {
			tx := DB.Table(Table_EthereumTransactions).Where("tx_hash = ?", receipt.TxHash).Update("block_number", receipt.BlockNumber)
			if tx.Error != nil || tx.RowsAffected == 0 {
				fmt.Printf("[%d] [%s] update fail!\n", id, receipt.TxHash)
				return
			}
			fmt.Printf("[%d] [%s] updated!\n", id, receipt.TxHash)
		}
	}

	id := 1
	for id <= 50000 {
		completeOne(uint(id))
		id++
	}
}

// RunDataCompletionV2 数据补全，主要是补全ComparisonTransactions表的block_number字段
func RunDataCompletionV2() {
	completeOne := func(id uint) {
		var receipt EthereumReceipt
		DB.Table(Table_ComparisonReceipts).Find(&receipt, id)
		if receipt.ID == id {
			tx := DB.Table(Table_ComparisonTransactions).Where("tx_hash = ?", receipt.TxHash).Update("block_number", receipt.BlockNumber)
			if tx.Error != nil || tx.RowsAffected == 0 {
				fmt.Printf("[%d] [%s] update fail!\n", id, receipt.TxHash)
				return
			}
			fmt.Printf("[%d] [%s] updated!\n", id, receipt.TxHash)
		}
	}

	var total int64
	DB.Table(Table_ComparisonReceipts).Count(&total)

	id := int64(1)
	for id <= total {
		completeOne(uint(id))
		id++
	}
}

// RunDataCleanUp 清除无用数据
func RunDataCleanUp() {
	completeOne := func(id uint) {
		var transaction ComparisonTransaction
		DB.Table(Table_ComparisonTransactions).Find(&transaction, id)
		if transaction.ID == id {
			var exist int64
			tx := DB.Table(Table_ComparisonReceipts).Where("tx_hash = ?", transaction.TxHash).Count(&exist)
			if tx.Error == nil && exist == 0 {
				fmt.Printf("[%d] [%s] doesn't have receipt, need to delete!\n", id, transaction.TxHash)
				tx1 := DB.Table(Table_ComparisonTransactions).Delete(transaction)
				if tx1.Error == nil && tx1.RowsAffected == 1 {
					fmt.Printf("[%d] [%s] deleted!\n", id, transaction.TxHash)
				}
				return
			}
			fmt.Printf("[%d] [%s] have receipt!\n", id, transaction.TxHash)
		}
	}

	var total int64
	DB.Table(Table_ComparisonTransactions).Count(&total)

	id := int64(1)
	for id <= total {
		completeOne(uint(id))
		id++
	}
}

// RunComparisonDatasetCollection 收集对比数据
// 根据套利数据集的中的交易，每个block_number随机收集两倍的交易即可
func RunComparisonDatasetCollection() {
	ctx := context.Background()
	// 先查出所有的 block_number
	var blockNumberStrs []string
	tx := DB.Table(Table_EthereumTransactions).Distinct("block_number").Find(&blockNumberStrs)
	if tx.Error != nil || len(blockNumberStrs) == 0 {
		fmt.Printf("select block number fail, err: %s\n", tx.Error.Error())
	}
	fmt.Printf("select block numbers success, count: %d\n", len(blockNumberStrs))

	// 排序
	var blockNumbers []int
	for _, blockNumberStr := range blockNumberStrs {
		n, err := strconv.Atoi(blockNumberStr)
		if err != nil {
			fmt.Printf("[%s] atoi fail, err: %s\n", blockNumberStr, err)
		}
		blockNumbers = append(blockNumbers, n)
	}
	slices.Sort(blockNumbers)

	collectOne := func(bn int) error {
		// 获取数据库中该区块交易个数
		var currTxs []*EthereumTransaction
		tx := DB.Table(Table_EthereumTransactions).Where("block_number", bn).Find(&currTxs)
		if tx.Error != nil {
			return fmt.Errorf("get tx count in database fail, err: %s", tx.Error.Error())
		}
		fmt.Printf("[%d] get tx count: %d\n", bn, len(currTxs))

		blockNumber := big.NewInt(int64(bn))
		// 拿到block
		block, err := EthClient.BlockByNumber(ctx, blockNumber)
		if err != nil {
			return fmt.Errorf("get eth block fail, err: %s", err)
		}
		if block == nil {
			return fmt.Errorf("get eth block fail, block is nil")
		}

		transactions := block.Transactions()
		total := len(transactions)

		need := 1
		selected := make(map[int]bool)
		for need <= 2*len(currTxs) {
			random := rand.Intn(total)
			target := transactions[random]
			exist := slices.ContainsFunc(currTxs, func(transaction *EthereumTransaction) bool {
				return transaction.TxHash == target.Hash().Hex()
			})
			if exist || selected[random] {
				continue
			}
			tx, _, err := EthClient.TransactionByHash(ctx, target.Hash())

			if err != nil {
				fmt.Printf("[%d] [%s] TransactionByHash error: %s\n", need, target.Hash().Hex(), err)
				continue
			}

			receipt, err := EthClient.TransactionReceipt(ctx, target.Hash())
			if err != nil {
				fmt.Printf("[%d] [%s] TransactionReceipt error: %s\n", need, target.Hash().Hex(), err)
				continue
			}

			comparisonTransaction, err := ConvertToComparisonTransaction(tx)
			if err != nil {
				fmt.Printf("[%d] [%s] ConvertToComparisonTransaction error: %s\n", need, target.Hash().Hex(), err)
				continue
			}
			comparisonReceipt, err := ConvertToComparisonReceipt(receipt)
			if err != nil {
				fmt.Printf("[%d] [%s] ConvertToComparisonReceipt error: %s\n", need, target.Hash().Hex(), err)
				continue
			}

			if comparisonReceipt.Status == FailedStatus {
				fmt.Printf("[%d] [%s] ComparisonReceipt FailedStatus, ignore this one\n", need, target.Hash().Hex())
				continue
			}

			// 在这里填充一下 block_number，就不需要补全了
			comparisonTransaction.BlockNumber = comparisonReceipt.BlockNumber

			err = DB.Transaction(func(tx *gorm.DB) error {
				tx1 := tx.Table(comparisonTransaction.TableName()).Create(comparisonTransaction)
				if tx1.Error != nil || tx1.RowsAffected == 0 {
					return tx.Error
				}
				tx2 := tx.Table(comparisonReceipt.TableName()).Create(comparisonReceipt)
				if tx2.Error != nil || tx2.RowsAffected == 0 {
					return tx.Error
				}
				fmt.Printf("[%d] [%s] comparison transaction inserted!\n", need, target.Hash().Hex())
				return nil
			})

			if err != nil {
				fmt.Printf("[%d] [%s] insert comparison transaction into db error: %s\n", need, target.Hash().Hex(), err)
				continue
			}

			need++
			selected[random] = true
		}

		return nil
	}

	curr := 47536417

	// 遍历所有的block_number
	for i := 0; i < len(blockNumbers); i++ {
		bn := blockNumbers[i]
		if bn < curr {
			continue
		}
		if err := collectOne(bn); err != nil {
			fmt.Printf("[%d/%d] [%d] collect fail, err: %s\n", i, len(blockNumbers), bn, err)
			i--
		}
	}
}
