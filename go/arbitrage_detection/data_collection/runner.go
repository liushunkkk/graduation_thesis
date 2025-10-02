package data_collection

import (
	"context"
	"errors"
	"fmt"
	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/core/types"
	"gorm.io/gorm"
	"math/big"
	"math/rand"
	"slices"
	"strconv"
	"time"
)

// RunDataCollection 收集数据，根据从研究院拿到的套利数据集进行数据的收集
// 从链上拿到数据，存到数据库中
func RunDataCollection() {
	var handleOne func(int) error
	handleOne = func(id int) (err error) {

		defer func() {
			if e := recover(); e != nil {
				err = errors.New(fmt.Sprint(e))
			}
		}()
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
			if (tx1.Error != nil || tx1.RowsAffected == 0) && !errors.Is(tx1.Error, gorm.ErrDuplicatedKey) {
				return tx.Error
			}
			tx2 := tx.Table(ethereumReceipt.TableName()).Create(ethereumReceipt)
			if (tx2.Error != nil || tx2.RowsAffected == 0) && !errors.Is(tx2.Error, gorm.ErrDuplicatedKey) {
				return tx.Error
			}
			fmt.Printf("[%d] [%s] arbitrary transaction inserted!\n", id, transaction.TxHash)
			return nil
		})

		if err != nil {
			fmt.Printf("[%d] [%s] insert arbitrary transaction into db error: %s\n", id, transaction.TxHash, err)
		}

		return
	}

	id := 51115

	for id <= 100000 {
		err := handleOne(id)
		if err != nil {
			time.Sleep(8 * time.Second)
			fmt.Println("sleep 8 seconds...")
			id--
		}
		id++
	}
}

// RunDataCompletionEthereum 数据补全，主要是补全EthereumTransaction表的block_number字段
func RunDataCompletionEthereum() {
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

	var maxId int
	DB.Table(Table_EthereumReceipts).Select("MAX(id)").Scan(&maxId)
	fmt.Printf("maxId=%d\n", maxId)
	for id <= maxId {
		completeOne(uint(id))
		id++
	}
}

// RunDataCompletionComparison 数据补全，主要是补全ComparisonTransactions表的block_number字段
func RunDataCompletionComparison() {
	completeOne := func(id uint) {
		var receipt ComparisonReceipt
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

	var maxId int
	DB.Table(Table_ComparisonReceipts).Select("MAX(id)").Scan(&maxId)
	fmt.Printf("maxId=%d\n", maxId)
	id := 1
	for id <= maxId {
		completeOne(uint(id))
		id++
	}
}

// RunDataCleanUpComparison 清除对照组无用数据
func RunDataCleanUpComparison() {
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

	var maxId int
	DB.Table(Table_ComparisonTransactions).Select("MAX(id)").Scan(&maxId)
	fmt.Printf("maxId=%d\n", maxId)
	id := 1
	for id <= maxId {
		completeOne(uint(id))
		id++
	}
}

// RunDataCleanUpEthereum 清除实验组无用数据
func RunDataCleanUpEthereum() {
	completeOne := func(id uint) {
		var transaction EthereumTransaction
		DB.Table(Table_EthereumTransactions).Find(&transaction, id)
		if transaction.ID == id {
			var exist int64
			tx := DB.Table(Table_EthereumReceipts).Where("tx_hash = ?", transaction.TxHash).Count(&exist)
			if tx.Error == nil && exist == 0 {
				fmt.Printf("[%d] [%s] doesn't have receipt, need to delete!\n", id, transaction.TxHash)
				tx1 := DB.Table(Table_EthereumTransactions).Delete(transaction)
				if tx1.Error == nil && tx1.RowsAffected == 1 {
					fmt.Printf("[%d] [%s] deleted!\n", id, transaction.TxHash)
				}
				return
			}
			fmt.Printf("[%d] [%s] have receipt!\n", id, transaction.TxHash)
		}
	}

	var maxId int
	DB.Table(Table_EthereumTransactions).Select("MAX(id)").Scan(&maxId)
	fmt.Printf("maxId=%d\n", maxId)
	id := 1
	for id <= maxId {
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
		tryCount := 0
		selected := make(map[int]bool)
		for need <= 2*len(currTxs) {
			tryCount++
			if tryCount > 8*len(currTxs) { // 防止一些条数不够，死循环的情况，直接弹出。
				break
			}
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
				time.Sleep(3 * time.Second)
				continue
			}

			// len(data) == 0 的先不要，如果已经执行很多次，都拿不到其他的了，那就还是拿一下
			if tryCount < 6*len(currTxs) && (len(string(tx.Data())) == 0 || string(tx.Data()) == "0x") {
				continue
			}

			receipt, err := EthClient.TransactionReceipt(ctx, target.Hash())
			if err != nil {
				fmt.Printf("[%d] [%s] TransactionReceipt error: %s\n", need, target.Hash().Hex(), err)
				time.Sleep(3 * time.Second)
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

	curr := 47541183

	// 遍历所有的block_number
	for i := 0; i < len(blockNumbers); i++ {
		bn := blockNumbers[i]
		if bn != curr {
			continue
		}
		if err := collectOne(bn); err != nil {
			fmt.Printf("[%d/%d] [%d] collect fail, err: %s\n", i, len(blockNumbers), bn, err)
			time.Sleep(8 * time.Second)
			fmt.Printf("sleep 8 second...")
			i--
		}
	}
}

// RunDataFillEthereum 补充数据，有些transaction没有receipt，重新补充一下
func RunDataFillEthereum() {
	var handleOne func(string) error
	handleOne = func(th string) (err error) {

		defer func() {
			if e := recover(); e != nil {
				err = errors.New(fmt.Sprint(e))
			}
		}()
		ctx := context.Background()
		txHash := common.HexToHash(th)

		receipt, err := EthClient.TransactionReceipt(ctx, txHash)
		if err != nil {
			fmt.Printf("[%s] TransactionReceipt error: %s\n", th, err)
		}

		ethereumReceipt, err := ConvertToEthereumReceipt(receipt)
		if err != nil {
			fmt.Printf("[%s] ConvertToEthereumReceipt error: %s\n", th, err)
		}

		err = DB.Transaction(func(tx *gorm.DB) error {
			tx2 := tx.Table(ethereumReceipt.TableName()).Create(ethereumReceipt)
			if (tx2.Error != nil || tx2.RowsAffected == 0) && !errors.Is(tx2.Error, gorm.ErrDuplicatedKey) {
				return tx.Error
			}
			fmt.Printf("[%s] arbitrary transaction inserted!\n", th)
			return nil
		})

		if err != nil {
			fmt.Printf("[%s] insert arbitrary transaction into db error: %s\n", th, err)
		}

		return
	}

	var txHashes []string
	DB.Table(Table_EthereumTransactions).Select("tx_hash").Scan(&txHashes)
	for id, txHash := range txHashes {
		fmt.Printf("[%d] %s\n", id, txHash)
		var receipt EthereumReceipt

		DB.Table(Table_EthereumReceipts).Where("tx_hash", txHash).Find(&receipt)
		if receipt.ID == 0 {
			fmt.Printf("[%d] %s has no receipts\n", id, txHash)
			err := handleOne(txHash)
			if err != nil {
				fmt.Println("fill failed...")
			}
		}
	}
}

// RunDataFillComparison 补充数据，有些transaction没有receipt，重新补充一下
func RunDataFillComparison() {
	var handleOne func(string) error
	handleOne = func(th string) (err error) {

		defer func() {
			if e := recover(); e != nil {
				err = errors.New(fmt.Sprint(e))
			}
		}()
		ctx := context.Background()
		txHash := common.HexToHash(th)

		receipt, err := EthClient.TransactionReceipt(ctx, txHash)
		if err != nil {
			fmt.Printf("[%s] TransactionReceipt error: %s\n", th, err)
		}

		comparisonReceipt, err := ConvertToComparisonReceipt(receipt)
		if err != nil {
			fmt.Printf("[%s] ConvertToEthereumReceipt error: %s\n", th, err)
		}

		err = DB.Transaction(func(tx *gorm.DB) error {
			tx2 := tx.Table(comparisonReceipt.TableName()).Create(comparisonReceipt)
			if (tx2.Error != nil || tx2.RowsAffected == 0) && !errors.Is(tx2.Error, gorm.ErrDuplicatedKey) {
				return tx.Error
			}
			fmt.Printf("[%s] arbitrary transaction inserted!\n", th)
			return nil
		})

		if err != nil {
			fmt.Printf("[%s] insert arbitrary transaction into db error: %s\n", th, err)
		}

		return
	}

	var txHashes []string
	DB.Table(Table_ComparisonTransactions).Select("tx_hash").Scan(&txHashes)
	for _, txHash := range txHashes {
		var receipt ComparisonReceipt

		DB.Table(Table_ComparisonReceipts).Where("tx_hash", txHash).Find(&receipt)
		if receipt.ID == 0 {
			fmt.Printf("%s has no receipts\n", txHash)
			err := handleOne(txHash)
			if err != nil {
				fmt.Println("fill failed...")
			}
		}
	}
}

// RunComparisonDatasetLess 主要用于比对数据是否对得上
func RunComparisonDatasetLess() {
	// 获取数据库中该区块交易个数
	var results1 []BlockTxCount
	DB.Table(Table_EthereumTransactions).
		Select("block_number, COUNT(id) as tx_count").
		Group("block_number").
		Order("block_number").
		Scan(&results1)

	var results2 []BlockTxCount
	DB.Table(Table_ComparisonTransactions).
		Select("block_number, COUNT(id) as tx_count").
		Group("block_number").
		Order("block_number").
		Scan(&results2)

	for bn, n := range results1 {
		if 2*n.TxCount != results2[bn].TxCount {
			fmt.Printf("len(txs1) != len(txs2), %d %d %d", n.BlockNumber, n.TxCount, results2[bn].TxCount)
		}
	}

}

// RunDataFillFromAddressEthereum 补充from信息
func RunDataFillFromAddressEthereum() {
	var handleOne func(int64) error
	handleOne = func(id int64) (err error) {

		defer func() {
			if e := recover(); e != nil {
				err = errors.New(fmt.Sprint(e))
			}
		}()

		var transaction EthereumTransaction
		DB.First(&transaction, id)
		if transaction.ID == 0 {
			fmt.Printf("[%d] has no transaction\n", id)
			return nil
		}

		newTx := &types.Transaction{}
		err = newTx.UnmarshalJSON([]byte(transaction.OriginJsonString))
		if err != nil {
			return err
		}

		sender, err := types.Sender(types.LatestSignerForChainID(big.NewInt(56)), newTx)
		if err != nil {
			return err
		}

		transaction.FromAddress = sender.Hex()

		fmt.Printf("[%d] => sender: %s\n", transaction.ID, transaction.FromAddress)

		tx := DB.Table(Table_EthereumTransactions).Where("id = ?", transaction.ID).UpdateColumn("from_address", transaction.FromAddress)

		if tx.Error != nil {
			fmt.Printf("[%d] update failed\n", id)
			return errors.New(tx.Error.Error())
		}

		return
	}

	var maxId int64
	DB.Table(Table_EthereumTransactions).Select("MAX(id)").Scan(&maxId)
	id := int64(1)
	for id <= maxId {
		err := handleOne(id)
		if err != nil {
			fmt.Printf("[%d] %s\n", id, err)
		}
		id++
	}
}

// RunDataFillFromAddressComparison 补充from信息
func RunDataFillFromAddressComparison() {
	var handleOne func(int64) error
	handleOne = func(id int64) (err error) {

		defer func() {
			if e := recover(); e != nil {
				err = errors.New(fmt.Sprint(e))
			}
		}()

		var transaction ComparisonTransaction
		DB.First(&transaction, id)
		if transaction.ID == 0 {
			fmt.Printf("[%d] has no transaction\n", id)
			return nil
		}

		newTx := &types.Transaction{}
		err = newTx.UnmarshalJSON([]byte(transaction.OriginJsonString))
		if err != nil {
			return err
		}

		sender, err := types.Sender(types.LatestSignerForChainID(big.NewInt(56)), newTx)
		if err != nil {
			return err
		}

		transaction.FromAddress = sender.Hex()

		fmt.Printf("[%d] => sender: %s\n", transaction.ID, transaction.FromAddress)

		tx := DB.Table(Table_ComparisonTransactions).Where("id = ?", transaction.ID).UpdateColumn("from_address", transaction.FromAddress)

		if tx.Error != nil {
			fmt.Printf("[%d] update failed\n", id)
			return errors.New(tx.Error.Error())
		}

		return
	}

	var maxId int64
	DB.Table(Table_ComparisonTransactions).Select("MAX(id)").Scan(&maxId)
	id := int64(1)
	for id <= maxId {
		err := handleOne(id)
		if err != nil {
			fmt.Printf("[%d] %s\n", id, err)
		}
		id++
	}
}
