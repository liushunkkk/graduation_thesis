package data_collection

const (
	LegacyTx     = "LegacyTx"
	AccessListTx = "AccessListTx"
	DynamicFeeTx = "DynamicFeeTx"
	OtherTx      = "OtherTx"

	FailedStatus     = "FailedStatus"
	SuccessfulStatus = "SuccessfulStatus"
	OtherStatus      = "OtherStatus"
)

const (
	Table_ArbitraryTransaction   = "arbitrary_transaction"
	Table_EthereumTransactions   = "ethereum_transactions"
	Table_EthereumReceipts       = "ethereum_receipts"
	Table_ComparisonTransactions = "comparison_transactions"
	Table_ComparisonReceipts     = "comparison_receipts"
)
