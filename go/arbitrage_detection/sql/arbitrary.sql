# ************************************************************
# Sequel Ace SQL dump
# 版本号： 20080
#
# https://sequel-ace.com/
# https://github.com/Sequel-Ace/Sequel-Ace
#
# 主机: 127.0.0.1 (MySQL 8.0.40)
# 数据库: arbitrary
# 生成时间: 2025-10-06 03:04:42 +0000
# ************************************************************


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
SET NAMES utf8mb4;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE='NO_AUTO_VALUE_ON_ZERO', SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;


# 转储表 arbitrary_transaction
# ------------------------------------------------------------

DROP TABLE IF EXISTS `arbitrary_transaction`;

CREATE TABLE `arbitrary_transaction` (
  `id` int NOT NULL AUTO_INCREMENT,
  `searcher` char(42) COLLATE utf8mb4_general_ci DEFAULT NULL,
  `builder` varchar(255) COLLATE utf8mb4_general_ci DEFAULT NULL,
  `from` char(42) COLLATE utf8mb4_general_ci DEFAULT NULL,
  `to` char(42) COLLATE utf8mb4_general_ci DEFAULT NULL,
  `block_num` bigint DEFAULT NULL,
  `tx_hash` char(66) COLLATE utf8mb4_general_ci DEFAULT NULL,
  `time_stamp` datetime DEFAULT NULL,
  `mev_type` varchar(255) COLLATE utf8mb4_general_ci DEFAULT NULL,
  `position` bigint DEFAULT NULL,
  `bribe_value` text COLLATE utf8mb4_general_ci,
  `bribee` varchar(255) COLLATE utf8mb4_general_ci DEFAULT NULL,
  `bribe_type` varchar(255) COLLATE utf8mb4_general_ci DEFAULT NULL,
  `arb_profit` double DEFAULT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `uni_arbitrary_transaction_tx_hash` (`tx_hash`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;



# 转储表 comparison_receipts
# ------------------------------------------------------------

DROP TABLE IF EXISTS `comparison_receipts`;

CREATE TABLE `comparison_receipts` (
  `id` bigint unsigned NOT NULL AUTO_INCREMENT,
  `tx_hash` char(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NOT NULL,
  `tx_type` varchar(20) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NOT NULL,
  `post_state` longtext CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci,
  `status` varchar(20) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NOT NULL,
  `cumulative_gas_used` bigint unsigned NOT NULL,
  `bloom` longtext CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci,
  `logs` longtext CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci,
  `contract_address` char(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci DEFAULT NULL,
  `gas_used` bigint unsigned NOT NULL,
  `effective_gas_price` varchar(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci DEFAULT NULL,
  `blob_gas_used` bigint unsigned NOT NULL,
  `blob_gas_price` varchar(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci DEFAULT NULL,
  `block_hash` char(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci DEFAULT NULL,
  `block_number` varchar(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci DEFAULT NULL,
  `transaction_index` int unsigned NOT NULL,
  `origin_json_string` longtext CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci,
  PRIMARY KEY (`id`),
  UNIQUE KEY `uni_tx_hash` (`tx_hash`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;



# 转储表 comparison_transactions
# ------------------------------------------------------------

DROP TABLE IF EXISTS `comparison_transactions`;

CREATE TABLE `comparison_transactions` (
  `id` bigint unsigned NOT NULL AUTO_INCREMENT,
  `tx_hash` char(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NOT NULL,
  `tx_type` varchar(20) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NOT NULL,
  `nonce` bigint unsigned NOT NULL,
  `gas_price` varchar(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci DEFAULT NULL,
  `gas_tip_cap` varchar(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci DEFAULT NULL,
  `gas_fee_cap` varchar(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci DEFAULT NULL,
  `gas` bigint unsigned NOT NULL,
  `to` char(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci DEFAULT NULL,
  `value` varchar(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci DEFAULT NULL,
  `data` longtext CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci,
  `access_list` longtext CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci,
  `block_number` varchar(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci DEFAULT NULL,
  `v` varchar(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci DEFAULT NULL,
  `r` varchar(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci DEFAULT NULL,
  `s` varchar(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci DEFAULT NULL,
  `origin_json_string` longtext CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci,
  `from_address` varchar(128) COLLATE utf8mb4_general_ci DEFAULT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `uni_tx_hash` (`tx_hash`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;



# 转储表 ethereum_receipts
# ------------------------------------------------------------

DROP TABLE IF EXISTS `ethereum_receipts`;

CREATE TABLE `ethereum_receipts` (
  `id` bigint unsigned NOT NULL AUTO_INCREMENT,
  `tx_hash` char(128) COLLATE utf8mb4_general_ci NOT NULL,
  `tx_type` varchar(20) COLLATE utf8mb4_general_ci NOT NULL,
  `post_state` longtext COLLATE utf8mb4_general_ci,
  `status` varchar(20) COLLATE utf8mb4_general_ci NOT NULL,
  `cumulative_gas_used` bigint unsigned NOT NULL,
  `bloom` longtext COLLATE utf8mb4_general_ci,
  `logs` longtext COLLATE utf8mb4_general_ci,
  `contract_address` char(128) COLLATE utf8mb4_general_ci DEFAULT NULL,
  `gas_used` bigint unsigned NOT NULL,
  `effective_gas_price` varchar(128) COLLATE utf8mb4_general_ci DEFAULT NULL,
  `blob_gas_used` bigint unsigned NOT NULL,
  `blob_gas_price` varchar(128) COLLATE utf8mb4_general_ci DEFAULT NULL,
  `block_hash` char(128) COLLATE utf8mb4_general_ci DEFAULT NULL,
  `block_number` varchar(128) COLLATE utf8mb4_general_ci DEFAULT NULL,
  `transaction_index` int unsigned NOT NULL,
  `origin_json_string` longtext COLLATE utf8mb4_general_ci,
  PRIMARY KEY (`id`),
  UNIQUE KEY `uni_tx_hash` (`tx_hash`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;



# 转储表 ethereum_transactions
# ------------------------------------------------------------

DROP TABLE IF EXISTS `ethereum_transactions`;

CREATE TABLE `ethereum_transactions` (
  `id` bigint unsigned NOT NULL AUTO_INCREMENT,
  `tx_hash` char(128) COLLATE utf8mb4_general_ci NOT NULL,
  `tx_type` varchar(20) COLLATE utf8mb4_general_ci NOT NULL,
  `nonce` bigint unsigned NOT NULL,
  `gas_price` varchar(128) COLLATE utf8mb4_general_ci DEFAULT NULL,
  `gas_tip_cap` varchar(128) COLLATE utf8mb4_general_ci DEFAULT NULL,
  `gas_fee_cap` varchar(128) COLLATE utf8mb4_general_ci DEFAULT NULL,
  `gas` bigint unsigned NOT NULL,
  `to` char(128) COLLATE utf8mb4_general_ci DEFAULT NULL,
  `value` varchar(128) COLLATE utf8mb4_general_ci DEFAULT NULL,
  `data` longtext COLLATE utf8mb4_general_ci,
  `access_list` longtext COLLATE utf8mb4_general_ci,
  `v` varchar(128) COLLATE utf8mb4_general_ci DEFAULT NULL,
  `r` varchar(128) COLLATE utf8mb4_general_ci DEFAULT NULL,
  `s` varchar(128) COLLATE utf8mb4_general_ci DEFAULT NULL,
  `origin_json_string` longtext COLLATE utf8mb4_general_ci,
  `block_number` varchar(128) COLLATE utf8mb4_general_ci DEFAULT NULL,
  `from_address` varchar(128) COLLATE utf8mb4_general_ci DEFAULT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `uni_tx_hash` (`tx_hash`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;




/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;
/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
