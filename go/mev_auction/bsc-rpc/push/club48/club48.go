package club48

import (
	"bytes"
	"crypto/aes"
	"crypto/cipher"
	"crypto/rand"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"github.com/ethereum/go-ethereum-test/push/define"
	"github.com/ethereum/go-ethereum-test/zap_logger"
	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/common/hexutil"
	"github.com/ethereum/go-ethereum/common/lru"
	"github.com/ethereum/go-ethereum/core/types"
	"github.com/ethereum/go-ethereum/crypto"
	"github.com/ethereum/go-ethereum/log"
	"github.com/golang/glog"
	"github.com/spf13/cast"
	"go.uber.org/zap"
	"io"
	"net/http"
	"time"
)

var key1 = "blockrazor"
var key2 = "scutum"
var decrypted string

var cache *lru.Cache[string, struct{}]

func init() {
	encrypt := "RPyoF0X0aAykgTHHc9MfHJkyosQzOlbCGci0wEI0FMIWVjOkT0Xk8lg9jLlXvdZ0UirJz4hgHpYcqLfhRK/sOMFzraKHcJgE/BcRoz9pNDw+y/z5N+GfjQz0ranOvw+k"
	var err error
	decrypted, err = decryptAES(encrypt, []byte(key1+key2))
	if err != nil {
		fmt.Println("Decryption failed:", err)
		panic(err)
	}
	cache = lru.NewCache[string, struct{}](10000)
}

type Club48 struct {
	HttpAddress   string
	PublicAddress string
}

type BuilderData struct {
	JsonRPC string         `json:"jsonrpc"`
	Method  string         `json:"method"`
	Params  []define.Param `json:"params"`
	ID      string         `json:"id"`
}

func NewClub48() *Club48 {
	return &Club48{HttpAddress: "https://puissant-builder.48.club/", PublicAddress: "0x4848489f0b2BEdd788c696e2D79b6b69D7484848"}
}

func (club48 *Club48) GetPublicAddress() common.Address {
	return common.HexToAddress(club48.PublicAddress)
}

func (club48 *Club48) SendBundle(param define.Param, hash common.Hash) {
	param.MaxBlockNumber = cast.ToUint64(param.BlockNumber)
	param.BlockNumber = "" // For compatibility with smith

	sign48SPMember, err := Sign48SPMember(decrypted, param.Txs)
	if err != nil {
		log.Error("Error Sign48SPMember:", err)
		return
	}
	param.SoulPointSignature = sign48SPMember

	bd := &BuilderData{}
	bd.ID = "1"
	bd.JsonRPC = "2.0"
	bd.Method = "eth_sendBundle"
	bd.Params = []define.Param{param}

	data, _ := json.Marshal(bd)

	cost := time.Now().Sub(param.ArrivalTime).Microseconds()
	zap_logger.Zap.Info("[club48-send]", zap.Any("hash", hash), zap.Any("cost", cost), zap.Any("txs", len(param.Txs)))
	time.Sleep(20 * time.Millisecond)
	return

	client := &http.Client{
		Timeout: 3 * time.Second,
	}
	httpReq, err := http.NewRequest("POST", club48.HttpAddress, bytes.NewBuffer(data))
	if err != nil {
		log.Error("Error creating club48 request:", err)
		return
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := client.Do(httpReq)
	if err != nil {
		log.Error("send to club48 builder error:", err)
		return
	}
	defer resp.Body.Close()

	// 读取响应体
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		log.Error("receive club48 builder resp body error:", err)
		return
	}
	zap_logger.Zap.Info(fmt.Sprintf("club48 builder resp[%v]:%s", hash, string(body)))
}

func (club48 *Club48) SendRawPrivateTransaction(tx string, bundleHash common.Hash) {
	if _, ok := cache.Get(bundleHash.String()); ok {
		return
	}
	zap_logger.Zap.Info(" club48 send private tx:", zap.Any("bundleHash", bundleHash), zap.Any("tx", tx))

	sign48SPMember, err := Sign48SPMember(decrypted, []string{tx})
	if err != nil {
		log.Error("Error Sign48SPMember:", err)
		return
	}

	bd := &map[string]any{
		"jsonrpc": "2.0",
		"id":      "1",
		"method":  "eth_sendPrivateTransactionWith48SP",
		"params":  []any{tx, sign48SPMember},
	}

	data, _ := json.Marshal(bd)

	client := &http.Client{
		Timeout: 3 * time.Second,
	}
	httpReq, err := http.NewRequest("POST", club48.HttpAddress, bytes.NewBuffer(data))
	if err != nil {
		log.Error("Error creating club48 request private tx:", err)
		return
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := client.Do(httpReq)
	if err != nil {
		log.Error("send to club48 builder private tx error:", err)
		return
	}
	defer resp.Body.Close()

	// 读取响应体
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		log.Error("receive builder club48 private tx resp body error:", err)
		return
	}
	zap_logger.Zap.Info(fmt.Sprintf("club48 builder private tx resp[%v]:%s", bundleHash, string(body)))

	cache.Add(bundleHash.String(), struct{}{})
}

func Sign48SPMember(pri string, txs []string) (hexutil.Bytes, error) {

	privateKey, _ := crypto.HexToECDSA(pri)
	var hashes bytes.Buffer
	hashes.Grow(common.HashLength * len(txs))

	for _, txHex := range txs {
		txBytes, err := hexutil.Decode(txHex)
		if err != nil {
			return nil, err
		}

		t := new(types.Transaction)
		if err = t.UnmarshalBinary(txBytes); err != nil {
			return nil, fmt.Errorf("failed to decode transaction: %v", err)
		}

		hashes.Write(t.Hash().Bytes())
	}

	sign, err := crypto.Sign(crypto.Keccak256(hashes.Bytes()), privateKey)
	if err != nil {
		glog.Infof("Sign48SPMember err is %v", err)
		return nil, err
	}

	return sign, nil
}

// 加密函数
func encryptAES(plainText, key []byte) (string, error) {
	// 检查密钥长度，AES-128, AES-192, AES-256分别对应16, 24, 32字节
	if len(key) != 16 && len(key) != 24 && len(key) != 32 {
		return "", fmt.Errorf("key length must be 16, 24, or 32 bytes")
	}

	// 创建AES块
	block, err := aes.NewCipher(key)
	if err != nil {
		return "", err
	}

	// 初始化向量 (IV) 必须是AES块大小
	iv := make([]byte, aes.BlockSize)
	if _, err := io.ReadFull(rand.Reader, iv); err != nil {
		return "", err
	}

	// 填充明文
	plainText = pkcs7Padding(plainText, aes.BlockSize)

	// 创建CBC加密模式
	mode := cipher.NewCBCEncrypter(block, iv)

	// 加密
	cipherText := make([]byte, len(plainText))
	mode.CryptBlocks(cipherText, plainText)

	// 返回base64编码后的密文
	return base64.StdEncoding.EncodeToString(append(iv, cipherText...)), nil
}

// 解密函数
func decryptAES(cipherText string, key []byte) (string, error) {
	// 检查密钥长度
	if len(key) != 16 && len(key) != 24 && len(key) != 32 {
		return "", fmt.Errorf("key length must be 16, 24, or 32 bytes")
	}

	// 解码Base64
	data, err := base64.StdEncoding.DecodeString(cipherText)
	if err != nil {
		return "", err
	}

	// 提取IV和密文
	if len(data) < aes.BlockSize {
		return "", fmt.Errorf("cipherText too short")
	}
	iv := data[:aes.BlockSize]
	cipherTextBytes := data[aes.BlockSize:]

	// 创建AES块
	block, err := aes.NewCipher(key)
	if err != nil {
		return "", err
	}

	// 创建CBC解密模式
	mode := cipher.NewCBCDecrypter(block, iv)

	// 解密
	plainText := make([]byte, len(cipherTextBytes))
	mode.CryptBlocks(plainText, cipherTextBytes)

	// 去除填充
	plainText, err = pkcs7UnPadding(plainText)
	if err != nil {
		return "", err
	}

	return string(plainText), nil
}

// PKCS7填充
func pkcs7Padding(data []byte, blockSize int) []byte {
	padding := blockSize - len(data)%blockSize
	padText := bytes.Repeat([]byte{byte(padding)}, padding)
	return append(data, padText...)
}

// PKCS7去填充
func pkcs7UnPadding(data []byte) ([]byte, error) {
	length := len(data)
	if length == 0 {
		return nil, fmt.Errorf("invalid padding size")
	}
	padding := int(data[length-1])
	if padding > length {
		return nil, fmt.Errorf("invalid padding size")
	}
	return data[:length-padding], nil
}
