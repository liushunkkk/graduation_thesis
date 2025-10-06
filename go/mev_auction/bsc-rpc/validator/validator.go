package validator

import (
	"context"
	"errors"
	"fmt"
	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/common/lru"
	"github.com/ethereum/go-ethereum/common/ms"
	"github.com/ethereum/go-ethereum/core"
	"strings"
	"time"
)

var club48Validator = map[string]struct{}{
	"0x38944092685a336cb6b9ea58836436709a2adc89": {},
	"0x8a239732871adc8829ea2f47e94087c5fbad47b6": {},
	"0x9bb56c2b4dbe5a06d79911c9899b6f817696acfc": {},
	"0xccb42a9b8d6c46468900527bc741938e78ab4577": {},
	"0xf8b99643fafc79d9404de68e48c4d49a3936f787": {},
	//"0x0dc5e1cae4d364d0c79c9ae6bddb5da49b10a7d9": {}, // 48club py
}
var Figment = "0x75b851a27d7101438f45fce31816501193239a83"
var TWStaking = "0x9f1b7fae54be07f4fee34eb1aacb39a1f7b6fc92"

var Server *ValidatorServer

type ValidatorServer struct {
	*ms.Server
	blockNum   int64
	blockChain *core.BlockChain
	list       *lru.Cache[int64, []string]
}

func NewValidatorServer(blockChain *core.BlockChain) *ValidatorServer {
	return &ValidatorServer{blockChain: blockChain, list: lru.NewCache[int64, []string](5)}
}

func (v *ValidatorServer) NextBlockIs48Club(cur int64) bool {
	blockHeight := cur
	// currentEpochBlockHeight will be the most recent block height that can be module by 200
	currentEpochBlockHeight := blockHeight / bscBlocksPerEpoch * bscBlocksPerEpoch
	previousEpochBlockHeight := currentEpochBlockHeight - bscBlocksPerEpoch

	currentEpochValidatorList, ok := v.list.Get(currentEpochBlockHeight)
	if !ok {
		return false
	}
	preEpochValidatorList, ok := v.list.Get(previousEpochBlockHeight)
	if !ok {
		return false
	}

	// 计算 未来的 验证者
	if len(currentEpochValidatorList) != 0 && len(preEpochValidatorList) != 0 {
		targetingBlockHeight := blockHeight + 1
		activationIndex := int64((len(preEpochValidatorList)/2+1)*4 - 1) // activationIndex = ceiling[ N / 2 ] where N = the length of previous validator list, it marks a watershed. To the leftward we use previous validator list, to the rightward(inclusive) we use current validator list. Reference: https://github.com/bnb-chain/docs-site/blob/master/docs/smart-chain/guides/concepts/consensus.md

		index := targetingBlockHeight - currentEpochBlockHeight

		var validatorAddr string
		if index > activationIndex {
			listIndex := targetingBlockHeight / 4 % int64(len(currentEpochValidatorList))
			validatorAddr = currentEpochValidatorList[listIndex]
		} else { // use list from previous epoch
			listIndex := targetingBlockHeight / 4 % int64(len(preEpochValidatorList))
			validatorAddr = preEpochValidatorList[listIndex]
		}
		_, exist := club48Validator[strings.ToLower(validatorAddr)]
		if exist {
			return true
		} else {
			return false
		}
	}
	return false
}

func (v *ValidatorServer) NextBlock(cur int64) string {
	blockHeight := cur
	// currentEpochBlockHeight will be the most recent block height that can be module by 200
	currentEpochBlockHeight := blockHeight / bscBlocksPerEpoch * bscBlocksPerEpoch
	previousEpochBlockHeight := currentEpochBlockHeight - bscBlocksPerEpoch

	currentEpochValidatorList, ok := v.list.Get(currentEpochBlockHeight)
	if !ok {
		return ""
	}
	preEpochValidatorList, ok := v.list.Get(previousEpochBlockHeight)
	if !ok {
		return ""
	}

	// 计算 未来的 验证者
	if len(currentEpochValidatorList) != 0 && len(preEpochValidatorList) != 0 {
		targetingBlockHeight := blockHeight + 1
		activationIndex := int64((len(preEpochValidatorList)/2+1)*4 - 1) // activationIndex = ceiling[ N / 2 ] where N = the length of previous validator list, it marks a watershed. To the leftward we use previous validator list, to the rightward(inclusive) we use current validator list. Reference: https://github.com/bnb-chain/docs-site/blob/master/docs/smart-chain/guides/concepts/consensus.md

		index := targetingBlockHeight - currentEpochBlockHeight

		var validatorAddr string
		if index > activationIndex {
			listIndex := targetingBlockHeight / 4 % int64(len(currentEpochValidatorList))
			validatorAddr = currentEpochValidatorList[listIndex]
		} else { // use list from previous epoch
			listIndex := targetingBlockHeight / 4 % int64(len(preEpochValidatorList))
			validatorAddr = preEpochValidatorList[listIndex]
		}
		return validatorAddr
	}
	return ""
}

func (v *ValidatorServer) ServerName() string {
	return "validatorServer"
}

func (v *ValidatorServer) MsgAction(ctx context.Context, msg interface{}, num int) (resp interface{}, err error) {
	return nil, nil
}

func (v *ValidatorServer) ActionGoroutineNum() int {
	return 0
}

var bscBlocksPerEpoch int64 = 200

func (v *ValidatorServer) Schedule() []ms.TimedTask {

	return []ms.TimedTask{{
		Task: func(num int) {
			if v.blockNum == 0 {
				header := v.blockChain.CurrentBlock()
				if header != nil {
					cur := header.Number.Int64()
					currentHeight := cur / bscBlocksPerEpoch * bscBlocksPerEpoch
					previousHeight := currentHeight - bscBlocksPerEpoch

					currentHeader := v.blockChain.GetHeaderByNumber(uint64(currentHeight))
					previousHeader := v.blockChain.GetHeaderByNumber(uint64(previousHeight))

					currentList, err := bscExtractValidatorListFromBlock(currentHeader.Extra)
					if err == nil {
						v.blockNum = currentHeight
						v.list.Add(currentHeight, currentList)
					}
					previousList, err := bscExtractValidatorListFromBlock(previousHeader.Extra)
					if err == nil {
						v.list.Add(previousHeight, previousList)
					}
				}
			} else {
				curNumber := v.blockNum + 200
				blockheader := v.blockChain.GetHeaderByNumber(uint64(curNumber))
				if blockheader != nil {
					list, err := bscExtractValidatorListFromBlock(blockheader.Extra)
					if err != nil {
						return
					}

					v.list.Add(curNumber, list)
					v.blockNum += 200
				}
			}
		},
		Time: 1 * time.Second,
	}}
}

func (v *ValidatorServer) SetServer(s *ms.Server) {
	v.Server = s
}

func bscExtractValidatorListFromBlock(b []byte) ([]string, error) {
	addressLength := 20
	bLSPublicKeyLength := 48

	// follow order in extra field, from Luban upgrade, https://github.com/bnb-chain/bsc/commit/c208d28a68c414541cfaf2651b7cff725d2d3221
	// |---Extra Vanity---|---Validators Number and Validators Bytes (or Empty)---|---Vote Attestation (or Empty)---|---Extra Seal---|
	extraVanityLength := 32  // Fixed number of extra-data prefix bytes reserved for signer vanity
	validatorNumberSize := 1 // Fixed number of extra prefix bytes reserved for validator number after Luban
	validatorBytesLength := addressLength + bLSPublicKeyLength
	extraSealLength := 65 // Fixed number of extra-data suffix bytes reserved for signer seal

	// 32 + 65 + 1
	if len(b) < 98 {
		return nil, errors.New("wrong extra data, too small")
	}

	data := b[extraVanityLength : len(b)-extraSealLength]
	dataLength := len(data)

	// parse Validators and Vote Attestation
	if dataLength > 0 {
		// parse Validators
		if data[0] != '\xf8' { // rlp format of attestation begin with 'f8'
			validatorNum := int(data[0])
			validatorBytesTotalLength := validatorNumberSize + validatorNum*validatorBytesLength
			if dataLength < validatorBytesTotalLength {
				return nil, fmt.Errorf("parse validators failed, validator list is not aligned")
			}

			validatorList := make([]string, 0, validatorNum)
			data = data[validatorNumberSize:]
			for i := 0; i < validatorNum; i++ {

				// Ensure that the slice boundaries are within range
				startIndex := i * validatorBytesLength
				endIndex := startIndex + addressLength
				if endIndex > len(data) {
					return nil, fmt.Errorf("parse validators failed, validator data out of bounds")
				}

				validatorAddr := common.BytesToAddress(data[startIndex:endIndex])
				validatorList = append(validatorList, validatorAddr.String())
			}

			return validatorList, nil
		}
	}

	return nil, errors.New("wrong extra data, validatorList len is zero")
}
