package mevshare

import (
	"github.com/ethereum/go-ethereum/params"
	"math/big"
	"strings"
)

var (
	ethDivisor  = new(big.Float).SetUint64(params.Ether)
	gweiDivisor = new(big.Float).SetUint64(params.GWei)

	big1  = big.NewInt(1)
	big10 = big.NewInt(10)
)

func formatUnits(value *big.Int, unit string) string {
	float := new(big.Float).SetInt(value)
	switch unit {
	case "eth":
		return float.Quo(float, ethDivisor).String()
	case "gwei":
		return float.Quo(float, gweiDivisor).String()
	default:
		return ""
	}
}

// Intersect returns the intersection of two string arrays, without duplicates
func Intersect(a, b []string) []string {
	if len(a) == 0 || len(b) == 0 {
		return []string{}
	}
	m := make(map[string]bool)
	for _, v := range a {
		v = strings.ToLower(v)
		m[v] = true
	}
	ret := []string{}
	for _, v := range b {
		v = strings.ToLower(v)
		if m[v] {
			ret = append(ret, v)
		}
	}
	return ret
}

// RoundUpWithPrecision rounds number up leaving only precisionDigits non-zero
// examples:
// RoundUpWithPrecision(123456, 2) = 130000
// RoundUpWithPrecision(111, 2) = 120
// RoundUpWithPrecision(199, 2) = 120
func RoundUpWithPrecision(number *big.Int, precisionDigits int) *big.Int {
	// Calculate the number of digits in the input number
	numDigits := len(number.String())

	if numDigits <= precisionDigits {
		return new(big.Int).Set(number)
	}
	// Calculate the power of 16 for the number of zero digits
	power := big.NewInt(int64(numDigits - precisionDigits))
	power.Exp(big10, power, nil)

	// Divide the number by the power of 16 to remove the extra digits
	div := new(big.Int).Div(number, power)

	// If the original number is not a multiple of the power of 16, round up
	if new(big.Int).Mul(div, power).Cmp(number) != 0 {
		div.Add(div, big1)
	}

	// Multiply the result by the power of 16 to add the extra digits back
	result := div.Mul(div, power)

	return result
}

func newerInclusion(old, newBundle *SendMevBundleArgs) bool {
	if old == nil {
		return true
	}
	if newBundle == nil {
		return false
	}
	if old.Inclusion.MaxBlock < newBundle.Inclusion.MaxBlock {
		return true
	}

	return false
}
