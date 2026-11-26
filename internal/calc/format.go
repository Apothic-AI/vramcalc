package calc

import (
	"math"
	"math/big"
	"strconv"
	"strings"
)

// JSToFixed formats a finite, non-negative x with the given number of decimals,
// the way JavaScript's Number.prototype.toFixed does. An exact tie rounds up,
// so 2.5 becomes "3" and 0.25 becomes "0.3", where strconv would round to even.
func JSToFixed(x float64, digits int) string {
	scale := int64(1)
	for i := 0; i < digits; i++ {
		scale *= 10
	}
	// 2048 bits is far more than x*10^digits+0.5 needs, so floor is exact.
	v := new(big.Float).SetPrec(2048).SetFloat64(x)
	v.Mul(v, new(big.Float).SetPrec(2048).SetInt64(scale))
	v.Add(v, big.NewFloat(0.5))
	n, _ := v.Int(nil)
	s := n.String()
	if digits == 0 {
		return s
	}
	if len(s) <= digits {
		s = strings.Repeat("0", digits-len(s)+1) + s
	}
	return s[:len(s)-digits] + "." + s[len(s)-digits:]
}

// FormatGiB mirrors fmtGB in the Space: one decimal below 10, whole numbers
// from 10 up. The unit is written GiB because the Space's README says its
// figures are GiB.
func FormatGiB(x float64) string {
	if x < 10 {
		return JSToFixed(x, 1) + " GiB"
	}
	return strconv.FormatFloat(math.Floor(x+0.5), 'f', 0, 64) + " GiB"
}

// FormatParams mirrors fmtParams in the Space: billions with one decimal, or
// millions for anything under a billion.
func FormatParams(params float64) string {
	if params >= 1e9 {
		return JSToFixed(params/1e9, 1) + "B"
	}
	return JSToFixed(params/1e6, 0) + "M"
}
