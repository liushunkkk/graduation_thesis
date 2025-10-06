package portal

import "github.com/zeromicro/go-zero/core/stat"

var Address string = "54.89.163.74:7002"
var ApiHealth []string

func init() {
	stat.DisableLog()
}
