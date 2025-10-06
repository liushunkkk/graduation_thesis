package limitip

import (
	"golang.org/x/time/rate"
	"sync"
)

var ipLimiters = struct {
	sync.Mutex
	clients map[string]*rate.Limiter
}{clients: make(map[string]*rate.Limiter)}

var LimitCount = 50

func GetLimiter(ip string) *rate.Limiter {
	ipLimiters.Lock()
	defer ipLimiters.Unlock()
	if limiter, exists := ipLimiters.clients[ip]; exists {
		return limiter
	}
	limiter := rate.NewLimiter(rate.Limit(LimitCount), LimitCount) // 每秒 50 请求，最大突发 50
	ipLimiters.clients[ip] = limiter
	return limiter
}
