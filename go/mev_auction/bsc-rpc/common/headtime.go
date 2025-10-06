package common

import (
	"fmt"
	"net"
	"net/http"
	"strings"
	"sync/atomic"
	"time"
)

var HeadTime atomic.Uint64
var HealthChecker HealthCheck
var BackUp = "186.233.185.53"
var IsBackUp bool

type HealthCheck struct {
}

func (p *HealthCheck) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	ip, _ := getPublicIP()
	if IsBackUp || ip == BackUp {
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("ok"))
		return
	}

	load := HeadTime.Load()
	if uint64(time.Now().Unix())-load > 12 {
		w.WriteHeader(http.StatusServiceUnavailable)
		w.Write([]byte("Service is not ready"))
		return
	}
	w.WriteHeader(http.StatusOK)
	w.Write([]byte("ok"))
}

func getPublicIP() (string, error) {
	addrs, err := net.InterfaceAddrs()
	if err != nil {
		return "", err
	}

	for _, addr := range addrs {
		// 检查是否是 IP 地址，跳过 Loopback 地址
		if ipNet, ok := addr.(*net.IPNet); ok && !ipNet.IP.IsLoopback() {
			ip := ipNet.IP.To4()
			if ip != nil && !isPrivateIP(ip.String()) {
				return ip.String(), nil
			}
		}
	}
	return "", fmt.Errorf("no public IP address found")
}

// 判断是否是内网 IP
func isPrivateIP(ip string) bool {
	privateBlocks := []string{
		"10.",      // 10.0.0.0 - 10.255.255.255
		"172.16.",  // 172.16.0.0 - 172.31.255.255
		"192.168.", // 192.168.0.0 - 192.168.255.255
	}

	for _, block := range privateBlocks {
		if strings.HasPrefix(ip, block) {
			return true
		}
	}
	return false
}
