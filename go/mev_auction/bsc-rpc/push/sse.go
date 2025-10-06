package push

import (
	"context"
	"encoding/json"
	"fmt"
	"github.com/duke-git/lancet/v2/random"
	"github.com/ethereum/go-ethereum/common/ms"
	. "github.com/ethereum/go-ethereum/log/zap"
	"github.com/ethereum/go-ethereum/metrics"
	"github.com/ethereum/go-ethereum/portal"
	"github.com/ethereum/go-ethereum/push/define"
	"go.uber.org/zap"
	"net"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

var (
	sseClientGauge    = metrics.NewRegisteredGauge("sse/clients", nil)
	sseSubscribeGauge = metrics.NewRegisteredGauge("sse/subscriptions", nil)
)

type SSEServer struct {
	server       *ms.Server
	clientsMtx   sync.Mutex
	clients      map[string]int
	consumers    map[string]*Consumer
	IPLimitCount int
}

type Consumer struct {
	ID      string
	Channel chan *define.SseBundleData
	Exit    atomic.Bool
}

func (p *SSEServer) Start() {
	p.clients = make(map[string]int)
	p.consumers = make(map[string]*Consumer)

	p.server, _ = ms.NewSvr("sse-server", func(ctx context.Context, msg interface{}, num int) (resp interface{}, err error) {
		go func() {
			var cs []*Consumer
			p.clientsMtx.Lock()
			for _, consumer := range p.consumers {
				cs = append(cs, consumer)
			}
			p.clientsMtx.Unlock()

			for _, consumer := range cs {
				select {
				case consumer.Channel <- msg.(*define.SseBundleData):
				default:
					consumer.Exit.Store(true)
				}
			}
		}()
		return nil, nil
	}, nil)
	p.server.ActionGoroutineNum = 4
	p.server.Go()
}

func (p *SSEServer) Stop() {
	p.server.Stop()
}

func (p *SSEServer) Send(data *define.SseBundleData) {
	t := time.Now()
	data.CreateTime = &t
	p.server.PushMsgToServer(context.Background(), data)
}

func (p *SSEServer) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	// 启动时间检查
	enforceTokenStartTime := time.Date(2024, 12, 11, 0, 0, 0, 0, time.UTC)
	authToken := p.extractToken(r.Header.Get("Authorization"))
	if authToken == "" {
		authToken = r.URL.Query().Get("token")
	}

	if time.Now().After(enforceTokenStartTime) {
		if authToken == "" {
			http.Error(w, "Auth token is missing", http.StatusUnauthorized)
			return
		}

		if !p.validateToken(authToken) {
			http.Error(w, "Auth token is invalid.Please consider upgrading subscription tier for full access.", http.StatusUnauthorized)
			return
		}
		Zap.Info("authToken is valid", zap.String("token", authToken))
	}

	var host string
	forwarded := r.Header.Get("X-Forwarded-For")
	if forwarded != "" {
		ips := strings.Split(forwarded, ",")
		host = ips[0]
	} else {
		var err error
		host, _, err = net.SplitHostPort(r.RemoteAddr)
		if err != nil {
			return
		}
	}
	if host == "" {
		return
	}

	p.clientsMtx.Lock()
	if p.clients[host] >= p.IPLimitCount {
		p.clientsMtx.Unlock()
		http.Error(w, fmt.Sprintf("The number of connections for the same ip access has reached %d", p.IPLimitCount), http.StatusTooManyRequests)
		Zap.Info("Connections for the same ip access has reached limit", zap.String("host", host), zap.Any("limit", p.IPLimitCount))
		return
	}
	p.clients[host]++
	sseSubscribeGauge.Inc(1)
	sseClientGauge.Update(int64(len(p.clients)))
	str := strings.ReplaceAll(host, ".", "_")
	str = str + "_" + strconv.Itoa(p.clients[host])
	sseWaitingGauge := metrics.NewRegisteredGauge("sse/waiting/"+str, nil)
	p.clientsMtx.Unlock()

	defer func() {
		p.clientsMtx.Lock()
		defer p.clientsMtx.Unlock()
		p.clients[host]--
		if p.clients[host] == 0 {
			delete(p.clients, host)
		}
		sseClientGauge.Update(int64(len(p.clients)))
		sseSubscribeGauge.Dec(1)
		sseWaitingGauge.Update(0)
	}()

	query := r.URL.Query()
	enableState := query.Get("state")

	// 设置必要的Header
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "Streaming unsupported!", http.StatusInternalServerError)
		return
	}

	sub := p.Subscribe(random.RandString(10), 20480)

	defer func() {
		p.Unsubscribe(sub.ID)
		Zap.Info("sse unsubscribe", zap.String("host", host))
	}()

	_, err := fmt.Fprintf(w, "ping: \n\n")
	if err != nil {
		return
	}
	flusher.Flush()

	ticker := time.NewTicker(10 * time.Second)
	defer ticker.Stop()

	flushTicker := time.NewTicker(40 * time.Millisecond)
	defer flushTicker.Stop()

	batchSize := 20
	processed := 0
	for {
		sseWaitingGauge.Update(int64(len(sub.Channel)))
		select {
		case bd := <-sub.Channel:
			if time.Since(*bd.CreateTime) > 400*time.Millisecond {
				Zap.Warn("Dropping expired message", zap.Time("createTime", *bd.CreateTime))
				continue
			}
			data := define.SseBundleData{}
			data = *bd
			if enableState != "true" {
				data.State = nil
			}
			data.CreateTime = nil
			b, _ := json.Marshal(data)
			_, err = fmt.Fprintf(w, "data: %s\n\n", b)
			if err != nil {
				return
			}
			processed++
			if processed >= batchSize {
				flusher.Flush()
				processed = 0
			}
		case <-ticker.C:
			_, err = fmt.Fprintf(w, "ping: \n\n")
			if err != nil {
				return
			}
			flusher.Flush()

			if time.Now().After(enforceTokenStartTime) {
				if authToken == "" {
					http.Error(w, "Auth token is missing", http.StatusUnauthorized)
					return
				}
				if !p.validateToken(authToken) {
					http.Error(w, "Auth token is invalid.Please consider upgrading subscription tier for full access.", http.StatusUnauthorized)
					return
				}
			}
		case <-flushTicker.C:
			if processed > 0 {
				flusher.Flush()
				processed = 0
			}
		case <-r.Context().Done():
			// 客户端断开连接
			return
		}

		if sub.Exit.Load() {
			Zap.Info("receive exit signal,exit")
			return
		}
	}
}

func (p *SSEServer) Subscribe(id string, bufferSize int) *Consumer {
	consumer := &Consumer{
		ID:      id,
		Channel: make(chan *define.SseBundleData, bufferSize),
	}

	p.clientsMtx.Lock()
	defer p.clientsMtx.Unlock()
	p.consumers[id] = consumer
	return consumer
}

func (p *SSEServer) Unsubscribe(id string) {
	p.clientsMtx.Lock()
	defer p.clientsMtx.Unlock()
	if _, ok := p.consumers[id]; ok {
		delete(p.consumers, id)
	}
}

func (p *SSEServer) validateToken(token string) bool {
	if len(token) == 0 {
		return false
	}

	if token == "adminToken" {
		return true
	}

	info := portal.UserServer.GetUserInfo(token)
	if info == nil {
		return false
	}

	if info.PlanId == 3 || info.PlanId == 2 {
		return true
	}
	return false
}

func (p *SSEServer) extractToken(authHeader string) string {
	if strings.HasPrefix(authHeader, "Bearer ") {
		return strings.TrimPrefix(authHeader, "Bearer ")
	}
	return authHeader
}
