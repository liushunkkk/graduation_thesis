package sse_transfer

import (
	"context"
	"fmt"
	"github.com/redis/go-redis/v9"
	"net/http"
	"time"
)

var rdb *redis.Client
var clients = make(map[chan string]struct{})

func streamHandler(w http.ResponseWriter, r *http.Request) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "Streaming unsupported", 500)
		return
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	ch := make(chan string, 10000)
	clients[ch] = struct{}{}
	defer func() { delete(clients, ch); close(ch) }()

	ticker := time.NewTicker(15 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case m := <-ch:
			fmt.Fprintf(w, "data: %s\n\n", m)
			flusher.Flush()
		case <-ticker.C:
			fmt.Fprint(w, ": ping\n\n")
			flusher.Flush()
		case <-r.Context().Done():
			return
		}
	}
}

func Init() {
	rdb = redis.NewClient(&redis.Options{Addr: "localhost:6379"})
	pubsub := rdb.Subscribe(context.Background(), "hints")
	go func() {
		ch := pubsub.Channel()
		for msg := range ch {
			for c := range clients {
				c <- msg.Payload
			}
		}
	}()

	http.HandleFunc("/stream", streamHandler)
}
