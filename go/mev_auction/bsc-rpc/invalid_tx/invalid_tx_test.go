package invalid_tx

import (
	"context"
	"fmt"
	"github.com/ethereum/go-ethereum/common"
	"testing"
	"time"
)

func TestServer_MsgAction(t *testing.T) {
	s := New()
	s.Start()

	s.PushMsgToServer(context.Background(), &TxMsg{
		Type:   0,
		TxHash: common.HexToHash("0x1234"),
	})

	resp, err := s.PostMsgToServer(context.Background(), &TxMsg{
		Type:   2,
		TxHash: common.HexToHash("0x1234"),
	})
	time.Sleep(2 * time.Second)
	fmt.Println(resp, err)
	s.Destroy()
}
