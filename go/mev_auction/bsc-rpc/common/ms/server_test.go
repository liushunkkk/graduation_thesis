package ms

import (
	"context"
	"fmt"
	"github.com/ethereum/go-ethereum/common"
	. "github.com/smartystreets/goconvey/convey"
	"github.com/stretchr/testify/assert"
	"testing"
	"time"
)

func TestNewSvr(t *testing.T) {

	t.Run("server-no-name", func(t *testing.T) {
		s, _ := NewSvr("", func(ctx context.Context, msg interface{}, num int) (resp interface{}, err error) {
			return msg, nil
		}, []TimedTask{{
			Task: func(num int) {
				return
			},
			Time: 0,
		}})

		s.Go()
		s.PushMsgToServer(context.Background(), "1")
		resp, err := s.PostMsgToServer(context.Background(), "2")
		assert.Nil(t, err)
		assert.Equal(t, resp, "2")
		s.Stop()
	})

	t.Run("server-name", func(t *testing.T) {
		s, _ := NewSvr("server", func(ctx context.Context, msg interface{}, num int) (resp interface{}, err error) {
			return msg, nil
		}, []TimedTask{{
			Task: func(num int) {
				return
			},
			Time: 0,
		}})

		s.Go()
		PushMsgToServer(context.Background(), "server", "1")
		resp, err := PostMsgToServer(context.Background(), "server", "2")
		assert.Nil(t, err)
		assert.Equal(t, resp, "2")
		StopServer("server")
	})

	t.Run("lastMsg-mode", func(t *testing.T) {
		var total int
		s, _ := NewSvr("lastMsg-mode-server", func(ctx context.Context, msg interface{}, num int) (resp interface{}, err error) {
			total++
			return msg, nil
		}, []TimedTask{{
			Task: func(num int) {
				return
			},
			Time: 0,
		}}, WithOptionLastOnly())

		s.Go()
		_, err := s.PostMsgToServer(context.Background(), "1")
		assert.NotNil(t, err)

		for i := 0; i < 10; i++ {
			s.PushMsgToServer(context.Background(), i)
		}
		time.Sleep(10 * time.Millisecond)
		assert.LessOrEqual(t, total, 10)
		s.Stop()
	})

	t.Run("timeout", func(t *testing.T) {
		s, _ := NewSvr("timeout", func(ctx context.Context, msg interface{}, num int) (resp interface{}, err error) {
			time.Sleep(20 * time.Millisecond)
			return msg, nil
		}, []TimedTask{{
			Task: func(num int) {
				return
			},
			Time: 0,
		}})

		s.Go()
		ctx, _ := context.WithTimeout(context.Background(), 10*time.Millisecond)
		_, err := s.PostMsgToServer(ctx, "1")
		assert.NotNil(t, err)
		s.Stop()
	})
}

func TestNewSvr1(t *testing.T) {
	Convey("server name repeated", t, func() {
		a, err := NewSvr("a", nil, nil)
		So(a, ShouldNotBeNil)
		So(err, ShouldBeNil)

		b, err := NewSvr("a", nil, nil)
		So(b, ShouldBeNil)
		So(err, ShouldNotBeNil)
	})
}

type BuilderParam struct {
	Param      *Param `json:"param"`
	BundleHash common.Hash
}

type Param struct {
	Txs               []string        `json:"txs"`
	MaxBlockNumber    uint64          `json:"maxBlockNumber,omitempty"`
	BlockNumber       string          `json:"blockNumber,omitempty"`
	MinTimestamp      uint64          `json:"minTimestamp"`
	MaxTimestamp      uint64          `json:"maxTimestamp"`
	RevertingTxHashes []string        `json:"revertingTxHashes"`
	BlockrazorOpts    *BlockrazorOpts `json:"blockrazorOpts,omitempty"`
}
type BlockrazorOpts struct {
	Version       string `json:"version"`
	SendCount     uint64 `json:"sendCount"`
	PrivacyPeriod uint32 `json:"privacyPeriod"`
	IsPrivate     bool   `json:"isPrivate"`
	RPCID         string `json:"rpcID"`
}

func TestBuilderParam(t *testing.T) {
	p := &Param{
		Txs:               []string{"111"},
		MaxTimestamp:      11,
		MinTimestamp:      11,
		RevertingTxHashes: []string{"11"},
		BlockrazorOpts: &BlockrazorOpts{
			SendCount:     1,
			PrivacyPeriod: 1,
			IsPrivate:     true,
		},
	}
	bp := &BuilderParam{
		Param:      p,
		BundleHash: common.Hash{},
	}
	fmt.Println("begin", bp.Param.BlockrazorOpts.SendCount)
	server, _ := NewSvr("a", func(ctx context.Context, msg interface{}, num int) (resp interface{}, err error) {
		ddd := msg.(*BuilderParam)
		fmt.Println("inner", ddd.Param.BlockrazorOpts.SendCount)
		return nil, nil
	}, nil)
	server.Go()
	server.PushMsgToServer(context.Background(), bp)
	time.Sleep(3 * time.Second)
	fmt.Println("after", bp.Param.BlockrazorOpts.SendCount)

}
